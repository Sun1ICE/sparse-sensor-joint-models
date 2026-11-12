"""Sparse two-sensor joint response reconstruction and unbalance estimation.

Inputs and outputs use native g with the last axis [Re, Im]. The network never
receives U or missing responses. Both families share feature construction and
phase handling. ``forward`` retains finite internal predictions for exact-zero
observations so training may still supervise them; public inference masks those
unidentifiable predictions (see predict.py).
"""
from __future__ import annotations

import functools
import math

import torch
from torch import nn
from torch.nn import functional as F

from node_spec import ANGLES_DEG, CASING_IDS, N_NODES, U_MAX

RPM_MIN, RPM_MAX = 3005.0, 5995.0
MODEL_VERSION = 2
LOW_SIGNAL_NORMALIZED_AMPLITUDE = 1e-6
REFERENCE_TIE_RELATIVE_TOLERANCE = 1e-6


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _scale_tensor(scale_g):
    scale = torch.as_tensor(scale_g, dtype=torch.float32).detach().clone()
    if scale.shape != (2, 3) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("scale_g must contain six finite positive RMS values, shape [2,3]")
    return scale


def _geometry():
    angles = torch.tensor(ANGLES_DEG, dtype=torch.float32) * (math.pi / 180.0)
    casing = F.one_hot(torch.tensor(CASING_IDS), num_classes=2).float()
    nodes = torch.cat((angles.sin()[:, None], angles.cos()[:, None], casing), dim=-1)
    # [target,source,7]; delta = target angle - source angle.
    delta = angles[:, None] - angles[None, :]
    same = (torch.tensor(CASING_IDS)[:, None] == torch.tensor(CASING_IDS)[None, :]).float()
    edges = torch.cat((delta.sin()[..., None], delta.cos()[..., None], same[..., None],
                       casing[None, :, :].expand(N_NODES, -1, -1),
                       casing[:, None, :].expand(-1, N_NODES, -1)), dim=-1)
    return nodes, edges


class EdgeAttentionBlock(nn.Module):
    """Four-head attention over all 182 directed non-self edges."""

    def __init__(self, width=128, heads=4, *, use_edge_attributes=True, uniform=False):
        super().__init__()
        self.width, self.heads, self.head_width = width, heads, width // heads
        self.use_edge_attributes, self.uniform = use_edge_attributes, uniform
        self.attention_norm = nn.LayerNorm(width)
        if uniform:
            self.value = nn.Linear(width, width)
        else:
            self.qkv = nn.Linear(width, 3 * width)
        if use_edge_attributes:
            if not uniform:
                self.edge_bias = nn.Linear(7, heads, bias=False)
            self.edge_value = nn.Linear(7, width, bias=False)
        self.output = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
        self.register_buffer("self_edges", torch.eye(N_NODES, dtype=torch.bool), persistent=False)

    def forward(self, state, edges):
        batch, nodes, _ = state.shape
        normalized = self.attention_norm(state)
        if self.uniform:
            value = self.value(normalized).view(batch, nodes, self.heads, self.head_width).transpose(1, 2)
            weights = (~self.self_edges).to(value.dtype)[None, None] / (nodes - 1)
            weights = weights.expand(batch, self.heads, -1, -1)
        else:
            qkv = self.qkv(normalized).view(batch, nodes, 3, self.heads, self.head_width)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
            if self.use_edge_attributes:
                scores = scores + self.edge_bias(edges).permute(2, 0, 1)[None]
            scores = scores.masked_fill(self.self_edges[None, None], float("-inf"))
            # Accumulate softmax in FP32 even when the backbone is autocast BF16.
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        message = torch.matmul(weights, value)
        if self.use_edge_attributes:
            edge_values = self.edge_value(edges).view(nodes, nodes, self.heads, self.head_width).permute(2, 0, 1, 3)
            message = message + torch.einsum("bhij,hijd->bhid", weights, edge_values)
        state = state + self.output(message.transpose(1, 2).reshape(batch, nodes, self.width))
        return state + self.ff(self.ff_norm(state))


class JointModel(nn.Module):
    def __init__(self, scale_g, family, mlp_width=None, task="joint"):
        super().__init__()
        scale = _scale_tensor(scale_g)
        self.family = family
        self.task = task
        self.register_buffer("scale_g", scale)
        self.register_buffer("node_scale_g", scale[torch.tensor(CASING_IDS)])
        node_geometry, edge_geometry = _geometry()
        self.register_buffer("node_geometry", node_geometry)
        self.register_buffer("edge_geometry", edge_geometry)
        self.model_config = {"version": MODEL_VERSION, "family": family, "task": task,
                             "scale_g": scale.cpu().tolist(), "mlp_width": mlp_width}

    def _features(self, response_g, rpm, observed_mask, validate_inputs=True):
        if response_g.ndim != 4 or tuple(response_g.shape[1:]) != (N_NODES, 3, 2):
            raise ValueError("response_g must have shape [B,14,3,2]")
        batch = response_g.shape[0]
        if rpm.shape != (batch,) or observed_mask.shape != (batch, N_NODES):
            raise ValueError("rpm must have shape [B] and observed_mask shape [B,14]")
        if observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be boolean with exactly two observed nodes per sample")
        # Value checks read device memory back (a full CUDA sync each). Training
        # batches skip them: their data were verified at load, masks built two-hot.
        if validate_inputs and not torch.all(observed_mask.sum(1) == 2):
            raise ValueError("observed_mask must be boolean with exactly two observed nodes per sample")
        if validate_inputs and (not torch.isfinite(rpm).all() or torch.any((rpm < RPM_MIN) | (rpm > RPM_MAX))):
            raise ValueError(f"rpm must be finite and in [{RPM_MIN},{RPM_MAX}]")
        # Clear missing NaN/Inf BEFORE finite checks, abs, division, or rotation.
        observed = torch.where(observed_mask[:, :, None, None], response_g.float(), 0.0)
        if validate_inputs and not torch.isfinite(observed).all():
            raise ValueError("Observed acceleration must be finite")
        magnitude = torch.hypot(observed[..., 0], observed[..., 1])
        magnitude_flat = magnitude.flatten(1)
        maximum_magnitude = magnitude_flat.amax(dim=1)
        # Numerically tied maxima choose the first canonical node/direction.
        # This makes equal-amplitude references stable under FP32 rotations.
        # It does not make genuine reference-channel crossings continuous.
        reference_candidates = magnitude_flat >= maximum_magnitude[:, None] * (1.0 - REFERENCE_TIE_RELATIVE_TOLERANCE)
        reference_index = reference_candidates.to(torch.int64).argmax(dim=1)
        reference_magnitude = magnitude_flat.gather(1, reference_index[:, None]).squeeze(1)
        reference = observed.reshape(batch, -1, 2).gather(1, reference_index[:, None, None].expand(-1, 1, 2)).squeeze(1)
        valid = reference_magnitude > 0
        safe_magnitude = torch.where(valid, reference_magnitude, torch.ones_like(reference_magnitude))
        phase = reference / safe_magnitude[:, None]
        # Identity phase [1, 0] for zero observations, without a host upload.
        phase = torch.stack((torch.where(valid, phase[:, 0], 1.0), torch.where(valid, phase[:, 1], 0.0)), dim=-1)
        real, imag = observed.unbind(-1)
        c, s = phase[:, 0, None, None], phase[:, 1, None, None]
        canonical = torch.stack((real * c + imag * s, imag * c - real * s), dim=-1)
        normalized = canonical / self.node_scale_g[None, :, :, None]
        normalized_magnitude = magnitude / self.node_scale_g[None]
        rpm_feature = ((rpm.float() - RPM_MIN) * (2.0 / (RPM_MAX - RPM_MIN)) - 1.0)[:, None, None]
        features = torch.cat((normalized.flatten(2), normalized_magnitude,
                              observed_mask[:, :, None].float(), self.node_geometry[None].expand(batch, -1, -1),
                              rpm_feature.expand(-1, N_NODES, -1)), dim=-1)
        low_signal = normalized_magnitude.flatten(1).amax(1) < LOW_SIGNAL_NORMALIZED_AMPLITUDE
        flags = {"zero_observation": ~valid, "low_signal": low_signal & valid}
        return features, phase, valid, flags, reference_magnitude

    def forward(self, response_g, rpm, observed_mask, *, validate_inputs=True):
        with torch.autocast(device_type=response_g.device.type, enabled=False):
            features, phase, valid, flags, reference_magnitude = self._features(
                response_g, rpm, observed_mask, validate_inputs)
        normalized_response, u_logit = self._backbone(features)
        with torch.autocast(device_type=response_g.device.type, enabled=False):
            response_hat_g, unbalance_g_cm = None, None
            if normalized_response is not None:
                canonical_g = normalized_response.float() * self.node_scale_g[None, :, :, None]
                real, imag = canonical_g.unbind(-1)
                c, s = phase[:, 0, None, None], phase[:, 1, None, None]
                response_hat_g = torch.stack((real * c - imag * s, imag * c + real * s), dim=-1)
            if u_logit is not None:
                unbalance_g_cm = torch.sigmoid(u_logit.float()) * U_MAX
        return {"response_hat_g": response_hat_g, "unbalance_g_cm": unbalance_g_cm,
                "prediction_valid": valid, "quality_flags": flags,
                "reference_magnitude_g": reference_magnitude}


class JointGNN(JointModel):
    def __init__(self, scale_g, family="gnn"):
        tasks = {"gnn": "joint", "gnn_no_edge": "joint", "gnn_uniform": "joint",
                 "gnn_response_only": "response", "gnn_u_only": "u"}
        if family not in tasks:
            raise ValueError(f"Unknown GNN family: {family!r}")
        super().__init__(scale_g, family, task=tasks[family])
        self.node_encoder = nn.Sequential(nn.Linear(15, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.blocks = nn.ModuleList([EdgeAttentionBlock(use_edge_attributes=family != "gnn_no_edge",
                                                        uniform=family == "gnn_uniform") for _ in range(3)])
        self.response_head = (nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 6))
                              if self.task != "u" else None)
        self.unbalance_head = (nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
                               if self.task != "response" else None)
        self.model_config["architecture"] = {
            "node_encoder": [15, 128, 128], "width": 128, "blocks": 3, "heads": 4,
            "edge_attributes": family != "gnn_no_edge", "aggregation": "uniform_nonself" if family == "gnn_uniform" else "attention",
            "response_head": [128, 64, 6] if self.response_head is not None else None,
            "unbalance_head": [128, 64, 1] if self.unbalance_head is not None else None}

    def _backbone(self, features):
        state = self.node_encoder(features)
        for block in self.blocks:
            state = block(state, self.edge_geometry)
        response = self.response_head(state).reshape(-1, N_NODES, 3, 2) if self.response_head is not None else None
        unbalance = self.unbalance_head(state.mean(dim=1)).squeeze(-1) if self.unbalance_head is not None else None
        return response, unbalance


class JointDeepSets(JointModel):
    """Shared observed-node encoding, invariant mean pooling and target decoding."""

    def __init__(self, scale_g):
        super().__init__(scale_g, "deepsets")
        self.node_encoder = nn.Sequential(nn.Linear(15, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.response_head = nn.Sequential(nn.Linear(261, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 6))
        self.unbalance_head = nn.Sequential(nn.Linear(256, 384), nn.SiLU(), nn.Linear(384, 1))
        self.model_config["architecture"] = {
            "node_encoder": [15, 256, 256], "pooling": "observed_only_mean",
            "target_features": "node_geometry_4_plus_rpm_1", "response_head": [261, 512, 256, 6],
            "unbalance_head": [256, 384, 1]}

    def _backbone(self, features):
        observed = features[..., 9:10]
        state = self.node_encoder(features)
        pooled = (state * observed).sum(dim=1) / observed.sum(dim=1)
        target = torch.cat((pooled[:, None].expand(-1, N_NODES, -1), features[..., 10:15]), dim=-1)
        response = self.response_head(target).reshape(-1, N_NODES, 3, 2)
        unbalance = self.unbalance_head(pooled).squeeze(-1)
        return response, unbalance


def _mlp_parameter_count(width):
    return (210 * width + width) + 2 * (width * width + width) + (width * 85 + 85)


def matched_mlp_width():
    # This isolated RNG context avoids changing either model or data RNG streams.
    with torch.random.fork_rng(devices=[]):
        reference_count = trainable_parameters(JointGNN(torch.ones(2, 3)))
    width = min(range(8, 2049, 8), key=lambda w: (abs(_mlp_parameter_count(w) - reference_count), w))
    if abs(_mlp_parameter_count(width) / reference_count - 1.0) > 0.05:
        raise RuntimeError("Cannot match MLP parameter count within 5 percent")
    return width


class JointMLP(JointModel):
    def __init__(self, scale_g):
        width = matched_mlp_width()
        super().__init__(scale_g, "mlp", mlp_width=width)
        self.network = nn.Sequential(nn.Linear(210, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 85))
        self.model_config["architecture"] = {"network": [210, width, width, width, 85]}

    def _backbone(self, features):
        output = self.network(features.flatten(1))
        return output[:, :84].reshape(-1, N_NODES, 3, 2), output[:, 84]


def build_model(family="gnn", scale_g=None):
    if scale_g is None:
        raise ValueError("The frozen training scale_g is required")
    if family in ("gnn", "gnn_no_edge", "gnn_uniform", "gnn_response_only", "gnn_u_only"):
        return JointGNN(scale_g, family)
    if family == "mlp":
        return JointMLP(scale_g)
    if family == "deepsets":
        return JointDeepSets(scale_g)
    raise ValueError(f"Unknown model family: {family!r}")


def _group_average(error, selected):
    """Per sample, average equally across represented casing x direction groups.

    ``error`` is [B,14,3], ``selected`` [B,14]. A same-casing observed pair
    represents three observed groups, not six; absent groups contribute neither
    zero nor an artificial denominator. Missing groups always include both casings.
    """
    parts, represented = [], []
    for left, right in ((0, 7), (7, 14)):
        weights = selected[:, left:right, None].float()
        count = weights.sum(dim=1)
        parts.append((error[:, left:right] * weights).sum(dim=1) / count.clamp_min(1))
        represented.append((count > 0).expand(-1, 3))
    values, present = torch.cat(parts, dim=1), torch.cat(represented, dim=1)
    return (values * present).sum(dim=1) / present.sum(dim=1).clamp_min(1)


def _differentiable_hypot(real, imag):
    """``torch.hypot`` with a zero subgradient at the origin instead of 0/0 = NaN.

    BF16 outputs can be exactly (0, 0); even under a zero loss weight the NaN
    survives (0 * NaN). Values and all other gradients are bitwise unchanged.
    """
    origin = (real == 0) & (imag == 0)
    magnitude = torch.hypot(torch.where(origin, torch.ones_like(real), real), imag)
    return torch.where(origin, torch.zeros_like(magnitude), magnitude)


@functools.lru_cache(maxsize=None)
def _casing_index(device):
    # Uploaded once per device: a blocking host-to-CUDA copy per loss call stalls training.
    return torch.tensor(CASING_IDS, device=device)


def _loss_values(outputs, target_g, unbalance_g_cm, observed_mask, scale_g, *, auxiliary):
    """Only evaluate labels for available tasks; no zero times an absent loss."""
    response, unbalance = outputs.get("response_hat_g"), outputs.get("unbalance_g_cm")
    if response is None and unbalance is None:
        raise ValueError("At least one prediction task must be available")
    device = response.device if response is not None else unbalance.device
    parts = {}
    with torch.autocast(device_type=device.type, enabled=False):
        if response is not None:
            scale = torch.as_tensor(scale_g, dtype=torch.float32, device=device)[_casing_index(device)]
            prediction = response.float() / scale[None, :, :, None]
            target = target_g.float() / scale[None, :, :, None]
            complex_error = (prediction - target).square().sum(dim=-1)
            parts["missing_complex"] = _group_average(complex_error, ~observed_mask)
            if auxiliary:
                pred_magnitude = _differentiable_hypot(prediction[..., 0], prediction[..., 1])
                target_magnitude = torch.hypot(target[..., 0], target[..., 1])
                amplitude_error = (pred_magnitude - target_magnitude).square()
                parts["missing_amplitude"] = _group_average(amplitude_error, ~observed_mask)
                parts["observed_complex"] = _group_average(complex_error, observed_mask)
        if unbalance is not None:
            parts["unbalance"] = ((unbalance.float() - unbalance_g_cm.float()) / U_MAX).square()
    return parts


def loss_components(outputs, target_g, unbalance_g_cm, observed_mask, scale_g):
    """Mean components for supported tasks only (absent task keys are omitted)."""
    return {name: values.mean() for name, values in
            _loss_values(outputs, target_g, unbalance_g_cm, observed_mask, scale_g, auxiliary=True).items()}


def training_loss(outputs, target_g, unbalance_g_cm, observed_mask, scale_g):
    parts = loss_components(outputs, target_g, unbalance_g_cm, observed_mask, scale_g)
    weights = {"missing_complex": 1., "missing_amplitude": .1, "observed_complex": .1, "unbalance": 1.}
    return sum(weights[name] * value for name, value in parts.items())


def selection_values(outputs, target_g, unbalance_g_cm, observed_mask, scale_g):
    """Per-sample validation objective: supported primary tasks, equal groups."""
    parts = _loss_values(outputs, target_g, unbalance_g_cm, observed_mask, scale_g, auxiliary=False)
    return sum(parts.values())


def selection_score(outputs, target_g, unbalance_g_cm, observed_mask, scale_g):
    return selection_values(outputs, target_g, unbalance_g_cm, observed_mask, scale_g).mean()
