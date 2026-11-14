"""Training-data-only piecewise-linear complex surrogate with phase-profiled U search.

The grid is calibrated exclusively from train.npz. No validation response or
original PCHIP generator enters the surrogate. Search checks every training U
knot and refines intervals neighbouring the best eight knots. This is a bounded
approximate search, not a claimed global optimum of the continuous objective.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from node_spec import CASING_IDS, U_MAX


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ClassicalInverse:
    task = "joint"
    family = "classical_linear_profile"

    def __init__(self, root, scale_g, *, top_knots=8, refinement_steps=24, chunk_size=32):
        if top_knots < 1 or refinement_steps < 0 or chunk_size < 1:
            raise ValueError("Expected positive knot and chunk counts and nonnegative refinement steps")
        root = Path(root)
        with np.load(root / "data/train.npz", allow_pickle=False) as archive:
            values = archive["response_g"].astype(np.float32)
            u = archive["unbalance_g_cm"].astype(np.float64)
            rpm = archive["rpm"].astype(np.float64)
        if u.ndim != 1 or rpm.ndim != 1:
            raise ValueError("Calibration axes must be one-dimensional")
        if values.shape != (len(u), len(rpm), 14, 3, 2):
            raise ValueError("Classical calibration grid shape differs")
        if not np.isfinite(values).all() or not np.isfinite(u).all() or not np.isfinite(rpm).all():
            raise ValueError("Nonfinite classical calibration data")
        order = np.argsort(u)
        self.u, self.rpm = u[order], rpm
        self.grid = (values[order, ..., 0] + 1j * values[order, ..., 1]).astype(np.complex64)
        if len(u) < 2 or len(rpm) < 2 or not (np.diff(self.u) > 0).all() or not (np.diff(rpm) > 0).all():
            raise ValueError("Strictly increasing calibration axes required")
        if self.u[0] != 0 or abs(self.u[-1] - U_MAX) > .001:
            raise ValueError("Training calibration must span the full supported U domain")
        scale = scale_g.detach().cpu().numpy() if hasattr(scale_g, "detach") else np.asarray(scale_g)
        if scale.shape != (2, 3) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Six positive training-only RMS scales required")
        self.scale = scale[np.asarray(CASING_IDS)]
        self.top_knots = min(int(top_knots), len(u))
        self.refinement_steps, self.chunk_size = int(refinement_steps), int(chunk_size)
        self.provenance = {"calibration": "data/train.npz", "train_sha256": _file_sha256(root / "data/train.npz"),
                           "calibration_u_knots": len(u), "calibration_rpm_knots": len(rpm),
                           "interpolation": "piecewise linear in RPM and U; all 14 complex three-axis responses",
                           "phase": "analytic weighted least squares common phase profile",
                           "weight": "six training-only casing/direction RMS values",
                           "search": "all training U knots, both adjacent intervals of best knots, golden refinement",
                           "top_knots": self.top_knots, "refinement_steps": self.refinement_steps,
                           "guarantee": "approximate bounded search; continuous global optimum is not guaranteed",
                           "zero_signal": "exact all-zero observed channels invalid; nonzero weak signals retained with low_signal flag",
                           "unidentifiable_phase": "zero profiled correlation flags common_phase_unidentifiable and invalid response",
                           "uses_original_pchip_generator": False, "uses_validation_for_calibration": False}

    def _rpm_grid(self, rpm):
        lo = np.clip(np.searchsorted(self.rpm, rpm, side="right") - 1, 0, len(self.rpm) - 2)
        fraction = (rpm - self.rpm[lo]) / (self.rpm[lo + 1] - self.rpm[lo])
        return self.grid[:, lo] * (1 - fraction[None, :, None, None]) + self.grid[:, lo + 1] * fraction[None, :, None, None]

    @staticmethod
    def _profile(predicted, observed):
        # Arrays [..., six observed complex channels]. phase maps predicted -> observed.
        correlation = np.sum(np.conj(predicted) * observed, axis=-1)
        cost = np.sum(abs(predicted) ** 2, axis=-1) + np.sum(abs(observed) ** 2, axis=-1) - 2 * abs(correlation)
        phase = np.divide(correlation, abs(correlation), out=np.ones_like(correlation), where=abs(correlation) > 0)
        phase = np.where(abs(correlation) > 0, phase, np.nan + 1j * np.nan)
        return np.maximum(cost, 0.), phase

    def _solve(self, observed, rpm, mask):
        n = len(rpm)
        nodes = np.argsort(~mask, axis=1)[:, :2]
        grid = self._rpm_grid(rpm)  # [K,B,14,3], float64 complex math for search.
        columns = np.arange(n)
        obs_scale = self.scale[nodes].reshape(n, 6)
        selected = grid[:, columns[:, None], nodes].reshape(len(self.u), n, 6) / obs_scale[None]
        z = observed[columns[:, None], nodes].reshape(n, 6) / obs_scale
        coarse, _ = self._profile(selected, z[None])
        top = np.argsort(coarse, axis=0)[:self.top_knots].T
        interval = np.concatenate((top - 1, top), axis=1).clip(0, len(self.u) - 2)
        lower, upper = self.u[interval], self.u[interval + 1]
        left_values = selected[interval, columns[:, None]]
        delta = selected[interval + 1, columns[:, None]] - left_values
        span = upper - lower
        def evaluate(u):
            predicted = left_values + ((u - self.u[interval]) / span)[..., None] * delta
            return self._profile(predicted, z[:, None])[0]
        ratio = (np.sqrt(5.) - 1.) / 2.
        left, right = lower.copy(), upper.copy()
        c, d = right - ratio * (right - left), left + ratio * (right - left)
        fc, fd = evaluate(c), evaluate(d)
        for _ in range(self.refinement_steps):
            choose = fc <= fd
            right = np.where(choose, d, right)
            left = np.where(choose, left, c)
            c, d = right - ratio * (right - left), left + ratio * (right - left)
            fc, fd = evaluate(c), evaluate(d)
        trial_u = (left + right) / 2
        trial_cost = evaluate(trial_u)
        # Exact projection when an interval starts at the zero response anchor.
        # This retains arbitrarily weak nonzero signals without an absolute search floor.
        numerator = abs(np.sum(np.conj(delta) * z[:, None], axis=-1))
        denominator = np.sum(abs(delta) ** 2, axis=-1)
        projection = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).clip(0, 1)
        projected_u = lower + projection * span
        projected_cost = evaluate(projected_u)
        use_projection = (np.sum(abs(left_values) ** 2, axis=-1) == 0) & (projected_cost < trial_cost)
        trial_u = np.where(use_projection, projected_u, trial_u)
        trial_cost = np.where(use_projection, projected_cost, trial_cost)
        best_interval = trial_cost.argmin(axis=1)
        uhat = trial_u[columns, best_interval]
        best_coarse = coarse.argmin(axis=0)
        use_coarse = coarse[best_coarse, columns] <= trial_cost[columns, best_interval]
        uhat = np.where(use_coarse, self.u[best_coarse], uhat)
        idx = np.clip(np.searchsorted(self.u, uhat, side="right") - 1, 0, len(self.u) - 2)
        frac = (uhat - self.u[idx]) / (self.u[idx + 1] - self.u[idx])
        response = grid[idx, columns] * (1 - frac[:, None, None]) + grid[idx + 1, columns] * frac[:, None, None]
        observed_prediction = response[columns[:, None], nodes].reshape(n, 6) / obs_scale
        _, phase = self._profile(observed_prediction, z)
        return response * phase[:, None, None], uhat, np.isfinite(phase)

    def __call__(self, response_g, rpm, mask):
        device = response_g.device
        x, r, m = (v.detach().cpu().numpy() for v in (response_g, rpm, mask))
        if x.ndim != 4 or x.shape[1:] != (14, 3, 2):
            raise ValueError("Expected [batch, 14, 3, 2] response tensor")
        if r.shape != (len(x),) or m.shape != (len(x), 14) or m.dtype != np.bool_:
            raise ValueError("Expected one speed per sample and a [batch, 14] boolean mask")
        observed = np.where(m[:, :, None, None], x, 0)
        finite = np.isfinite(observed).all(axis=(1, 2, 3))
        valid = finite & np.isfinite(r) & (r >= self.rpm[0]) & (r <= self.rpm[-1]) & (m.sum(axis=1) == 2)
        has_signal = np.any(observed != 0, axis=(1, 2, 3))
        normalized = observed.astype(np.float64) / self.scale[None, :, :, None]
        maximum = np.max(np.hypot(normalized[..., 0], normalized[..., 1]), axis=(1, 2))
        low_signal = finite & has_signal & (maximum < 1e-6)
        valid &= has_signal
        phase_unidentifiable = np.zeros(len(x), dtype=bool)
        prediction = np.full_like(x, np.nan, dtype=np.float32)
        unbalance = np.full(len(x), np.nan, dtype=np.float32)
        ids = np.flatnonzero(valid)
        for start in range(0, len(ids), self.chunk_size):
            take = ids[start:start + self.chunk_size]
            complex_x = observed[take, ..., 0] + 1j * observed[take, ..., 1]
            yh, uh, identified = self._solve(complex_x, r[take], m[take])
            phase_unidentifiable[take] = ~identified
            prediction[take, ..., 0], prediction[take, ..., 1] = yh.real, yh.imag
            unbalance[take] = uh
        valid &= np.isfinite(prediction).all(axis=(1, 2, 3)) & np.isfinite(unbalance)
        return {"response_hat_g": torch.as_tensor(prediction, device=device),
                "unbalance_g_cm": torch.as_tensor(unbalance, device=device),
                "prediction_valid": torch.as_tensor(valid, device=device),
                "quality_flags": {"classical_invalid_input_or_prediction": torch.as_tensor(~valid, device=device),
                                  "zero_observation": torch.as_tensor(finite & ~has_signal, device=device),
                                  "low_signal": torch.as_tensor(low_signal, device=device),
                                  "common_phase_unidentifiable": torch.as_tensor(phase_unidentifiable, device=device)}}
