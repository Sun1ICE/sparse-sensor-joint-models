"""Portable public inference; invalid predictions become NaN / JSON null."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models import MODEL_VERSION, build_model
from node_spec import N_NODES, NODE_IDS


class Predictor:
    def __init__(self, model, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, response_g, rpm, observed_mask):
        response = torch.as_tensor(response_g, dtype=torch.float32, device=self.device)
        speed = torch.as_tensor(rpm, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(observed_mask, device=self.device)
        output = self.model(response, speed, mask)
        valid = output["prediction_valid"]
        if output["response_hat_g"] is not None:
            output["response_hat_g"] = output["response_hat_g"].masked_fill(~valid[:, None, None, None], float("nan"))
        if output["unbalance_g_cm"] is not None:
            output["unbalance_g_cm"] = output["unbalance_g_cm"].masked_fill(~valid, float("nan"))
        return {key: ({name: value.cpu().numpy() for name, value in item.items()} if isinstance(item, dict)
                      else item.cpu().numpy() if item is not None else None) for key, item in output.items()}

    def predict_two_points(self, node_ids, response_g, rpm):
        """Two shared sensor IDs, acceleration [B,2,3,2], RPM [B].

        Reordering both IDs and their response axis leaves the prediction unchanged.
        For varying pairs in a batch use ``predict`` with canonical 14-node masks.
        """
        if len(node_ids) != 2 or len(set(node_ids)) != 2:
            raise ValueError("Exactly two distinct casing-qualified node IDs are required")
        if any(node_id not in NODE_IDS for node_id in node_ids):
            raise ValueError(f"Unknown node ID; valid IDs: {', '.join(NODE_IDS)}")
        response = np.asarray(response_g, dtype=np.float32)
        if response.ndim != 4 or response.shape[1:] != (2, 3, 2):
            raise ValueError("Two-point response_g must have shape [B,2,3,2]")
        full = np.zeros((len(response), N_NODES, 3, 2), dtype=np.float32)
        mask = np.zeros((len(response), N_NODES), dtype=bool)
        indices = [NODE_IDS.index(node_id) for node_id in node_ids]
        full[:, indices] = response
        mask[:, indices] = True
        return self.predict(full, rpm, mask)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    version = config.get("version")
    if version not in (1, MODEL_VERSION):
        raise ValueError("Unsupported model checkpoint version")
    if version == 1 and config.get("family") not in ("gnn", "mlp"):
        raise ValueError("Version 1 checkpoints only support the original gnn/mlp families")
    if checkpoint.get("family", config["family"]) != config["family"]:
        raise ValueError("Checkpoint family metadata disagrees")
    model = build_model(family=config["family"], scale_g=config["scale_g"])
    if config.get("mlp_width") not in (None, model.model_config["mlp_width"]):
        raise ValueError("Checkpoint MLP width does not match this model version")
    if version == MODEL_VERSION:
        if config.get("task") != model.task or config.get("architecture") != model.model_config["architecture"]:
            raise ValueError("Checkpoint task/architecture does not match this model version")
    # These buffers are contract data, not learned weights. Strict tensor shape
    # loading alone would silently accept a changed scale or node ordering.
    for name in ("scale_g", "node_scale_g", "node_geometry", "edge_geometry"):
        expected = getattr(model, name).cpu()
        actual = checkpoint["state_dict"].get(name)
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected):
            raise ValueError(f"Checkpoint frozen {name} differs from its model contract")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return Predictor(model, device=device)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="JSON with node_ids, response_g [B,2,3,2], rpm [B]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    predictor = load_checkpoint(args.checkpoint, args.device)
    if "node_ids" in payload:
        result = predictor.predict_two_points(payload["node_ids"], payload["response_g"], payload["rpm"])
    else:
        result = predictor.predict(payload["response_g"], payload["rpm"], payload["observed_mask"])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
