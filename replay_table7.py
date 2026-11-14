"""Generate the four Table 7 pairings from archived 1X features and frozen predictors."""
import argparse
import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from classical import ClassicalInverse
from node_spec import NODE_IDS
from predict import load_checkpoint
from prepare_tables import build_tables

ROOT = Path(__file__).resolve().parent
SCENARIOS = ("SS", "SM", "MS", "MM")
METRICS = ("complex_sse_g2", "target_energy_g2", "complex_count", "u_abs_error_sum_g_cm", "valid_cases")


def input_arrays(device):
    with np.load(ROOT / "processed/m6_paired_1x.npz", allow_pickle=False) as z:
        if tuple(z["node_ids"]) != NODE_IDS:
            raise ValueError("Sensor order mismatch")
        targets = [z[k].astype(np.float64) for k in ("synthetic_1x_g", "measured_1x_g")]
        inputs = [torch.as_tensor(t, dtype=torch.float32, device=device) for t in targets]
        return (targets, inputs, torch.as_tensor(z["rpm"], dtype=torch.float32, device=device),
                z["unbalance_g_cm"].astype(np.float32).astype(np.float64),
                z["phase_valid"] & z["amplitude_valid"])


def group_sums(model, pairs, arrays):
    targets, inputs, rpm, true_u, channel_valid = arrays
    sums = defaultdict(lambda: np.zeros(len(METRICS), dtype=np.float64))
    for i, j in pairs:
        kind = "TT" if j < 7 else ("II" if i >= 7 else "TI")
        mask = torch.zeros((len(rpm), len(NODE_IDS)), dtype=torch.bool, device=rpm.device)
        mask[:, [i, j]] = True
        results = [model(torch.where(mask[:, :, None, None], t, 0.0), rpm, mask) for t in inputs]
        common = channel_valid[:, [i, j], :].all(axis=(1, 2))
        preds, estimates = [], []
        for result in results:
            y = result["response_hat_g"].cpu().numpy().astype(np.float64)
            u = result["unbalance_g_cm"].cpu().numpy().astype(np.float64)
            common &= result["prediction_valid"].cpu().numpy() & np.isfinite(y).all(axis=(1, 2, 3)) & np.isfinite(u)
            preds.append(y)
            estimates.append(u)
        valid = channel_valid & common[:, None, None] & ~mask.cpu().numpy()[:, :, None]
        for scenario in SCENARIOS:
            x = int(scenario[0] == "M")
            t = targets[int(scenario[1] == "M")]
            residual = (preds[x] - t)[valid]
            observed_target = t[valid]
            sums[scenario, kind] += (np.square(residual).sum(), np.square(observed_target).sum(),
                                     valid.sum(), np.abs(estimates[x] - true_u)[common].sum(), common.sum())
    return sums


@torch.inference_mode()
def replay(output, device):
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    records = json.loads((ROOT / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    splits = {}
    for split in ("split0", "split1", "split2"):
        path = ROOT / f"training/pair_splits/{split}.json"
        splits[split] = json.loads(path.read_text(encoding="utf-8"))["heldout_pairs"]
    rows = []

    def collect(model, split, family, seed, arrays):
        sums = group_sums(model, splits[split], arrays)
        for (scenario, kind), totals in sums.items():
            row = dict(zip(METRICS, totals))
            row["complex_count"] = int(row["complex_count"])
            row["valid_cases"] = int(row["valid_cases"])
            rows.append(dict(split_id=split, family=family, seed=seed, scenario=scenario, pair_type=kind, **row))
        print(f"{split} {family} {seed} complete", flush=True)

    arrays = input_arrays(device)
    for r in records:
        ck = ROOT / r["checkpoint_path"]
        if hashlib.sha256(ck.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {ck.name}")
        model = load_checkpoint(ck, device).model
        collect(model, r["split_id"], r["family"], r["seed"], arrays)

    arrays = input_arrays(torch.device("cpu"))
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        build_tables(directory / "data", include_validation=False)
        with np.load(directory / "data/preprocessing.npz") as data:
            # The published baseline used the same FP32 RMS scales as the frozen networks.
            inversion = ClassicalInverse(directory, data["scale_g"].astype(np.float32))
        for split in splits:
            collect(inversion, split, "classical_linear_profile", "", arrays)
    if len(rows) != 360:
        raise ValueError("Unexpected number of grouped results")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split_id", "family", "seed", "scenario", "pair_type", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    replay(args.output, torch.device(args.device))
