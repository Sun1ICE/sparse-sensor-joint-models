"""Recreate the synthetic training and validation grids from processed speed anchors."""
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from node_spec import ANGLES_DEG, CASING_IDS, CASINGS, DIRECTIONS, NODE_IDS

ROOT = Path(__file__).resolve().parent


def build_tables(target, include_validation=True):
    target.mkdir(parents=True, exist_ok=True)
    keys = ["train", "validation"] if include_validation else ["train"]
    with np.load(ROOT / "processed/pchip_speed_grid.npz", allow_pickle=False) as data:
        surface = PchipInterpolator(data["anchor_u"], data["anchor_response_g"], axis=0, extrapolate=False)
        for key in keys:
            u = data[key + "_u"]
            path = target / (key + ".npz")
            response = surface(u)
            if not np.isfinite(response).all():
                raise ValueError("Query outside the calibrated range")
            np.savez_compressed(path, response_g=response, unbalance_g_cm=u,
                                level_indices=data[key + "_indices"], rpm=data["rpm"],
                                angles_deg=np.asarray(ANGLES_DEG, dtype=np.float64), directions=DIRECTIONS,
                                casing_ids=CASING_IDS, casings=CASINGS, node_ids=NODE_IDS)
            print(f"Wrote {path} with {len(u)} levels", flush=True)
        np.savez(target / "preprocessing.npz", scale_g=data["scale_g"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory for generated arrays")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new or empty output directory")
    build_tables(args.output)
