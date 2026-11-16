"""Exercise both prediction methods on deterministic simulated observations.

Run from the repository root with python -m examples.simulated_pair. The default
GNN uses randomly initialized weights and RMS scales from the simulated table.
An optional --checkpoint PATH loads a locally trained joint model.
"""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from classical import ClassicalInverse
from models import build_model
from node_spec import ANGLES_DEG, CASING_IDS, NODE_IDS, U_MAX
from predict import Predictor, load_checkpoint


def simulated_response(unbalance, rpm):
    """Return a complex 14-site, three-direction field in g at one U and RPM."""
    angle = np.deg2rad(ANGLES_DEG)[:, None]
    casing = np.asarray(CASING_IDS)[:, None]
    direction = np.arange(3)[None, :]
    speed = (rpm - 4500.0) / 1500.0
    amplitude = (0.2 + 0.04 * direction + 0.025 * np.cos(angle) + 0.035 * casing)
    phase = 0.7 * angle + 0.35 * direction + 0.18 * casing
    transfer = amplitude * np.exp(1j * phase) * (1 + (0.1 + 0.03j) * speed)
    return unbalance / U_MAX * transfer


def real_imag(z):
    return np.stack((z.real, z.imag), axis=-1).astype(np.float32)


def write_simulated_table(root):
    levels = np.linspace(0, U_MAX, 25)
    speeds = np.linspace(3005, 5995, 9)
    grid = np.asarray([[simulated_response(u, rpm) for rpm in speeds] for u in levels])
    data = root / "data"
    data.mkdir()
    np.savez_compressed(data / "train.npz", response_g=real_imag(grid),
                        unbalance_g_cm=levels, rpm=speeds)
    rms_by_casing = [
        np.sqrt(np.mean(abs(grid[:, :, np.asarray(CASING_IDS) == c]) ** 2, axis=(0, 1, 2)))
        for c in range(2)
    ]
    return np.asarray(rms_by_casing)


def check_outputs(name, result, rotation):
    # The neural predictor and the inverse share these output conventions.
    valid = np.asarray(result["prediction_valid"])
    np.testing.assert_array_equal(valid, [True, True, False])
    response = np.asarray(result["response_hat_g"])
    u = np.asarray(result["unbalance_g_cm"])
    assert response.shape == (3, 14, 3, 2) and u.shape == (3,)
    field = response[..., 0] + 1j * response[..., 1]
    np.testing.assert_allclose(field[1], field[0] * rotation, atol=4e-6, rtol=4e-5)
    np.testing.assert_allclose(u[1], u[0], atol=0.001, rtol=2e-6)
    assert np.asarray(result["quality_flags"]["zero_observation"])[2]
    print(f"{name}: simulated nonzero input -> U = {u[0]:.4f} g·cm; "
          "common phase rotation verified; zero observation flagged.")
    return field[0], u[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Path to a locally trained joint-model checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    pair = ("I135", "I315")
    nodes = [NODE_IDS.index(node) for node in pair]
    rpm = 4217.0
    u_true = 478.25
    truth = simulated_response(u_true, rpm) * np.exp(0.4j)
    rotation = np.exp(-0.73j)
    fixed_pair = truth[nodes]
    inputs = real_imag(np.stack((fixed_pair, fixed_pair * rotation, np.zeros_like(fixed_pair))))
    speeds = np.full(3, rpm, dtype=np.float32)

    full_input = np.zeros((3, 14, 3, 2), dtype=np.float32)
    full_input[:, nodes] = inputs
    mask = np.zeros((len(speeds), 14), dtype=bool)
    mask[:, nodes] = True
    with TemporaryDirectory() as work:
        root = Path(work)
        scales = write_simulated_table(root)
        if args.checkpoint is None:
            torch.manual_seed(0)
            model = Predictor(build_model("gnn", scale_g=scales), device="cpu")
            model_name = "Randomly initialized GNN (interface demonstration)"
        else:
            model = load_checkpoint(args.checkpoint, device="cpu")
            model_name = "Checkpoint model"
        check_outputs(model_name, model.predict_two_points(pair, inputs, speeds), rotation)

        inverse = ClassicalInverse(root, scales)
        result = inverse(torch.from_numpy(full_input), torch.from_numpy(speeds), torch.from_numpy(mask))
        estimated, u_estimated = check_outputs("Simulated-table inversion", result, rotation)
        np.testing.assert_allclose(u_estimated, u_true, atol=0.002, rtol=0)
        np.testing.assert_allclose(estimated, truth, atol=2e-6, rtol=1e-5)
        print(f"Simulated-table recovery checked against analytic U = {u_true} g·cm "
              "and the full complex response field.")


if __name__ == "__main__":
    main()
