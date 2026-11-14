"""Read already-prepared complex 1X response curves and compute training scales."""
import json
from pathlib import Path

import numpy as np
import torch

from models import RPM_MAX, RPM_MIN
from node_spec import ANGLES_DEG, CASING_IDS, DIRECTIONS, NODE_IDS, U_MAX

HERE = Path(__file__).resolve().parent


def read_curves(path):
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in (("node_ids", NODE_IDS), ("angles_deg", ANGLES_DEG),
                              ("casing_ids", CASING_IDS), ("directions", DIRECTIONS)):
            if key in archive and not np.array_equal(archive[key], expected):
                raise ValueError(f"{path.name}: unexpected ordering of {key}")
        return archive["response_g"], archive["unbalance_g_cm"], archive["rpm"]


def demo_curves(levels):
    # Seven response channels per casing, three directions, and a different
    # phase offset for each channel. All coefficients are arbitrary.
    rpm = np.linspace(3500, 5500, 4)
    channel = np.arange(42).reshape(1, 1, 14, 3)
    u = np.array(levels, dtype=np.float64)
    speed = rpm[None, :, None, None] / 4500
    magnitude = (u[:, None, None, None] / U_MAX) * speed * (0.1 + channel * 0.002)
    phase = speed * 0.3 + channel * 0.13
    z = magnitude * np.exp(1j * phase)
    return np.stack([z.real, z.imag], axis=-1), u, rpm


def rms_scales(y):
    # Accumulate in small chunks to bound peak memory for float64 archives.
    energy = np.zeros((2, 3), dtype=np.float64)
    for start in range(0, len(y), 16):
        block = np.asarray(y[start:start + 16], dtype=np.float64)
        for c, nodes in enumerate((slice(0, 7), slice(7, 14))):
            energy[c] += np.square(block[:, :, nodes]).sum(axis=(0, 1, 2, 4))
    scale = np.sqrt(energy / (y.shape[0] * y.shape[1] * 7)).astype(np.float32)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Training data must provide six finite positive response scales")
    return scale


class DevelopmentData:
    def __init__(self, config, split, device, data_dir=None):
        self.config = config
        self.device = torch.device(device)
        masks = json.loads((HERE / "pair_splits" / f"{split}.json").read_text(encoding="utf-8-sig"))
        self.train_pairs = np.asarray(masks["train_pairs"], dtype=np.int64)
        if data_dir is None:
            train = demo_curves([0, 300, 650, 935.77])
            validation = demo_curves([200, 700])
        else:
            train = read_curves(data_dir / "train.npz")
            validation = read_curves(data_dir / "validation.npz")
        for name, (y, u, rpm) in (("train", train), ("val", validation)):
            if u.ndim != 1 or rpm.ndim != 1 or u.size == 0 or rpm.size == 0:
                raise ValueError("Expected nonempty 1-D unbalance and speed axes")
            if y.shape != (len(u), len(rpm), 14, 3, 2) or not np.isfinite(y).all():
                raise ValueError(f"Invalid {name} response array")
            if np.any(~np.isfinite(u)) or np.any(u < 0) or np.any(u > U_MAX):
                raise ValueError("Invalid nominal unbalance")
            if not np.all((rpm >= RPM_MIN) & (rpm <= RPM_MAX)) or np.any(np.diff(rpm) <= 0):
                raise ValueError("Invalid speed grid")
            if len(np.unique(u.astype(np.float32))) != len(u):
                raise ValueError("Unbalance values must be distinct inside each development set")
            for suffix, arr in (("u", u), ("y", y)):
                setattr(self, name + "_" + suffix, torch.as_tensor(arr, dtype=torch.float32, device=device))
        if not np.array_equal(train[2], validation[2]):
            raise ValueError("Training and validation must have matching speed axes")
        if np.intersect1d(train[1].astype(np.float32), validation[1].astype(np.float32)).size:
            raise ValueError("Training and validation must use disjoint unbalance levels")
        self.rpm = torch.as_tensor(train[2], dtype=torch.float32, device=device)
        scale = rms_scales(train[0])
        if data_dir is not None and (data_dir / "preprocessing.npz").is_file():
            with np.load(data_dir / "preprocessing.npz", allow_pickle=False) as prep:
                stored = prep["scale_g"].astype(np.float32)
            if stored.shape != (2, 3) or not np.allclose(scale, stored, rtol=1e-6, atol=0):
                raise ValueError("Supplied scale_g differs from the six training-only RMS scales")
            scale = stored
        self.scale_g = torch.as_tensor(scale, device=device)
