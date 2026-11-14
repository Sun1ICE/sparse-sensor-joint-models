"""Development batching and counter-based noise used for training and model selection."""
import functools

import numpy as np
import torch
from node_spec import CASING_IDS

NOISE_LEVELS = (0., .01, .03, .05)


def _hash32(x):
    # Explicit two's-complement modulo arithmetic: same counters on CPU/CUDA.
    x = x & 0xffffffff
    x = ((x ^ (x >> 16)) * 0x7feb352d) & 0xffffffff
    x = ((x ^ (x >> 15)) * 0x846ca68b) & 0xffffffff
    return (x ^ (x >> 16)) & 0xffffffff


def deterministic_normal(sample_ids, noise_indices, seed):
    """One standard normal per physical node/axis/ReIm, never per layout."""
    ids = torch.as_tensor(sample_ids, dtype=torch.int64)
    levels = torch.as_tensor(noise_indices, dtype=torch.int64, device=ids.device)
    channels = torch.arange(84, device=ids.device, dtype=torch.int64)
    counter = ids[:, None] * 0x9e3779b1 + channels[None] * 0x85ebca77
    counter = counter + levels[:, None] * 0x27d4eb2d + int(seed) * 104729
    a = (_hash32(counter).double() + .5) / 4294967296.
    b = (_hash32(counter + 0x165667b1).double() + .5) / 4294967296.
    z = torch.sqrt(-2. * torch.log(a)) * torch.cos(2. * np.pi * b)
    return z.float().reshape(-1, 14, 3, 2)


@functools.lru_cache(maxsize=256)
def _device_table(values, dtype, device):
    # Small constant tables are uploaded once: each blocking host-to-CUDA copy
    # stalls the GPU queue, and make_batch runs hundreds of times per epoch.
    return torch.tensor(values, dtype=dtype, device=device)


def make_batch(data, sample_ids, pair_indices, noise_indices, *, pairs=None,
               split="validation", noise_seed=20261022, noise_levels=NOISE_LEVELS):
    pairs = data.train_pairs if pairs is None else pairs
    device = data.device
    ids = torch.as_tensor(sample_ids, dtype=torch.long, device=device)
    pi = torch.as_tensor(pair_indices, dtype=torch.long, device=device)
    ni = torch.as_tensor(noise_indices, dtype=torch.long, device=device)
    source, us = (data.train_y, data.train_u) if split == "train" else (data.val_y, data.val_u)
    if split not in ("train", "validation"):
        raise ValueError("Development batches support only train/validation")
    nr = len(data.rpm)
    ui, ri = ids // nr, ids % nr
    target = source[ui, ri]
    pair_table = _device_table(tuple(map(tuple, np.asarray(pairs, dtype=np.int64).tolist())), torch.int64, device)
    nodes = pair_table[pi]
    mask = torch.zeros((len(ids), 14), dtype=torch.bool, device=device)
    mask.scatter_(1, nodes, True)
    frac = _device_table(tuple(map(float, noise_levels)), torch.float32, device)[ni]
    node_scale = data.scale_g[_device_table(CASING_IDS, torch.int64, device)]
    # Fraction-derived stream key remains identical for a one-level resumed block.
    noise_key = torch.round(frac * 1000000.).to(torch.int64)
    noise = deterministic_normal(ids, noise_key, noise_seed) * frac[:, None, None, None] * node_scale[None, :, :, None] / np.sqrt(2.)
    response = torch.where(mask[:, :, None, None], target + noise, torch.zeros_like(target))
    return {"response_g": response, "target_g": target, "rpm": data.rpm[ri], "u": us[ui],
            "mask": mask, "sample_index": ids, "pair_index": pi, "pair_nodes": nodes,
            "noise_pct": frac, "noise_index": ni}


def training_batches(data, seed, epoch, batch_size=1024):
    n = data.train_y.shape[0] * data.train_y.shape[1]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(data.config.get("data_seed", 20261023))]))
    order = rng.permutation(n)
    pi, ni = rng.integers(0, len(data.train_pairs), n), rng.integers(0, len(NOISE_LEVELS), n)
    # One upload per epoch; per-batch slices then stay on the device.
    order, pi, ni = (torch.as_tensor(x, dtype=torch.long, device=data.device) for x in (order, pi, ni))
    noise_seed = int(data.config.get("data_seed", 20261023)) + int(seed) * 100003 + int(epoch) * 97
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield make_batch(data, order[start:end], pi[start:end], ni[start:end],
                         split="train", noise_seed=noise_seed)


def monitor_batches(data, batch_size=256):
    n = data.val_y.shape[0] * data.val_y.shape[1]
    seed = int(data.config.get("monitor_seed", 20261022))
    order = np.random.default_rng(seed).permutation(n)
    assignments = np.empty(n, dtype=np.int64)
    assignments[order] = np.arange(n) % len(data.train_pairs)
    assignments = torch.as_tensor(assignments, device=data.device)
    for level in range(4):
        for start in range(0, n, batch_size):
            ids = torch.arange(start, min(start + batch_size, n), dtype=torch.long, device=data.device)
            yield make_batch(data, ids, assignments[ids], torch.full_like(ids, level), noise_seed=seed)


def exhaustive_batches(data, pairs, noise_levels=NOISE_LEVELS, batch_size=256, split="validation"):
    if split != "validation":
        raise ValueError("Exhaustive development analysis uses validation only")
    n = data.val_y.shape[0] * data.val_y.shape[1]
    seed = int(data.config.get("monitor_seed", 20261022))
    for level in range(len(noise_levels)):
        for pair in range(len(pairs)):
            for start in range(0, n, batch_size):
                ids = np.arange(start, min(start + batch_size, n), dtype=np.int64)
                yield make_batch(data, ids, np.full(len(ids), pair), np.full(len(ids), level),
                                 pairs=pairs, noise_seed=seed, noise_levels=noise_levels)
