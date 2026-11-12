"""Canonical ordered sensor contract shared by data, model and inference.

Equal angles in different casings are distinct nodes. Directions are radial,
tangential, axial; every complex component is stored as [Re, Im] in native g.
"""
from __future__ import annotations

N_NODES = 14
ANGLES_DEG = (0, 45, 90, 140, 220, 270, 315,
              0, 90, 135, 180, 225, 270, 315)
CASING_IDS = (0,) * 7 + (1,) * 7
CASINGS = ("turbine", "intermediate")
DIRECTIONS = ("radial", "tangential", "axial")
U_MAX = 935.77
GROUP_SLICES = (slice(0, 7), slice(7, 14))
EXCLUDED_NODE_IDS = ("I045",)
NODE_IDS = tuple(("T" if casing_id == 0 else "I") + f"{angle:03d}"
                 for casing_id, angle in zip(CASING_IDS, ANGLES_DEG))
NODE_SPEC = [
    {"node_id": node_id, "casing_id": casing_id,
     "casing": CASINGS[casing_id], "angle_deg": angle}
    for node_id, casing_id, angle in zip(NODE_IDS, CASING_IDS, ANGLES_DEG)
]
