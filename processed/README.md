# Processed inputs for Table 7

All accelerations are peak once-per-revolution (1X) coefficients in g.
The last dimension stores real and imaginary parts, in that order.
Node order is defined in `node_spec.py`; directions are radial, tangential, axial.

- `m6_paired_1x.npz` contains 200 window-level measured 1X responses and their matched synthetic 1X responses, evaluated at the same M6 window speeds and nominal U (864.54 g·cm). Both arrays have shape [200,14,3,2] in the same window-specific phase frame. The file includes only window ID, speed, nominal U, node IDs, validity masks and the two coefficient arrays. M6 is one measured condition with correlated windows. It was held out of the surface and training.
- `pchip_speed_grid.npz` contains five speed-gridded, force-phase-aligned anchors [5,599,14,3,2]. Four are derived from the M2/M3/M5/M8 conditions. The zero-U anchor is assumed. At each of 599 speeds (3005–5995 r/min, 5 r/min spacing), interpolate the real and imaginary components independently along U using PCHIP. The bundled training and validation U values and level indices recover the original split. The six RMS scales were fitted to training data only. The inversion uses the regenerated **training table** and piecewise linear interpolation, including at speeds between grid points.

`prepare_tables.py` regenerates the full synthetic train and validation response arrays. Their float64 contents were checked element by element against the original archives. The small preprocessing file contains the six scales used by the joint models. Online GNN inference loads these scales from the checkpoint.

The published M6 run uses unperturbed observed 1X coefficients. M/S identify measured/synthetic sources. For each layout, validity is shared by both sets of predictions and all input–target pairings. Both observed sensors must have valid channels. For scoring, use only valid channels of the remaining 12 sites. Complex squared errors sum the real and imaginary component errors per channel. The paired input phasors are already in the same reference frame. No extra phase fitting is applied to neural output for scoring.

The full replay was checked using Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1 and PyTorch 2.11.0+cu128. The GNN, MLP and DeepSets ran in FP32 on CUDA with TF32 disabled. As in the original evaluation, inversion used the same FP32 RMS scales, and errors and energies were accumulated in float64. All 48 printed mean/SD entries agreed with the reference table.
