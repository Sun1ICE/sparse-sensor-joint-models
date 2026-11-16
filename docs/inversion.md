# Training-table inversion

`classical.py` implements the piecewise-linear inverse used for the reported comparison. It uses two triaxial observed sites (six complex 1X acceleration coefficients, in g) and a rotational speed (r/min). It returns 14 triaxial complex responses in the input phase frame and a scalar unbalance magnitude in g·cm.

## Calibration and inputs

Pass `ClassicalInverse(root, scale_g)` the parent of `data/train.npz` and the six training RMS scales in an array of shape `[2, 3]`. Row order is turbine then intermediate casing. Direction order is radial, tangential, axial. Compute each RMS from the squared complex magnitudes of its seven sites over all training U values and speeds. The archive contains these arrays:

| Key | Shape | Contents |
| --- | --- | --- |
| `unbalance_g_cm` | `[K]` | Distinct training U knots spanning 0 to 935.77 g·cm |
| `rpm` | `[S]` | Increasing speed knots in r/min |
| `response_g` | `[K, S, 14, 3, 2]` | Full-field real and imaginary parts in g, in the order defined by `node_spec.py` |

The study used 1,000 U knots and 599 speeds from 3005 to 5995 r/min in steps of 5. It constructed training responses from the PCHIP surface offline. Online inversion reads the resulting training table and linearly interpolates its complex components. The M6 evaluation uses the same calibrated table.

## Objective and search

For observed channel $j$, write $z_j$ for its complex measurement, $h_j(U,n)$ for the tabulated response interpolated to the candidate magnitude and measured speed, and $s_j$ for the corresponding training RMS. Minimize

$$
\sum_{j=1}^6 |z_j-e^{i\phi}h_j(U,n)|^2/s_j^2
$$

over $U$ and the shared phase $\phi$. At each trial $U$, form $c=\sum_j\overline{h_j}z_j/s_j^2$. When $|c|>0$, the minimizing phase factor is $c/|c|$. The profiled cost equals $\sum_j(|z_j|^2+|h_j|^2)/s_j^2 -2|c|$.

1. Interpolate in speed at all training U knots and evaluate the profiled cost for the six observed channels at every knot.
2. Take both available adjacent U intervals of the eight lowest-cost knots. Evaluate each interval by linearly interpolating its six complex channels and running 24 golden-section refinements. On intervals anchored at an exactly zero response vector, also test the analytic amplitude projection clipped to the interval.
3. Compare the refined candidates with the best original knot. At the selected U, interpolate all 42 complex channels and rotate them by the profiled phase factor.

This bounded search approximates the continuous optimum. The interval widths depend on the training knots. `top_knots` and `refinement_steps` control search effort; `chunk_size` controls the number of samples evaluated together. Defaults are 8, 24 and 32.

## Prediction and validity

Call the inverse with PyTorch tensors `inverse(response_g, rpm, mask)`. Their shapes are `[B,14,3,2]`, `[B]`, and `[B,14]`; the mask is boolean. The search runs on the CPU, and results are returned on the response tensor's device. The routine masks unobserved channels internally. Check `prediction_valid` before interpreting either prediction. All-zero observations, invalid speed, an observed-node count different from two, and nonfinite observed data invalidate a prediction. Nonzero but weak observations carry a quality flag. A zero complex correlation at the solution leaves the common phase unidentified and invalidates the joint result.

## Simulated example

From the repository root, run:

```sh
python -m examples.simulated_pair
```

The example creates a small analytical calibration table in a temporary directory and observes two sites in a common rotated phase frame. By default it exercises a randomly initialized GNN using RMS scales from the simulated table. Supply a local checkpoint with `--checkpoint PATH` to use that model and its training scales. It checks the output shape, recovery of the analytic field by the simulated-table inverse, the effect of a shared input phase rotation on both methods, and their all-zero validity flags. The neural output on these artificial responses is only an interface demonstration. For the manuscript's measured performance, use the Table 7 reproduction materials.
