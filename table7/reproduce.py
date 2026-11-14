"""Recreate Table 7 from per-family, split, seed and layout-type sufficient statistics."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
METHODS = {
    "gnn": "GNN",
    "mlp": "MLP",
    "deepsets": "DeepSets",
    "classical_linear_profile": "Training-table inversion",
}
SCENARIOS = ("SS", "SM", "MS", "MM")
SPLITS = ("split0", "split1", "split2")
TYPES = ("TT", "II", "TI")
METRICS = ("complex_rmse_g", "relative_l2", "u_mae_g_cm")


def load_metrics(path):
    """Compute the three per-type metrics after pooling all valid observations."""
    metrics = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), 2):
            key = tuple(row[field] for field in ("family", "scenario", "split_id", "seed", "pair_type"))
            if key in metrics:
                raise ValueError(f"Duplicate group in row {row_number}: {key}")
            sse = float(row["complex_sse_g2"])
            energy = float(row["target_energy_g2"])
            absolute = float(row["u_abs_error_sum_g_cm"])
            channels = int(row["complex_count"])
            cases = int(row["valid_cases"])
            if not all(math.isfinite(v) for v in (sse, energy, absolute)):
                raise ValueError(f"Nonfinite statistic in row {row_number}")
            if min(sse, absolute) < 0 or energy <= 0 or not (0 < cases <= 7 * 200 and 0 < channels <= cases * 36):
                raise ValueError(f"Invalid sum or coverage in row {row_number}")
            metrics[key] = (math.sqrt(sse / channels), math.sqrt(sse / energy), absolute / cases)
    return metrics


def summarize(metrics):
    split_metrics = defaultdict(list)
    for family in METHODS:
        seeds = ("",) if family == "classical_linear_profile" else ("42", "43", "44")
        for scenario in SCENARIOS:
            for split_id in SPLITS:
                seed_metrics = []
                for seed in seeds:
                    by_type = []
                    for pair_type in TYPES:
                        key = family, scenario, split_id, seed, pair_type
                        if key not in metrics:
                            raise ValueError(f"Missing group: {key}")
                        by_type.append(metrics[key])
                    seed_metrics.append(tuple(map(statistics.mean, zip(*by_type))))
                split_metrics[family, scenario].append(tuple(map(statistics.mean, zip(*seed_metrics))))
    if len(metrics) != 360:
        raise ValueError(f"Expected exactly 360 groups, found {len(metrics)}")
    for (family, scenario), split_values in split_metrics.items():
        for name, values in zip(METRICS, zip(*split_values)):
            yield {
                "family": family, "scenario": scenario, "metric": name,
                "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            }


def paper_cells(rows):
    """Use Table 7's column units and printed precision."""
    by_key = {(row["family"], row["scenario"], row["metric"]): row for row in rows}
    for family in METHODS:
        for scenario in SCENARIOS:
            cells = []
            for metric, factor in zip(METRICS, (1000, 100, 1)):
                places = 3
                if family == "classical_linear_profile" and scenario == "SM" and metric != "u_mae_g_cm":
                    places = 5
                if family == "classical_linear_profile" and scenario == "SS" and metric == "relative_l2":
                    places = 4
                row = by_key[family, scenario, metric]
                cells.append(f"{factor * row['mean']:.{places}f} ± {factor * row['sample_sd']:.{places}f}")
            yield [METHODS[family], scenario, *cells]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "group_sums.csv")
    parser.add_argument("--output", type=Path, help="Write 48 unrounded mean/SD rows in native units.")
    args = parser.parse_args()
    results = list(summarize(load_metrics(args.input)))
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print("| Method | Pairing | Complex RMSE (10⁻³ g) | Relative L2 (%) | U MAE (g·cm) |")
    print("|---|---|---:|---:|---:|")
    for cells in paper_cells(results):
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
