"""
aggregate_ceiling_study.py

Aggregate small-n ceiling study results across multiple seeds.

Inputs:
    results/ceiling/ceiling_study_small_n_seed*.json

Outputs:
    results/ceiling/aggregate_ceiling_study.json
    figures/ceiling/aggregate_ceiling_study_nrmse.png
    figures/ceiling/aggregate_ceiling_study_ambiguity_gap.png

Purpose:
    Summarize whether the hard sparse-recovery regime is algorithmically hard
    or information-limited across multiple operator seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "ceiling"
FIGURES_DIR = ROOT / "figures" / "ceiling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pattern",
        type=str,
        default="ceiling_study_small_n_seed*.json",
        help="Glob pattern inside results/ceiling/ to aggregate.",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="aggregate_ceiling_study",
    )
    return p.parse_args()


def load_jsons(pattern: str):
    files = sorted(RESULTS_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matched {RESULTS_DIR / pattern}. "
            "Run the seed experiments first."
        )

    runs = []
    for path in files:
        with path.open() as f:
            data = json.load(f)
        runs.append((path, data))

    return runs


def common_k_values(runs):
    k_sets = []
    for _, data in runs:
        k_sets.append(set(data["by_k"].keys()))
    common = sorted(set.intersection(*k_sets), key=lambda x: int(x))
    return common


def aggregate_metric(runs, k_values, method, metric):
    """
    Aggregate seed-level means.

    Each individual JSON already stores mean/std across test signals.
    Here we aggregate the mean value across seed files.
    """
    means = []

    for k in k_values:
        vals = []
        for _, data in runs:
            try:
                vals.append(float(data["by_k"][k][method][metric]["mean"]))
            except KeyError:
                vals.append(np.nan)
        means.append(
            {
                "k": int(k),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "values": vals,
            }
        )

    return means


def aggregate_ambiguity_gap(runs, k_values):
    out = []

    for k in k_values:
        vals = []
        for _, data in runs:
            try:
                vals.append(
                    float(
                        data["by_k"][k]["exact_l0_info"]["ambiguity_gap"]["mean"]
                    )
                )
            except KeyError:
                vals.append(np.nan)

        out.append(
            {
                "k": int(k),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "values": vals,
            }
        )

    return out


def plot_nrmse(summary, out_path):
    methods = ["naive", "omp", "cosamp", "htp", "exact_l0", "oracle"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for method in methods:
        rows = summary["methods"][method]["nrmse"]
        ks = np.array([r["k"] for r in rows])
        means = np.array([r["mean"] for r in rows])
        stds = np.array([r["std"] for r in rows])

        ax.plot(ks, means, marker="o", label=method)
        ax.fill_between(ks, means - stds, means + stds, alpha=0.15)

    ax.set_xlabel("sparsity k")
    ax.set_ylabel("NRMSE")
    ax.set_title("Small-n ceiling study: mean ± std across seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


def plot_ambiguity_gap(summary, out_path):
    rows = summary["ambiguity_gap"]
    ks = np.array([r["k"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    stds = np.array([r["std"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.plot(ks, means, marker="o", label="exact L0 ambiguity gap")
    ax.fill_between(ks, means - stds, means + stds, alpha=0.15)

    ax.set_xlabel("sparsity k")
    ax.set_ylabel("second-best residual - best residual")
    ax.set_title("Support ambiguity gap across seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


def main():
    args = parse_args()

    runs = load_jsons(args.pattern)
    k_values = common_k_values(runs)

    print("=" * 78)
    print("Aggregate small-n ceiling study")
    print("=" * 78)
    print("Input files:")
    for path, _ in runs:
        print(f"  {path}")
    print(f"Common k values: {k_values}")

    methods = ["naive", "omp", "cosamp", "htp", "exact_l0", "oracle"]

    summary = {
        "input_files": [str(path) for path, _ in runs],
        "n_runs": len(runs),
        "common_k_values": [int(k) for k in k_values],
        "methods": {},
        "ambiguity_gap": aggregate_ambiguity_gap(runs, k_values),
        "interpretation": (
            "If exact_l0 and oracle stay near zero while greedy methods degrade, "
            "then the regime has algorithmic headroom rather than being purely "
            "information-limited. If the ambiguity gap shrinks with k, the task "
            "is approaching a support-identification ceiling."
        ),
    }

    for method in methods:
        summary["methods"][method] = {
            "nrmse": aggregate_metric(runs, k_values, method, "nrmse"),
            "iou": aggregate_metric(runs, k_values, method, "iou"),
            "residual": aggregate_metric(runs, k_values, method, "residual"),
        }

    out_json = RESULTS_DIR / f"{args.out_prefix}.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    out_nrmse = FIGURES_DIR / f"{args.out_prefix}_nrmse.png"
    out_gap = FIGURES_DIR / f"{args.out_prefix}_ambiguity_gap.png"

    plot_nrmse(summary, out_nrmse)
    plot_ambiguity_gap(summary, out_gap)

    print("\nSummary:")
    for method in methods:
        vals = summary["methods"][method]["nrmse"]
        formatted = ", ".join(
            f"k={v['k']}: {v['mean']:.4f}±{v['std']:.4f}" for v in vals
        )
        print(f"  {method:<9} {formatted}")

    print("\nAmbiguity gap:")
    for v in summary["ambiguity_gap"]:
        print(f"  k={v['k']}: {v['mean']:.3e} ± {v['std']:.3e}")


if __name__ == "__main__":
    main()
