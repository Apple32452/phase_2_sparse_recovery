"""
aggregate_learned_block_scorer.py

Aggregate fixed learned-block-scorer results.

Inputs:
    results/learned_block_scorer/learned_block_scorer_m96_k40_fixed.json
    results/learned_block_scorer/learned_block_scorer_m96_k55_fixed.json

Outputs:
    results/learned_block_scorer/aggregate_learned_block_scorer.json
    figures/learned_block_scorer/aggregate_learned_block_scorer_nrmse.png
    figures/learned_block_scorer/aggregate_learned_block_scorer_gains.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "learned_block_scorer"
FIGURES_DIR = ROOT / "figures" / "learned_block_scorer"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


INPUT_FILES = [
    "learned_block_scorer_m96_k40_fixed.json",
    "learned_block_scorer_m96_k55_fixed.json",
]


def load_runs():
    runs = []

    for name in INPUT_FILES:
        path = RESULTS_DIR / name

        if not path.exists():
            print(f"Warning: missing {path}, skipping.")
            continue

        with path.open() as f:
            data = json.load(f)

        cfg = data["config"]
        label = f"m={cfg['m']}, k={cfg['k']}"

        runs.append(
            {
                "path": str(path),
                "label": label,
                "m": cfg["m"],
                "k": cfg["k"],
                "data": data,
            }
        )

    if not runs:
        raise FileNotFoundError("No fixed learned-block-scorer JSON files found.")

    return runs


def metric(run, method, metric_name):
    return float(run["data"]["summary"][method][metric_name]["mean"])


def metric_std(run, method, metric_name):
    return float(run["data"]["summary"][method][metric_name]["std"])


def build_rows(runs):
    rows = []

    for run in runs:
        cosamp = metric(run, "cosamp", "nrmse")
        block = metric(run, "block_score_topk", "nrmse")
        learned = metric(run, "learned_block_scorer", "nrmse")
        oracle = metric(run, "oracle", "nrmse")
        naive = metric(run, "naive", "nrmse")

        row = {
            "label": run["label"],
            "m": run["m"],
            "k": run["k"],
            "naive_nrmse": naive,
            "cosamp_nrmse": cosamp,
            "block_score_topk_nrmse": block,
            "learned_block_scorer_nrmse": learned,
            "oracle_nrmse": oracle,
            "gain_learned_vs_cosamp": cosamp - learned,
            "gain_block_vs_cosamp": cosamp - block,
            "gain_learned_vs_block": block - learned,
            "cosamp_iou": metric(run, "cosamp", "iou"),
            "block_score_topk_iou": metric(run, "block_score_topk", "iou"),
            "learned_block_scorer_iou": metric(run, "learned_block_scorer", "iou"),
            "oracle_support_size": metric(run, "oracle", "support_size"),
        }

        rows.append(row)

    return rows


def plot_nrmse(runs, rows):
    methods = [
        "naive",
        "cosamp",
        "block_score_topk",
        "learned_block_scorer",
        "oracle",
    ]

    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    for i, method in enumerate(methods):
        means = [metric(run, method, "nrmse") for run in runs]
        stds = [metric_std(run, method, "nrmse") for run in runs]
        ax.bar(x + (i - len(methods) / 2) * width, means, width, yerr=stds, capsize=3, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NRMSE")
    ax.set_title("Fixed learned block scorer results")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_learned_block_scorer_nrmse.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_gains(rows):
    labels = [r["label"] for r in rows]
    learned_vs_cosamp = [r["gain_learned_vs_cosamp"] for r in rows]
    block_vs_cosamp = [r["gain_block_vs_cosamp"] for r in rows]
    learned_vs_block = [r["gain_learned_vs_block"] for r in rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    ax.bar(x - width, learned_vs_cosamp, width, label="learned - CoSaMP gain")
    ax.bar(x, block_vs_cosamp, width, label="block_score - CoSaMP gain")
    ax.bar(x + width, learned_vs_block, width, label="learned - block_score gain")

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NRMSE gain")
    ax.set_title("Positive gain means lower NRMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    for i, r in enumerate(rows):
        ax.text(i - width, learned_vs_cosamp[i], f"{learned_vs_cosamp[i]:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, block_vs_cosamp[i], f"{block_vs_cosamp[i]:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width, learned_vs_block[i], f"{learned_vs_block[i]:+.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_learned_block_scorer_gains.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def main():
    runs = load_runs()
    rows = build_rows(runs)

    out = {
        "input_files": [r["path"] for r in runs],
        "rows": rows,
        "interpretation": (
            "Positive gain_learned_vs_cosamp means learned_block_scorer has lower "
            "NRMSE than CoSaMP. Positive gain_learned_vs_block means learned_block_scorer "
            "has lower NRMSE than the hand-designed block_score_topk baseline."
        ),
    }

    out_json = RESULTS_DIR / "aggregate_learned_block_scorer.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_json}")

    print("\nAggregate learned block scorer results")
    print("-" * 90)
    print(
        f"{'setting':<14} {'CoSaMP':>10} {'block_topk':>12} "
        f"{'learned':>10} {'gain L-C':>10} {'gain L-B':>10} {'oracle size':>12}"
    )

    for r in rows:
        print(
            f"{r['label']:<14} "
            f"{r['cosamp_nrmse']:>10.4f} "
            f"{r['block_score_topk_nrmse']:>12.4f} "
            f"{r['learned_block_scorer_nrmse']:>10.4f} "
            f"{r['gain_learned_vs_cosamp']:>+10.4f} "
            f"{r['gain_learned_vs_block']:>+10.4f} "
            f"{r['oracle_support_size']:>12.2f}"
        )

    plot_nrmse(runs, rows)
    plot_gains(rows)


if __name__ == "__main__":
    main()
