"""
aggregate_structured_priors.py

Aggregate structured-prior recovery results across multiple regimes.

Inputs:
    results/structured_priors/structured_priors.json
    results/structured_priors/structured_priors_k55.json
    results/structured_priors/structured_priors_k70.json
    results/structured_priors/structured_priors_m96_k40.json
    results/structured_priors/structured_priors_m96_k55.json

Outputs:
    results/structured_priors/aggregate_structured_priors.json
    figures/structured_priors/aggregate_structured_priors_block_gain.png
    figures/structured_priors/aggregate_structured_priors_cluster_gain.png
    figures/structured_priors/aggregate_structured_priors_heatmap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "structured_priors"
FIGURES_DIR = ROOT / "figures" / "structured_priors"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


INPUT_FILES = [
    "structured_priors.json",
    "structured_priors_k55.json",
    "structured_priors_k70.json",
    "structured_priors_m96_k40.json",
    "structured_priors_m96_k55.json",
]


def load_results():
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
        raise FileNotFoundError("No structured-prior result files found.")

    return runs


def get_nrmse(run, family, method):
    return float(run["data"]["summary"][family][method]["nrmse"]["mean"])


def get_iou(run, family, method):
    return float(run["data"]["summary"][family][method]["iou"]["mean"])


def build_summary(runs):
    rows = []

    for run in runs:
        for family in run["data"]["families"]:
            row = {
                "label": run["label"],
                "m": run["m"],
                "k": run["k"],
                "family": family,
            }

            for method in run["data"]["methods"]:
                row[f"{method}_nrmse"] = get_nrmse(run, family, method)
                row[f"{method}_iou"] = get_iou(run, family, method)

            # Positive gain means the structured heuristic beats CoSaMP.
            if family == "block_sparse":
                row["structured_method"] = "block_score_topk"
                row["structured_gain_vs_cosamp"] = (
                    row["cosamp_nrmse"] - row["block_score_topk_nrmse"]
                )
            elif family in ["cluster_sparse", "markov_sparse"]:
                row["structured_method"] = "smoothed_topk"
                row["structured_gain_vs_cosamp"] = (
                    row["cosamp_nrmse"] - row["smoothed_topk_nrmse"]
                )
            else:
                row["structured_method"] = "none"
                row["structured_gain_vs_cosamp"] = np.nan

            rows.append(row)

    return rows


def plot_block_gain(rows):
    block_rows = [r for r in rows if r["family"] == "block_sparse"]

    labels = [r["label"] for r in block_rows]
    cosamp = [r["cosamp_nrmse"] for r in block_rows]
    block = [r["block_score_topk_nrmse"] for r in block_rows]
    gains = [r["structured_gain_vs_cosamp"] for r in block_rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, cosamp, width, label="CoSaMP")
    ax.bar(x + width / 2, block, width, label="block_score_topk")

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("NRMSE")
    ax.set_title("Block-sparse recovery: structured prior vs CoSaMP")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, g in enumerate(gains):
        text = f"gain={g:+.3f}"
        ax.text(i, max(cosamp[i], block[i]) + 0.03, text, ha="center", fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "aggregate_structured_priors_block_gain.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_cluster_gain(rows):
    cluster_rows = [r for r in rows if r["family"] == "cluster_sparse"]

    labels = [r["label"] for r in cluster_rows]
    cosamp = [r["cosamp_nrmse"] for r in cluster_rows]
    smooth = [r["smoothed_topk_nrmse"] for r in cluster_rows]
    gains = [r["structured_gain_vs_cosamp"] for r in cluster_rows]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, cosamp, width, label="CoSaMP")
    ax.bar(x + width / 2, smooth, width, label="smoothed_topk")

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("NRMSE")
    ax.set_title("Cluster-sparse recovery: structured prior vs CoSaMP")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, g in enumerate(gains):
        text = f"gain={g:+.3f}"
        ax.text(i, max(cosamp[i], smooth[i]) + 0.03, text, ha="center", fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "aggregate_structured_priors_cluster_gain.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_gain_heatmap(rows):
    families = ["block_sparse", "cluster_sparse", "markov_sparse"]
    labels = []
    for r in rows:
        if r["family"] == "block_sparse":
            labels.append(r["label"])

    labels = list(dict.fromkeys(labels))

    heat = np.zeros((len(families), len(labels)))

    for i, fam in enumerate(families):
        for j, label in enumerate(labels):
            match = [r for r in rows if r["family"] == fam and r["label"] == label]
            if match:
                heat[i, j] = match[0]["structured_gain_vs_cosamp"]
            else:
                heat[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(heat, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families)

    ax.set_title("Structured-prior gain over CoSaMP\npositive = structured heuristic better")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CoSaMP NRMSE - structured heuristic NRMSE")

    for i in range(len(families)):
        for j in range(len(labels)):
            val = heat[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "aggregate_structured_priors_heatmap.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def main():
    runs = load_results()
    rows = build_summary(runs)

    out = {
        "input_files": [r["path"] for r in runs],
        "rows": rows,
        "interpretation": (
            "Positive structured_gain_vs_cosamp means the structured heuristic "
            "has lower NRMSE than CoSaMP. The most important cases are "
            "block_score_topk on block_sparse and smoothed_topk on cluster_sparse."
        ),
    }

    out_json = RESULTS_DIR / "aggregate_structured_priors.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_json}")

    print("\nStructured-prior gain over CoSaMP:")
    print("positive = structured heuristic better")
    print("-" * 78)

    for r in rows:
        if r["family"] in ["block_sparse", "cluster_sparse", "markov_sparse"]:
            print(
                f"{r['label']:<12} {r['family']:<16} "
                f"{r['structured_method']:<18} "
                f"gain={r['structured_gain_vs_cosamp']:+.4f}"
            )

    plot_block_gain(rows)
    plot_cluster_gain(rows)
    plot_gain_heatmap(rows)


if __name__ == "__main__":
    main()
