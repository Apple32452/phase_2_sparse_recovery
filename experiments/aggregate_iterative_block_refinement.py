"""
aggregate_iterative_block_refinement.py

Aggregate iterative learned block-refinement results.

This script summarizes:
  1. one-step refinement performance across seeds
  2. gains over CoSaMP, block_score_topk, and learned_block_scorer
  3. iteration ablation: refine-iters = 1, 2, 4

Outputs:
  results/iterative_learned_block_refinement/aggregate_iterative_block_refinement.json
  figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_seed_summary.png
  figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_gains.png
  figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_iter_ablation.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "iterative_learned_block_refinement"
FIGURES_DIR = ROOT / "figures" / "iterative_learned_block_refinement"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# One-step refinement files.
# Seed 0 files use the *_iter1 name.
# Seed 2 files use *_seed2_iter1.
# If you later run seed1 with refine-iters=1, this script will automatically include it.
SEED_FILES = {
    "m=96,k=40": [
        ("seed0", "iterative_learned_block_refinement_m96_k40_iter1.json"),
        ("seed1", "iterative_learned_block_refinement_m96_k40_seed1_iter1.json"),
        ("seed2", "iterative_learned_block_refinement_m96_k40_seed2_iter1.json"),
    ],
    "m=96,k=55": [
        ("seed0", "iterative_learned_block_refinement_m96_k55_iter1.json"),
        ("seed1", "iterative_learned_block_refinement_m96_k55_seed1_iter1.json"),
        ("seed2", "iterative_learned_block_refinement_m96_k55_seed2_iter1.json"),
    ],
}


# Iteration ablation files, seed 0.
ITER_ABLATION_FILES = {
    "m=96,k=40": [
        (1, "iterative_learned_block_refinement_m96_k40_iter1.json"),
        (2, "iterative_learned_block_refinement_m96_k40_iter2.json"),
        (4, "iterative_learned_block_refinement_m96_k40_iter4.json"),
    ],
    "m=96,k=55": [
        (1, "iterative_learned_block_refinement_m96_k55_iter1.json"),
        (2, "iterative_learned_block_refinement_m96_k55_iter2.json"),
        (4, "iterative_learned_block_refinement_m96_k55_iter4.json"),
    ],
}


METHODS = [
    "naive",
    "cosamp",
    "block_score_topk",
    "learned_block_scorer",
    "iterative_refinement",
    "oracle",
]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def method_metric(data, method, metric="nrmse"):
    return float(data["summary"][method][metric]["mean"])


def method_std(data, method, metric="nrmse"):
    return float(data["summary"][method][metric]["std"])


def collect_seed_summary():
    rows = []

    for setting, files in SEED_FILES.items():
        for seed_name, filename in files:
            path = RESULTS_DIR / filename

            if not path.exists():
                print(f"Warning: missing {path}; skipping.")
                continue

            data = load_json(path)
            row = {
                "setting": setting,
                "seed": seed_name,
                "path": str(path),
                "k": int(data["config"]["k"]),
                "m": int(data["config"]["m"]),
                "refine_iters": int(data["config"]["refine_iters"]),
            }

            for method in METHODS:
                row[f"{method}_nrmse"] = method_metric(data, method, "nrmse")
                row[f"{method}_iou"] = method_metric(data, method, "iou")
                row[f"{method}_support_size"] = method_metric(data, method, "support_size")

            row["gain_iterative_vs_cosamp"] = (
                row["cosamp_nrmse"] - row["iterative_refinement_nrmse"]
            )
            row["gain_iterative_vs_block_score"] = (
                row["block_score_topk_nrmse"] - row["iterative_refinement_nrmse"]
            )
            row["gain_iterative_vs_learned"] = (
                row["learned_block_scorer_nrmse"] - row["iterative_refinement_nrmse"]
            )

            rows.append(row)

    if not rows:
        raise FileNotFoundError("No seed-summary files found.")

    return rows


def collect_iteration_ablation():
    rows = []

    for setting, files in ITER_ABLATION_FILES.items():
        for refine_iters, filename in files:
            path = RESULTS_DIR / filename

            if not path.exists():
                print(f"Warning: missing {path}; skipping.")
                continue

            data = load_json(path)

            row = {
                "setting": setting,
                "refine_iters": refine_iters,
                "path": str(path),
                "k": int(data["config"]["k"]),
                "m": int(data["config"]["m"]),
                "iterative_nrmse": method_metric(data, "iterative_refinement", "nrmse"),
                "iterative_iou": method_metric(data, "iterative_refinement", "iou"),
                "cosamp_nrmse": method_metric(data, "cosamp", "nrmse"),
                "block_score_topk_nrmse": method_metric(data, "block_score_topk", "nrmse"),
                "learned_block_scorer_nrmse": method_metric(data, "learned_block_scorer", "nrmse"),
            }

            row["gain_iterative_vs_cosamp"] = row["cosamp_nrmse"] - row["iterative_nrmse"]
            row["gain_iterative_vs_block_score"] = (
                row["block_score_topk_nrmse"] - row["iterative_nrmse"]
            )
            row["gain_iterative_vs_learned"] = (
                row["learned_block_scorer_nrmse"] - row["iterative_nrmse"]
            )

            rows.append(row)

    if not rows:
        raise FileNotFoundError("No iteration-ablation files found.")

    return rows


def summarize_by_setting(seed_rows):
    summary = {}

    for setting in sorted(set(r["setting"] for r in seed_rows)):
        setting_rows = [r for r in seed_rows if r["setting"] == setting]
        summary[setting] = {"n_seeds": len(setting_rows), "methods": {}, "gains": {}}

        for method in METHODS:
            vals = np.array([r[f"{method}_nrmse"] for r in setting_rows], dtype=float)
            summary[setting]["methods"][method] = {
                "nrmse_mean": float(np.mean(vals)),
                "nrmse_std_across_seeds": float(np.std(vals)),
                "values": vals.tolist(),
            }

        for gain_key in [
            "gain_iterative_vs_cosamp",
            "gain_iterative_vs_block_score",
            "gain_iterative_vs_learned",
        ]:
            vals = np.array([r[gain_key] for r in setting_rows], dtype=float)
            summary[setting]["gains"][gain_key] = {
                "mean": float(np.mean(vals)),
                "std_across_seeds": float(np.std(vals)),
                "values": vals.tolist(),
            }

    return summary


def plot_seed_summary(summary):
    settings = list(summary.keys())
    methods_to_plot = [
        "cosamp",
        "block_score_topk",
        "learned_block_scorer",
        "iterative_refinement",
        "oracle",
    ]

    x = np.arange(len(settings))
    width = 0.15

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for i, method in enumerate(methods_to_plot):
        means = [summary[s]["methods"][method]["nrmse_mean"] for s in settings]
        stds = [summary[s]["methods"][method]["nrmse_std_across_seeds"] for s in settings]

        ax.bar(
            x + (i - len(methods_to_plot) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylabel("NRMSE")
    ax.set_title("One-step learned block refinement: aggregate across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_iterative_block_refinement_seed_summary.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_gains(summary):
    settings = list(summary.keys())

    gain_names = [
        ("gain_iterative_vs_cosamp", "iterative vs CoSaMP"),
        ("gain_iterative_vs_block_score", "iterative vs block_score"),
        ("gain_iterative_vs_learned", "iterative vs learned_block"),
    ]

    x = np.arange(len(settings))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for i, (gain_key, label) in enumerate(gain_names):
        means = [summary[s]["gains"][gain_key]["mean"] for s in settings]
        stds = [summary[s]["gains"][gain_key]["std_across_seeds"] for s in settings]

        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=label,
        )

        for j, val in enumerate(means):
            ax.text(
                x[j] + (i - 1) * width,
                val,
                f"{val:+.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylabel("NRMSE gain")
    ax.set_title("Positive gain means iterative refinement has lower NRMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_iterative_block_refinement_gains.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_iteration_ablation(iter_rows):
    settings = sorted(set(r["setting"] for r in iter_rows))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for setting in settings:
        rows = sorted(
            [r for r in iter_rows if r["setting"] == setting],
            key=lambda r: r["refine_iters"],
        )

        xs = [r["refine_iters"] for r in rows]
        ys = [r["iterative_nrmse"] for r in rows]

        ax.plot(xs, ys, marker="o", label=setting)

        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("refinement iterations")
    ax.set_ylabel("iterative_refinement NRMSE")
    ax.set_title("Refinement iteration ablation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_iterative_block_refinement_iter_ablation.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def main():
    seed_rows = collect_seed_summary()
    iter_rows = collect_iteration_ablation()
    setting_summary = summarize_by_setting(seed_rows)

    out = {
        "seed_rows": seed_rows,
        "iteration_ablation_rows": iter_rows,
        "setting_summary": setting_summary,
        "interpretation": (
            "One-step learned block refinement is evaluated across available seeds. "
            "Positive gain values mean iterative_refinement has lower NRMSE than the baseline. "
            "The iteration ablation tests whether additional refinement steps help or cause drift."
        ),
    }

    out_json = RESULTS_DIR / "aggregate_iterative_block_refinement.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_json}")

    print("\nAggregate one-step learned block refinement across available seeds")
    print("-" * 100)
    print(
        f"{'setting':<12} {'n_seeds':>7} {'CoSaMP':>10} {'block':>10} "
        f"{'learned':>10} {'iterative':>12} {'gain I-C':>10} {'gain I-B':>10}"
    )

    for setting, s in setting_summary.items():
        print(
            f"{setting:<12} "
            f"{s['n_seeds']:>7d} "
            f"{s['methods']['cosamp']['nrmse_mean']:>10.4f} "
            f"{s['methods']['block_score_topk']['nrmse_mean']:>10.4f} "
            f"{s['methods']['learned_block_scorer']['nrmse_mean']:>10.4f} "
            f"{s['methods']['iterative_refinement']['nrmse_mean']:>12.4f} "
            f"{s['gains']['gain_iterative_vs_cosamp']['mean']:>+10.4f} "
            f"{s['gains']['gain_iterative_vs_block_score']['mean']:>+10.4f}"
        )

    print("\nIteration ablation")
    print("-" * 60)
    print(f"{'setting':<12} {'iters':>6} {'iterative NRMSE':>18} {'gain vs block':>15}")

    for r in sorted(iter_rows, key=lambda x: (x["setting"], x["refine_iters"])):
        print(
            f"{r['setting']:<12} "
            f"{r['refine_iters']:>6d} "
            f"{r['iterative_nrmse']:>18.4f} "
            f"{r['gain_iterative_vs_block_score']:>+15.4f}"
        )

    plot_seed_summary(setting_summary)
    plot_gains(setting_summary)
    plot_iteration_ablation(iter_rows)


if __name__ == "__main__":
    main()
