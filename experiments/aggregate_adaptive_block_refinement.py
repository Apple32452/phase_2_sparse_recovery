"""
aggregate_adaptive_block_refinement.py

Aggregate adaptive learned block-refinement results across seeds.

Inputs:
  results/adaptive_learned_block_refinement/
    adaptive_learned_block_refinement_m96_k40.json
    adaptive_learned_block_refinement_m96_k55.json
    adaptive_learned_block_refinement_m96_k40_seed1.json
    adaptive_learned_block_refinement_m96_k55_seed1.json
    adaptive_learned_block_refinement_m96_k40_seed2.json
    adaptive_learned_block_refinement_m96_k55_seed2.json

Outputs:
  results/adaptive_learned_block_refinement/aggregate_adaptive_block_refinement.json
  figures/adaptive_learned_block_refinement/aggregate_adaptive_block_refinement_nrmse.png
  figures/adaptive_learned_block_refinement/aggregate_adaptive_block_refinement_gains.png
  figures/adaptive_learned_block_refinement/aggregate_adaptive_block_refinement_steps.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "adaptive_learned_block_refinement"
FIGURES_DIR = ROOT / "figures" / "adaptive_learned_block_refinement"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


SEED_FILES = {
    "m=96,k=40": [
        ("seed0", "adaptive_learned_block_refinement_m96_k40.json"),
        ("seed1", "adaptive_learned_block_refinement_m96_k40_seed1.json"),
        ("seed2", "adaptive_learned_block_refinement_m96_k40_seed2.json"),
    ],
    "m=96,k=55": [
        ("seed0", "adaptive_learned_block_refinement_m96_k55.json"),
        ("seed1", "adaptive_learned_block_refinement_m96_k55_seed1.json"),
        ("seed2", "adaptive_learned_block_refinement_m96_k55_seed2.json"),
    ],
}


METHODS = [
    "cosamp",
    "block_score_topk",
    "learned_block_scorer",
    "one_step_refinement",
    "fixed_iterative_refinement",
    "adaptive_refinement",
    "oracle",
]


GAIN_BASELINES = [
    ("cosamp", "adaptive vs CoSaMP"),
    ("block_score_topk", "adaptive vs block_score"),
    ("learned_block_scorer", "adaptive vs learned_block"),
    ("one_step_refinement", "adaptive vs one_step"),
    ("fixed_iterative_refinement", "adaptive vs fixed_iterative"),
]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def get_mean(data, method: str, metric: str = "nrmse") -> float:
    return float(data["summary"][method][metric]["mean"])


def collect_rows():
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
                "m": int(data["config"]["m"]),
                "k": int(data["config"]["k"]),
                "n_test": int(data["config"]["n_test"]),
                "change_weight": float(data["config"]["change_weight"]),
                "min_improvement": float(data["config"]["min_improvement"]),
                "adaptive_steps_mean": float(data["adaptive_steps"]["mean"]),
                "adaptive_steps_std": float(data["adaptive_steps"]["std"]),
                "adaptive_steps_median": float(data["adaptive_steps"]["median"]),
            }

            for method in METHODS:
                row[f"{method}_nrmse"] = get_mean(data, method, "nrmse")
                row[f"{method}_iou"] = get_mean(data, method, "iou")
                row[f"{method}_support_size"] = get_mean(data, method, "support_size")

            for baseline, _ in GAIN_BASELINES:
                row[f"gain_adaptive_vs_{baseline}"] = (
                    row[f"{baseline}_nrmse"] - row["adaptive_refinement_nrmse"]
                )

            rows.append(row)

    if not rows:
        raise FileNotFoundError("No adaptive refinement result files were found.")

    return rows


def mean_std_se(vals):
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "se": float(np.std(vals) / np.sqrt(max(len(vals), 1))),
        "values": vals.tolist(),
    }


def summarize(rows):
    out = {}

    for setting in sorted(set(r["setting"] for r in rows)):
        setting_rows = [r for r in rows if r["setting"] == setting]
        out[setting] = {
            "n_seeds": len(setting_rows),
            "methods": {},
            "gains": {},
            "adaptive_steps": mean_std_se(
                [r["adaptive_steps_mean"] for r in setting_rows]
            ),
        }

        for method in METHODS:
            out[setting]["methods"][method] = {
                "nrmse": mean_std_se([r[f"{method}_nrmse"] for r in setting_rows]),
                "iou": mean_std_se([r[f"{method}_iou"] for r in setting_rows]),
                "support_size": mean_std_se(
                    [r[f"{method}_support_size"] for r in setting_rows]
                ),
            }

        for baseline, label in GAIN_BASELINES:
            key = f"gain_adaptive_vs_{baseline}"
            out[setting]["gains"][key] = {
                "label": label,
                **mean_std_se([r[key] for r in setting_rows]),
            }

    return out


def plot_nrmse(summary):
    settings = list(summary.keys())
    methods = [
        "cosamp",
        "block_score_topk",
        "learned_block_scorer",
        "one_step_refinement",
        "fixed_iterative_refinement",
        "adaptive_refinement",
        "oracle",
    ]

    x = np.arange(len(settings))
    width = 0.11

    fig, ax = plt.subplots(figsize=(11.0, 5.0))

    for i, method in enumerate(methods):
        means = [summary[s]["methods"][method]["nrmse"]["mean"] for s in settings]
        ses = [summary[s]["methods"][method]["nrmse"]["se"] for s in settings]

        ax.bar(
            x + (i - len(methods) / 2) * width,
            means,
            width,
            yerr=ses,
            capsize=3,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylabel("NRMSE")
    ax.set_title("Adaptive learned block refinement: mean ± SE across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_adaptive_block_refinement_nrmse.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_gains(summary):
    settings = list(summary.keys())

    gain_keys = [
        ("gain_adaptive_vs_cosamp", "vs CoSaMP"),
        ("gain_adaptive_vs_block_score_topk", "vs block_score"),
        ("gain_adaptive_vs_learned_block_scorer", "vs learned_block"),
        ("gain_adaptive_vs_one_step_refinement", "vs one_step"),
        ("gain_adaptive_vs_fixed_iterative_refinement", "vs fixed_iterative"),
    ]

    x = np.arange(len(settings))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11.0, 5.0))

    for i, (key, label) in enumerate(gain_keys):
        means = [summary[s]["gains"][key]["mean"] for s in settings]
        ses = [summary[s]["gains"][key]["se"] for s in settings]

        xpos = x + (i - len(gain_keys) / 2) * width

        ax.bar(
            xpos,
            means,
            width,
            yerr=ses,
            capsize=3,
            label=label,
        )

        for j, val in enumerate(means):
            ax.text(
                xpos[j],
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
    ax.set_title("Positive gain means adaptive refinement has lower NRMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_adaptive_block_refinement_gains.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_steps(summary):
    settings = list(summary.keys())

    means = [summary[s]["adaptive_steps"]["mean"] for s in settings]
    ses = [summary[s]["adaptive_steps"]["se"] for s in settings]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(settings))
    ax.bar(x, means, yerr=ses, capsize=4)

    for i, val in enumerate(means):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylabel("accepted refinement steps")
    ax.set_title("Adaptive stopping behavior across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "aggregate_adaptive_block_refinement_steps.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def main():
    rows = collect_rows()
    summary = summarize(rows)

    out = {
        "rows": rows,
        "summary": summary,
        "interpretation": (
            "Adaptive refinement is aggregated across seeds. Error bars in plots "
            "use standard error across seeds. Positive gains mean adaptive_refinement "
            "has lower NRMSE than the baseline."
        ),
    }

    out_json = RESULTS_DIR / "aggregate_adaptive_block_refinement.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_json}")

    print("\nAggregate adaptive refinement across seeds")
    print("-" * 120)
    print(
        f"{'setting':<12} {'n':>3} "
        f"{'CoSaMP':>9} {'block':>9} {'learned':>9} "
        f"{'one_step':>9} {'fixed':>9} {'adaptive':>9} "
        f"{'gain A-C':>10} {'gain A-B':>10} {'gain A-1':>10} "
        f"{'steps':>8}"
    )

    for setting, s in summary.items():
        print(
            f"{setting:<12} "
            f"{s['n_seeds']:>3d} "
            f"{s['methods']['cosamp']['nrmse']['mean']:>9.4f} "
            f"{s['methods']['block_score_topk']['nrmse']['mean']:>9.4f} "
            f"{s['methods']['learned_block_scorer']['nrmse']['mean']:>9.4f} "
            f"{s['methods']['one_step_refinement']['nrmse']['mean']:>9.4f} "
            f"{s['methods']['fixed_iterative_refinement']['nrmse']['mean']:>9.4f} "
            f"{s['methods']['adaptive_refinement']['nrmse']['mean']:>9.4f} "
            f"{s['gains']['gain_adaptive_vs_cosamp']['mean']:>+10.4f} "
            f"{s['gains']['gain_adaptive_vs_block_score_topk']['mean']:>+10.4f} "
            f"{s['gains']['gain_adaptive_vs_one_step_refinement']['mean']:>+10.4f} "
            f"{s['adaptive_steps']['mean']:>8.3f}"
        )

    plot_nrmse(summary)
    plot_gains(summary)
    plot_steps(summary)


if __name__ == "__main__":
    main()
