"""
adaptive_stopping_ablation.py

Robustness ablation for adaptive learned block refinement.

Question:
    Is the adaptive stopping rule robust to the support-stability weight?

We test:
    change_weight in {0.00, 0.01, 0.05, 0.10, 0.20}

on selected important cells:
    (m=96, k=40)
    (m=96, k=55)
    (m=112, k=55)
    (m=128, k=70)
    (m=80, k=70)

Outputs:
    results/adaptive_learned_block_refinement/adaptive_stopping_ablation.json

    figures/adaptive_learned_block_refinement/adaptive_stopping_ablation_nrmse_by_weight.png
    figures/adaptive_learned_block_refinement/adaptive_stopping_ablation_gain_cosamp.png
    figures/adaptive_learned_block_refinement/adaptive_stopping_ablation_steps_by_weight.png
    figures/adaptive_learned_block_refinement/adaptive_stopping_ablation_heatmap.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from adaptive_learned_block_refinement import (
    make_gaussian_operator,
    train_block_scorer,
    block_sparse_signal,
    add_noise,
    support_lstsq,
    nrmse,
    iou,
    cosamp,
    block_score_topk,
    learned_block_support,
    fixed_iterative_learned_block_refinement,
    adaptive_learned_block_refinement,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "adaptive_learned_block_refinement"
FIGURES_DIR = ROOT / "figures" / "adaptive_learned_block_refinement"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_cells(s: str):
    cells = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        m_str, k_str = item.split(":")
        cells.append((int(m_str), int(k_str)))
    return cells


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--n", type=int, default=256)
    p.add_argument(
        "--cells",
        type=str,
        default="96:40,96:55,112:55,128:70,80:70",
        help="Comma-separated m:k pairs, e.g. 96:40,96:55",
    )
    p.add_argument("--weights", type=str, default="0.00,0.01,0.05,0.10,0.20")
    p.add_argument("--seeds", type=str, default="0,1,2")

    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-test", type=int, default=120)

    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--refine-iters", type=int, default=4)
    p.add_argument("--min-improvement", type=float, default=1e-4)

    p.add_argument("--out-prefix", type=str, default="adaptive_stopping_ablation")

    return p.parse_args()


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "se": float(np.std(vals) / np.sqrt(max(len(vals), 1))),
        "median": float(np.median(vals)),
        "values": vals.tolist(),
    }


def eval_support(A, y, x_true, S_true, S_pred):
    x_hat = support_lstsq(A, y, S_pred)
    return {
        "nrmse": float(nrmse(x_hat, x_true)),
        "iou": float(iou(set(S_pred), set(S_true))),
        "support_size": int(len(S_pred)),
    }


def run_seed(args, m: int, k: int, seed: int, weights: list[float]):
    rng = np.random.default_rng(seed)

    cell_args = SimpleNamespace(
        n=args.n,
        m=m,
        k=k,
        block_size=args.block_size,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=seed,
        noise_std=args.noise_std,
        max_iters=args.max_iters,
        refine_iters=args.refine_iters,
        change_weight=0.05,
        min_improvement=args.min_improvement,
    )

    A = make_gaussian_operator(m, args.n, seed)

    print("\n" + "=" * 90)
    print(f"Stopping-weight ablation cell: m={m}, k={k}, seed={seed}")
    print("=" * 90)

    print("Training learned block scorer...")
    clf = train_block_scorer(A, cell_args, rng)

    baselines = {
        "cosamp": {"nrmse": [], "iou": []},
        "block_score_topk": {"nrmse": [], "iou": []},
        "learned_block_scorer": {"nrmse": [], "iou": []},
        "one_step_refinement": {"nrmse": [], "iou": []},
        "oracle": {"nrmse": [], "iou": []},
    }

    adaptive = {
        str(w): {"nrmse": [], "iou": [], "steps": []}
        for w in weights
    }

    print("Evaluating...")

    for _ in range(args.n_test):
        x_true, S_true, _ = block_sparse_signal(args.n, k, args.block_size, rng)
        y = add_noise(A @ x_true, args.noise_std, rng)

        S_cosamp = cosamp(A, y, k, max_iters=args.max_iters)
        S_block = block_score_topk(A, y, k, args.block_size)
        S_learned = learned_block_support(clf, A, y, k, args.block_size)

        S_one_step = fixed_iterative_learned_block_refinement(
            clf,
            A,
            y,
            k,
            args.block_size,
            refine_iters=1,
        )

        base_supports = {
            "cosamp": S_cosamp,
            "block_score_topk": S_block,
            "learned_block_scorer": S_learned,
            "one_step_refinement": S_one_step,
            "oracle": S_true,
        }

        for method, S_pred in base_supports.items():
            res = eval_support(A, y, x_true, S_true, S_pred)
            baselines[method]["nrmse"].append(res["nrmse"])
            baselines[method]["iou"].append(res["iou"])

        for w in weights:
            S_adapt, accepted_steps, _ = adaptive_learned_block_refinement(
                clf,
                A,
                y,
                k,
                args.block_size,
                max_refine_iters=args.refine_iters,
                change_weight=w,
                min_improvement=args.min_improvement,
            )

            res = eval_support(A, y, x_true, S_true, S_adapt)

            adaptive[str(w)]["nrmse"].append(res["nrmse"])
            adaptive[str(w)]["iou"].append(res["iou"])
            adaptive[str(w)]["steps"].append(float(accepted_steps))

    seed_summary = {
        "setting": f"m={m},k={k}",
        "m": m,
        "k": k,
        "seed": seed,
        "baselines": {},
        "adaptive": {},
    }

    for method, metrics in baselines.items():
        seed_summary["baselines"][method] = {
            metric: summarize(vals)
            for metric, vals in metrics.items()
        }

    for w in weights:
        key = str(w)
        seed_summary["adaptive"][key] = {
            metric: summarize(vals)
            for metric, vals in adaptive[key].items()
        }

        adaptive_mean = seed_summary["adaptive"][key]["nrmse"]["mean"]

        seed_summary["adaptive"][key]["gains"] = {
            "vs_cosamp": seed_summary["baselines"]["cosamp"]["nrmse"]["mean"] - adaptive_mean,
            "vs_block_score_topk": seed_summary["baselines"]["block_score_topk"]["nrmse"]["mean"] - adaptive_mean,
            "vs_learned_block_scorer": seed_summary["baselines"]["learned_block_scorer"]["nrmse"]["mean"] - adaptive_mean,
            "vs_one_step_refinement": seed_summary["baselines"]["one_step_refinement"]["nrmse"]["mean"] - adaptive_mean,
        }

    print("\nSeed summary")
    print("-" * 90)
    print(
        f"Baselines: CoSaMP={seed_summary['baselines']['cosamp']['nrmse']['mean']:.4f}, "
        f"block={seed_summary['baselines']['block_score_topk']['nrmse']['mean']:.4f}, "
        f"one_step={seed_summary['baselines']['one_step_refinement']['nrmse']['mean']:.4f}"
    )

    for w in weights:
        key = str(w)
        print(
            f"  weight={w:<5} "
            f"adaptive={seed_summary['adaptive'][key]['nrmse']['mean']:.4f} "
            f"steps={seed_summary['adaptive'][key]['steps']['mean']:.3f} "
            f"gain_vs_cosamp={seed_summary['adaptive'][key]['gains']['vs_cosamp']:+.4f} "
            f"gain_vs_one_step={seed_summary['adaptive'][key]['gains']['vs_one_step_refinement']:+.4f}"
        )

    return seed_summary


def aggregate(seed_results, cells, weights):
    out = {}

    for m, k in cells:
        setting = f"m={m},k={k}"
        rows = [r for r in seed_results if r["setting"] == setting]

        if not rows:
            continue

        out[setting] = {
            "m": m,
            "k": k,
            "n_seeds": len(rows),
            "baselines": {},
            "adaptive": {},
        }

        for method in [
            "cosamp",
            "block_score_topk",
            "learned_block_scorer",
            "one_step_refinement",
            "oracle",
        ]:
            out[setting]["baselines"][method] = {
                "nrmse": summarize(
                    [r["baselines"][method]["nrmse"]["mean"] for r in rows]
                ),
                "iou": summarize(
                    [r["baselines"][method]["iou"]["mean"] for r in rows]
                ),
            }

        for w in weights:
            key = str(w)

            out[setting]["adaptive"][key] = {
                "nrmse": summarize(
                    [r["adaptive"][key]["nrmse"]["mean"] for r in rows]
                ),
                "iou": summarize(
                    [r["adaptive"][key]["iou"]["mean"] for r in rows]
                ),
                "steps": summarize(
                    [r["adaptive"][key]["steps"]["mean"] for r in rows]
                ),
                "gains": {},
            }

            for gain_key in [
                "vs_cosamp",
                "vs_block_score_topk",
                "vs_learned_block_scorer",
                "vs_one_step_refinement",
            ]:
                out[setting]["adaptive"][key]["gains"][gain_key] = summarize(
                    [r["adaptive"][key]["gains"][gain_key] for r in rows]
                )

        best_weight = min(
            weights,
            key=lambda w: out[setting]["adaptive"][str(w)]["nrmse"]["mean"],
        )

        out[setting]["best_weight"] = float(best_weight)
        out[setting]["best_adaptive_nrmse"] = out[setting]["adaptive"][str(best_weight)]["nrmse"]["mean"]

    return out


def plot_nrmse_by_weight(agg, cells, weights, out_prefix):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    x = np.asarray(weights)

    for m, k in cells:
        setting = f"m={m},k={k}"
        if setting not in agg:
            continue

        means = [agg[setting]["adaptive"][str(w)]["nrmse"]["mean"] for w in weights]
        ses = [agg[setting]["adaptive"][str(w)]["nrmse"]["se"] for w in weights]

        ax.errorbar(x, means, yerr=ses, marker="o", capsize=3, label=setting)

    ax.set_xlabel("support-stability weight")
    ax.set_ylabel("adaptive refinement NRMSE")
    ax.set_title("Adaptive stopping robustness: NRMSE by change_weight")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / f"{out_prefix}_nrmse_by_weight.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_gain_cosamp(agg, cells, weights, out_prefix):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    x = np.asarray(weights)

    for m, k in cells:
        setting = f"m={m},k={k}"
        if setting not in agg:
            continue

        means = [
            agg[setting]["adaptive"][str(w)]["gains"]["vs_cosamp"]["mean"]
            for w in weights
        ]
        ses = [
            agg[setting]["adaptive"][str(w)]["gains"]["vs_cosamp"]["se"]
            for w in weights
        ]

        ax.errorbar(x, means, yerr=ses, marker="o", capsize=3, label=setting)

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("support-stability weight")
    ax.set_ylabel("NRMSE gain over CoSaMP")
    ax.set_title("Adaptive stopping robustness: gain over CoSaMP")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / f"{out_prefix}_gain_cosamp.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_steps_by_weight(agg, cells, weights, out_prefix):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    x = np.asarray(weights)

    for m, k in cells:
        setting = f"m={m},k={k}"
        if setting not in agg:
            continue

        means = [agg[setting]["adaptive"][str(w)]["steps"]["mean"] for w in weights]
        ses = [agg[setting]["adaptive"][str(w)]["steps"]["se"] for w in weights]

        ax.errorbar(x, means, yerr=ses, marker="o", capsize=3, label=setting)

    ax.set_xlabel("support-stability weight")
    ax.set_ylabel("accepted refinement steps")
    ax.set_title("Adaptive stopping behavior by change_weight")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / f"{out_prefix}_steps_by_weight.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def plot_heatmap(agg, cells, weights, out_prefix):
    mat = np.full((len(cells), len(weights)), np.nan)

    for i, (m, k) in enumerate(cells):
        setting = f"m={m},k={k}"
        if setting not in agg:
            continue

        for j, w in enumerate(weights):
            mat[i, j] = agg[setting]["adaptive"][str(w)]["nrmse"]["mean"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    im = ax.imshow(mat, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(len(weights)))
    ax.set_xticklabels([str(w) for w in weights])
    ax.set_yticks(np.arange(len(cells)))
    ax.set_yticklabels([f"m={m},k={k}" for m, k in cells])

    ax.set_xlabel("support-stability weight")
    ax.set_ylabel("setting")
    ax.set_title("Adaptive NRMSE by stopping weight")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("NRMSE")

    for i in range(len(cells)):
        for j in range(len(weights)):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()

    out = FIGURES_DIR / f"{out_prefix}_heatmap.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out}")


def main():
    args = parse_args()

    cells = parse_cells(args.cells)
    weights = parse_float_list(args.weights)
    seeds = parse_int_list(args.seeds)

    print("=" * 100)
    print("Adaptive stopping-weight robustness ablation")
    print("=" * 100)
    print(f"cells={cells}")
    print(f"weights={weights}")
    print(f"seeds={seeds}")
    print(f"n_train={args.n_train}, n_test={args.n_test}")
    print(f"refine_iters={args.refine_iters}, min_improvement={args.min_improvement}")

    seed_results = []

    for m, k in cells:
        for seed in seeds:
            result = run_seed(args, m=m, k=k, seed=seed, weights=weights)
            seed_results.append(result)

            partial = {
                "config": vars(args),
                "cells": cells,
                "weights": weights,
                "seeds": seeds,
                "seed_results": seed_results,
            }

            partial_path = RESULTS_DIR / f"{args.out_prefix}_partial.json"
            with partial_path.open("w") as f:
                json.dump(partial, f, indent=2)

    agg = aggregate(seed_results, cells, weights)

    out = {
        "config": vars(args),
        "cells": cells,
        "weights": weights,
        "seeds": seeds,
        "seed_results": seed_results,
        "aggregate": agg,
        "interpretation": (
            "This ablation tests sensitivity to the support-stability penalty. "
            "Lower adaptive NRMSE is better. Positive gains mean adaptive has lower "
            "NRMSE than the baseline."
        ),
    }

    out_json = RESULTS_DIR / f"{args.out_prefix}.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {out_json}")

    print("\nAggregate stopping ablation")
    print("-" * 120)
    print(
        f"{'setting':<14} {'best_w':>8} {'best_NRMSE':>11} "
        f"{'w=0':>9} {'w=0.01':>9} {'w=0.05':>9} {'w=0.1':>9} {'w=0.2':>9}"
    )

    for setting, s in agg.items():
        vals = []
        for w in weights:
            vals.append(s["adaptive"][str(w)]["nrmse"]["mean"])

        def get_w(target):
            if target in weights:
                return s["adaptive"][str(target)]["nrmse"]["mean"]
            return np.nan

        print(
            f"{setting:<14} "
            f"{s['best_weight']:>8.2f} "
            f"{s['best_adaptive_nrmse']:>11.4f} "
            f"{get_w(0.0):>9.4f} "
            f"{get_w(0.01):>9.4f} "
            f"{get_w(0.05):>9.4f} "
            f"{get_w(0.1):>9.4f} "
            f"{get_w(0.2):>9.4f}"
        )

    plot_nrmse_by_weight(agg, cells, weights, args.out_prefix)
    plot_gain_cosamp(agg, cells, weights, args.out_prefix)
    plot_steps_by_weight(agg, cells, weights, args.out_prefix)
    plot_heatmap(agg, cells, weights, args.out_prefix)


if __name__ == "__main__":
    main()
