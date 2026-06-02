"""
adaptive_phase_diagram.py

Phase diagram for adaptive learned block refinement.

Goal:
    Test whether adaptive learned block refinement works across a grid of
    measurement counts m and sparsity levels k, not only at two selected points.

This script evaluates:
    - CoSaMP
    - block_score_topk
    - learned_block_scorer
    - one_step_refinement
    - adaptive_refinement
    - oracle

Outputs:
    results/adaptive_learned_block_refinement/adaptive_phase_diagram.json

    figures/adaptive_learned_block_refinement/adaptive_phase_diagram_adaptive_nrmse.png
    figures/adaptive_learned_block_refinement/adaptive_phase_diagram_gain_cosamp.png
    figures/adaptive_learned_block_refinement/adaptive_phase_diagram_gain_block.png
    figures/adaptive_learned_block_refinement/adaptive_phase_diagram_gain_one_step.png
    figures/adaptive_learned_block_refinement/adaptive_phase_diagram_steps.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

# Reuse the implementation from the adaptive refinement experiment.
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


METHODS = [
    "cosamp",
    "block_score_topk",
    "learned_block_scorer",
    "one_step_refinement",
    "adaptive_refinement",
    "oracle",
]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--n", type=int, default=256)
    p.add_argument("--m-values", type=str, default="80,96,112,128")
    p.add_argument("--k-values", type=str, default="25,40,55,70")
    p.add_argument("--seeds", type=str, default="0,1,2")

    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-test", type=int, default=120)

    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--refine-iters", type=int, default=4)

    p.add_argument("--change-weight", type=float, default=0.05)
    p.add_argument("--min-improvement", type=float, default=1e-4)

    p.add_argument(
        "--out-prefix",
        type=str,
        default="adaptive_phase_diagram",
    )

    return p.parse_args()


def eval_support(A, y, x_true, S_true, S_pred):
    x_hat = support_lstsq(A, y, S_pred)
    return {
        "nrmse": float(nrmse(x_hat, x_true)),
        "iou": float(iou(set(S_pred), set(S_true))),
        "support_size": int(len(S_pred)),
    }


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "se": float(np.std(arr) / np.sqrt(max(len(arr), 1))),
        "median": float(np.median(arr)),
        "values": arr.tolist(),
    }


def run_cell(args, m: int, k: int, seed: int):
    """
    Run one phase-diagram cell for one (m,k,seed).
    """
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
        change_weight=args.change_weight,
        min_improvement=args.min_improvement,
    )

    A = make_gaussian_operator(m, args.n, seed)

    print("\n" + "=" * 90)
    print(f"Running cell: m={m}, k={k}, seed={seed}")
    print("=" * 90)

    print("Training learned block scorer...")
    clf = train_block_scorer(A, cell_args, rng)

    store = {
        method: {"nrmse": [], "iou": [], "support_size": []}
        for method in METHODS
    }

    adaptive_steps = []

    print("Evaluating...")

    for _ in range(args.n_test):
        x_true, S_true, _ = block_sparse_signal(
            args.n,
            k,
            args.block_size,
            rng,
        )
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

        S_adaptive, accepted_steps, _ = adaptive_learned_block_refinement(
            clf,
            A,
            y,
            k,
            args.block_size,
            max_refine_iters=args.refine_iters,
            change_weight=args.change_weight,
            min_improvement=args.min_improvement,
        )

        adaptive_steps.append(int(accepted_steps))

        supports = {
            "cosamp": S_cosamp,
            "block_score_topk": S_block,
            "learned_block_scorer": S_learned,
            "one_step_refinement": S_one_step,
            "adaptive_refinement": S_adaptive,
            "oracle": S_true,
        }

        for method, S_pred in supports.items():
            res = eval_support(A, y, x_true, S_true, S_pred)
            for metric in ["nrmse", "iou", "support_size"]:
                store[method][metric].append(res[metric])

    summary = {
        "config": {
            "n": args.n,
            "m": m,
            "k": k,
            "seed": seed,
            "block_size": args.block_size,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "noise_std": args.noise_std,
            "refine_iters": args.refine_iters,
            "change_weight": args.change_weight,
            "min_improvement": args.min_improvement,
        },
        "methods": {},
        "adaptive_steps": summarize(adaptive_steps),
    }

    for method in METHODS:
        summary["methods"][method] = {
            metric: summarize(store[method][metric])
            for metric in ["nrmse", "iou", "support_size"]
        }

    # Gains: positive means adaptive has lower NRMSE.
    adaptive_nrmse = summary["methods"]["adaptive_refinement"]["nrmse"]["mean"]
    summary["gains"] = {}

    for baseline in [
        "cosamp",
        "block_score_topk",
        "learned_block_scorer",
        "one_step_refinement",
    ]:
        base_nrmse = summary["methods"][baseline]["nrmse"]["mean"]
        summary["gains"][f"adaptive_vs_{baseline}"] = float(base_nrmse - adaptive_nrmse)

    print("\nCell summary")
    print("-" * 90)
    print(
        f"{'method':<26} {'NRMSE':>10} {'IoU':>10} {'support':>10}"
    )
    for method in METHODS:
        nrm = summary["methods"][method]["nrmse"]["mean"]
        io = summary["methods"][method]["iou"]["mean"]
        supp = summary["methods"][method]["support_size"]["mean"]
        print(f"{method:<26} {nrm:>10.4f} {io:>10.4f} {supp:>10.2f}")

    print(
        f"accepted steps mean={summary['adaptive_steps']['mean']:.3f}, "
        f"median={summary['adaptive_steps']['median']:.3f}"
    )
    print(
        f"gain adaptive vs CoSaMP={summary['gains']['adaptive_vs_cosamp']:+.4f}, "
        f"vs block={summary['gains']['adaptive_vs_block_score_topk']:+.4f}, "
        f"vs one_step={summary['gains']['adaptive_vs_one_step_refinement']:+.4f}"
    )

    return summary


def aggregate_cells(cell_results, m_values, k_values):
    """
    Aggregate over seeds for each (m,k).
    """
    aggregate = {}

    for m in m_values:
        for k in k_values:
            key = f"m={m},k={k}"
            cells = [
                c for c in cell_results
                if c["config"]["m"] == m and c["config"]["k"] == k
            ]

            if not cells:
                continue

            aggregate[key] = {
                "m": m,
                "k": k,
                "n_seeds": len(cells),
                "methods": {},
                "gains": {},
                "adaptive_steps": summarize(
                    [c["adaptive_steps"]["mean"] for c in cells]
                ),
            }

            for method in METHODS:
                aggregate[key]["methods"][method] = {
                    "nrmse": summarize(
                        [c["methods"][method]["nrmse"]["mean"] for c in cells]
                    ),
                    "iou": summarize(
                        [c["methods"][method]["iou"]["mean"] for c in cells]
                    ),
                }

            for gain_key in [
                "adaptive_vs_cosamp",
                "adaptive_vs_block_score_topk",
                "adaptive_vs_learned_block_scorer",
                "adaptive_vs_one_step_refinement",
            ]:
                aggregate[key]["gains"][gain_key] = summarize(
                    [c["gains"][gain_key] for c in cells]
                )

    return aggregate


def make_matrix(aggregate, m_values, k_values, getter):
    """
    Build heatmap matrix with rows=k and columns=m.
    """
    mat = np.full((len(k_values), len(m_values)), np.nan)

    for i, k in enumerate(k_values):
        for j, m in enumerate(m_values):
            key = f"m={m},k={k}"
            if key in aggregate:
                mat[i, j] = getter(aggregate[key])

    return mat


def plot_heatmap(
    mat,
    m_values,
    k_values,
    title,
    colorbar_label,
    out_path,
    annotate=True,
):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    im = ax.imshow(mat, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(m_values)))
    ax.set_xticklabels(m_values)
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_yticklabels(k_values)

    ax.set_xlabel("measurements m")
    ax.set_ylabel("sparsity k")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    if annotate:
        for i in range(len(k_values)):
            for j in range(len(m_values)):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")


def plot_all(aggregate, m_values, k_values, out_prefix):
    adaptive_nrmse = make_matrix(
        aggregate,
        m_values,
        k_values,
        lambda x: x["methods"]["adaptive_refinement"]["nrmse"]["mean"],
    )

    gain_cosamp = make_matrix(
        aggregate,
        m_values,
        k_values,
        lambda x: x["gains"]["adaptive_vs_cosamp"]["mean"],
    )

    gain_block = make_matrix(
        aggregate,
        m_values,
        k_values,
        lambda x: x["gains"]["adaptive_vs_block_score_topk"]["mean"],
    )

    gain_one_step = make_matrix(
        aggregate,
        m_values,
        k_values,
        lambda x: x["gains"]["adaptive_vs_one_step_refinement"]["mean"],
    )

    accepted_steps = make_matrix(
        aggregate,
        m_values,
        k_values,
        lambda x: x["adaptive_steps"]["mean"],
    )

    plot_heatmap(
        adaptive_nrmse,
        m_values,
        k_values,
        title="Adaptive refinement NRMSE",
        colorbar_label="NRMSE",
        out_path=FIGURES_DIR / f"{out_prefix}_adaptive_nrmse.png",
    )

    plot_heatmap(
        gain_cosamp,
        m_values,
        k_values,
        title="Adaptive gain over CoSaMP",
        colorbar_label="CoSaMP NRMSE - adaptive NRMSE",
        out_path=FIGURES_DIR / f"{out_prefix}_gain_cosamp.png",
    )

    plot_heatmap(
        gain_block,
        m_values,
        k_values,
        title="Adaptive gain over block_score_topk",
        colorbar_label="block_score NRMSE - adaptive NRMSE",
        out_path=FIGURES_DIR / f"{out_prefix}_gain_block.png",
    )

    plot_heatmap(
        gain_one_step,
        m_values,
        k_values,
        title="Adaptive gain over one_step_refinement",
        colorbar_label="one_step NRMSE - adaptive NRMSE",
        out_path=FIGURES_DIR / f"{out_prefix}_gain_one_step.png",
    )

    plot_heatmap(
        accepted_steps,
        m_values,
        k_values,
        title="Accepted adaptive refinement steps",
        colorbar_label="accepted steps",
        out_path=FIGURES_DIR / f"{out_prefix}_steps.png",
    )


def main():
    args = parse_args()

    m_values = parse_int_list(args.m_values)
    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)

    print("=" * 100)
    print("Adaptive learned block-refinement phase diagram")
    print("=" * 100)
    print(f"n={args.n}")
    print(f"m_values={m_values}")
    print(f"k_values={k_values}")
    print(f"seeds={seeds}")
    print(f"n_train={args.n_train}, n_test={args.n_test}")
    print(f"refine_iters={args.refine_iters}")
    print(f"change_weight={args.change_weight}, min_improvement={args.min_improvement}")

    cell_results = []

    for m in m_values:
        for k in k_values:
            for seed in seeds:
                cell = run_cell(args, m=m, k=k, seed=seed)
                cell_results.append(cell)

                # Save partial result after every cell.
                partial = {
                    "config": vars(args),
                    "m_values": m_values,
                    "k_values": k_values,
                    "seeds": seeds,
                    "cell_results": cell_results,
                }

                partial_path = RESULTS_DIR / f"{args.out_prefix}_partial.json"
                with partial_path.open("w") as f:
                    json.dump(partial, f, indent=2)

    aggregate = aggregate_cells(cell_results, m_values, k_values)

    out = {
        "config": vars(args),
        "m_values": m_values,
        "k_values": k_values,
        "seeds": seeds,
        "cell_results": cell_results,
        "aggregate": aggregate,
        "interpretation": (
            "Positive gains mean adaptive_refinement has lower NRMSE than the baseline. "
            "Heatmaps are aggregated over seeds."
        ),
    }

    out_json = RESULTS_DIR / f"{args.out_prefix}.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {out_json}")

    print("\nAggregate phase diagram table")
    print("-" * 120)
    print(
        f"{'setting':<14} {'n':>3} {'adapt':>9} {'cosamp':>9} "
        f"{'block':>9} {'one_step':>9} {'gain C':>9} {'gain B':>9} "
        f"{'gain 1':>9} {'steps':>8}"
    )

    for key, val in aggregate.items():
        adapt = val["methods"]["adaptive_refinement"]["nrmse"]["mean"]
        cos = val["methods"]["cosamp"]["nrmse"]["mean"]
        block = val["methods"]["block_score_topk"]["nrmse"]["mean"]
        one = val["methods"]["one_step_refinement"]["nrmse"]["mean"]

        gain_c = val["gains"]["adaptive_vs_cosamp"]["mean"]
        gain_b = val["gains"]["adaptive_vs_block_score_topk"]["mean"]
        gain_1 = val["gains"]["adaptive_vs_one_step_refinement"]["mean"]
        steps = val["adaptive_steps"]["mean"]

        print(
            f"{key:<14} {val['n_seeds']:>3d} "
            f"{adapt:>9.4f} {cos:>9.4f} {block:>9.4f} {one:>9.4f} "
            f"{gain_c:>+9.4f} {gain_b:>+9.4f} {gain_1:>+9.4f} "
            f"{steps:>8.3f}"
        )

    plot_all(aggregate, m_values, k_values, args.out_prefix)


if __name__ == "__main__":
    main()
