"""
ceiling_study_small_n.py

Goal:
    Test whether the hard sparse-recovery region is algorithmically hard
    or information/statistically ambiguous.

Question:
    When OMP/CoSaMP/HTP fail, does exact L0 search still succeed?

Interpretation:
    - If exact L0 succeeds but CoSaMP/OMP/HTP fail:
        There is algorithmic headroom.
    - If exact L0 also fails or finds many equally good supports:
        We may be near a task/information ceiling.
    - If oracle support + LS succeeds but exact L0 is ambiguous:
        The true support is recoverable only with privileged support knowledge,
        not reliably from y alone.

Outputs:
    results/ceiling/ceiling_study_small_n.json
    figures/ceiling/ceiling_study_small_n.png
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "ceiling"
FIGURES_DIR = ROOT / "figures" / "ceiling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--m", type=int, default=10)
    p.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-combos", type=int, default=50000,
                   help="Skip exact L0 if binom(n,k) exceeds this.")
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--out-prefix", type=str, default="ceiling_study_small_n")
    return p.parse_args()


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------

def normalize_columns(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), eps)


def make_gaussian_operator(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)).astype(np.float64)
    return normalize_columns(A)


def make_sparse_signal(n: int, k: int, rng: np.random.Generator):
    x = np.zeros(n, dtype=np.float64)
    support = rng.choice(n, size=k, replace=False)
    amplitudes = rng.uniform(0.5, 2.0, size=k) * rng.choice([-1.0, 1.0], size=k)
    x[support] = amplitudes
    return x, set(int(i) for i in support)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def best_k_support(scores: np.ndarray, k: int) -> set[int]:
    idx = np.argpartition(-np.abs(scores), k - 1)[:k]
    return set(int(i) for i in idx)


def support_lstsq(A: np.ndarray, y: np.ndarray, support: Iterable[int]) -> np.ndarray:
    n = A.shape[1]
    support = sorted(int(i) for i in support)
    x_hat = np.zeros(n, dtype=np.float64)

    if len(support) == 0:
        return x_hat

    x_s, *_ = np.linalg.lstsq(A[:, support], y, rcond=None)
    x_hat[support] = x_s
    return x_hat


def nrmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.linalg.norm(x_hat - x_true) / max(np.linalg.norm(x_true), 1e-12))


def iou(S_pred: set[int], S_true: set[int]) -> float:
    union = len(S_pred | S_true)
    return len(S_pred & S_true) / union if union else 1.0


def residual_score(A: np.ndarray, y: np.ndarray, support: Iterable[int]) -> float:
    x_hat = support_lstsq(A, y, support)
    return float(np.linalg.norm(y - A @ x_hat) ** 2)


# ---------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------

def naive_topk(A: np.ndarray, y: np.ndarray, k: int):
    S = best_k_support(A.T @ y, k)
    x_hat = support_lstsq(A, y, S)
    return S, x_hat


def omp(A: np.ndarray, y: np.ndarray, k: int):
    residual = y.copy()
    selected = []

    for _ in range(k):
        scores = np.abs(A.T @ residual)
        if selected:
            scores[selected] = -np.inf

        j = int(np.argmax(scores))
        selected.append(j)

        x_s, *_ = np.linalg.lstsq(A[:, selected], y, rcond=None)
        residual = y - A[:, selected] @ x_s

    S = set(selected)
    x_hat = support_lstsq(A, y, S)
    return S, x_hat


def cosamp(A: np.ndarray, y: np.ndarray, k: int, max_iters: int = 30, tol: float = 1e-10):
    n = A.shape[1]
    x = np.zeros(n)
    S_prev = set()
    prev_res_norm = np.inf

    for _ in range(max_iters):
        r = y - A @ x
        res_norm = float(np.linalg.norm(r))

        if res_norm < tol:
            break

        u = A.T @ r
        omega = best_k_support(u, min(2 * k, n))

        current_support = set(int(i) for i in np.nonzero(np.abs(x) > 0)[0])
        T = sorted(omega | current_support)

        b_t, *_ = np.linalg.lstsq(A[:, T], y, rcond=None)
        b = np.zeros(n)
        b[T] = b_t

        S_new = best_k_support(b, k)
        x = support_lstsq(A, y, S_new)

        if S_new == S_prev and S_prev:
            break

        if np.isfinite(prev_res_norm):
            improvement = prev_res_norm - res_norm
            if improvement >= 0.0 and improvement <= tol * max(1.0, prev_res_norm):
                break

        prev_res_norm = res_norm
        S_prev = S_new

    if not S_prev:
        return naive_topk(A, y, k)

    return S_prev, support_lstsq(A, y, S_prev)


def htp(A: np.ndarray, y: np.ndarray, k: int, max_iters: int = 100):
    n = A.shape[1]
    spec = float(np.linalg.norm(A, 2))
    step = 0.95 / max(spec ** 2, 1e-12)

    x = np.zeros(n)
    S_prev = set()

    for _ in range(max_iters):
        x_aux = x + step * A.T @ (y - A @ x)
        S_new = best_k_support(x_aux, k)

        x = support_lstsq(A, y, S_new)

        if S_new == S_prev:
            break

        S_prev = S_new

    return S_prev, support_lstsq(A, y, S_prev)


def exact_l0_search(A: np.ndarray, y: np.ndarray, k: int, max_combos: int):
    """
    Exhaustive L0 support search:
        argmin_{|S|=k} ||y - P_{A_S} y||_2^2

    Returns:
        support, x_hat, best_residual, second_best_residual, n_combos
    """
    n = A.shape[1]
    n_combos = math.comb(n, k)

    if n_combos > max_combos:
        return None, None, np.nan, np.nan, n_combos

    best_S = None
    best_res = np.inf
    second_res = np.inf

    for comb in itertools.combinations(range(n), k):
        res = residual_score(A, y, comb)

        if res < best_res:
            second_res = best_res
            best_res = res
            best_S = set(int(i) for i in comb)
        elif res < second_res:
            second_res = res

    x_hat = support_lstsq(A, y, best_S)
    return best_S, x_hat, float(best_res), float(second_res), n_combos


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_method(name, fn, A, y, x_true, S_true, k):
    S_pred, x_hat = fn(A, y, k)
    return {
        "method": name,
        "nrmse": nrmse(x_hat, x_true),
        "iou": iou(S_pred, S_true),
        "residual": residual_score(A, y, S_pred),
    }


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite))}


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    A = make_gaussian_operator(args.m, args.n, seed=args.seed)

    print("=" * 78)
    print(" Small-n ceiling study")
    print("=" * 78)
    print(f"n={args.n}, m={args.m}, k_values={args.k_values}, n_test={args.n_test}")
    print(f"noise_std={args.noise_std}, max_combos={args.max_combos}")

    all_results = {
        "config": vars(args),
        "interpretation": {
            "exact_l0_succeeds_but_algorithms_fail": "algorithmic headroom",
            "exact_l0_also_fails_or_ambiguous": "possible information/statistical ceiling",
            "oracle_succeeds_but_exact_l0_ambiguous": "privileged support knowledge helps but y alone is ambiguous",
        },
        "by_k": {},
    }

    methods = {
        "naive": naive_topk,
        "omp": omp,
        "cosamp": cosamp,
        "htp": htp,
    }

    for k in args.k_values:
        print("\n" + "-" * 78)
        print(f"k={k}, binom(n,k)={math.comb(args.n, k)}")
        print("-" * 78)

        store = {
            name: {"nrmse": [], "iou": [], "residual": []}
            for name in ["naive", "omp", "cosamp", "htp", "exact_l0", "oracle"]
        }
        ambiguity_gaps = []
        exact_l0_combos = math.comb(args.n, k)
        exact_l0_skipped = exact_l0_combos > args.max_combos

        for t in range(args.n_test):
            x_true, S_true = make_sparse_signal(args.n, k, rng)
            y = A @ x_true

            if args.noise_std > 0:
                y = y + args.noise_std * rng.standard_normal(args.m)

            for name, fn in methods.items():
                out = evaluate_method(name, fn, A, y, x_true, S_true, k)
                store[name]["nrmse"].append(out["nrmse"])
                store[name]["iou"].append(out["iou"])
                store[name]["residual"].append(out["residual"])

            # Oracle true support + LS
            x_oracle = support_lstsq(A, y, S_true)
            store["oracle"]["nrmse"].append(nrmse(x_oracle, x_true))
            store["oracle"]["iou"].append(1.0)
            store["oracle"]["residual"].append(residual_score(A, y, S_true))

            # Exact L0 search
            S_l0, x_l0, best_res, second_res, n_combos = exact_l0_search(
                A, y, k, max_combos=args.max_combos
            )

            if S_l0 is None:
                store["exact_l0"]["nrmse"].append(np.nan)
                store["exact_l0"]["iou"].append(np.nan)
                store["exact_l0"]["residual"].append(np.nan)
                ambiguity_gaps.append(np.nan)
            else:
                store["exact_l0"]["nrmse"].append(nrmse(x_l0, x_true))
                store["exact_l0"]["iou"].append(iou(S_l0, S_true))
                store["exact_l0"]["residual"].append(best_res)
                ambiguity_gaps.append(second_res - best_res)

        summary = {}
        for name, metrics in store.items():
            summary[name] = {
                "nrmse": summarize(metrics["nrmse"]),
                "iou": summarize(metrics["iou"]),
                "residual": summarize(metrics["residual"]),
            }

        summary["exact_l0_info"] = {
            "n_combinations": exact_l0_combos,
            "skipped": exact_l0_skipped,
            "ambiguity_gap": summarize(ambiguity_gaps),
        }

        all_results["by_k"][str(k)] = summary

        print(f"{'method':<10} {'NRMSE':>18} {'IoU':>18} {'residual':>18}")
        for name in ["naive", "omp", "cosamp", "htp", "exact_l0", "oracle"]:
            s = summary[name]
            print(
                f"{name:<10} "
                f"{s['nrmse']['mean']:>8.4f} ± {s['nrmse']['std']:<7.4f} "
                f"{s['iou']['mean']:>8.4f} ± {s['iou']['std']:<7.4f} "
                f"{s['residual']['mean']:>8.2e} ± {s['residual']['std']:<7.2e}"
            )

        gap = summary["exact_l0_info"]["ambiguity_gap"]
        print(
            f"Exact L0 combinations={exact_l0_combos}, skipped={exact_l0_skipped}, "
            f"ambiguity_gap={gap['mean']:.2e} ± {gap['std']:.2e}"
        )

    # Save JSON
    out_json = RESULTS_DIR / f"{args.out_prefix}.json"
    with out_json.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nWrote {out_json}")

    # Plot NRMSE curves
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    k_vals = args.k_values
    plot_methods = ["naive", "omp", "cosamp", "htp", "exact_l0", "oracle"]

    for name in plot_methods:
        ys = [all_results["by_k"][str(k)][name]["nrmse"]["mean"] for k in k_vals]
        ax.plot(k_vals, ys, marker="o", label=name)

    ax.set_xlabel("sparsity k")
    ax.set_ylabel("NRMSE")
    ax.set_title(f"Small-n ceiling study: n={args.n}, m={args.m}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_png = FIGURES_DIR / f"{args.out_prefix}.png"
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
