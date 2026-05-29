"""
structured_priors.py

Structured-prior sparse recovery experiment.

Goal:
    Test whether support structure creates regimes where prior-aware recovery
    can outperform generic sparse-recovery algorithms.

Signal families:
    1. iid_sparse      : random support
    2. block_sparse    : support appears in contiguous blocks
    3. cluster_sparse  : support indices cluster near a few centers
    4. markov_sparse   : support follows a simple Markov activation process

Methods:
    - naive top-k correlation
    - OMP
    - CoSaMP
    - HTP
    - smoothed top-k correlation, a simple cluster-prior baseline
    - block-score top-k, a simple block-prior baseline
    - oracle support + LS

Outputs:
    results/structured_priors/structured_priors.json
    figures/structured_priors/structured_priors_nrmse.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "structured_priors"
FIGURES_DIR = ROOT / "figures" / "structured_priors"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--cluster-radius", type=int, default=3)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--out-prefix", type=str, default="structured_priors")
    return p.parse_args()


# ---------------------------------------------------------------------
# Operators and signals
# ---------------------------------------------------------------------

def normalize_columns(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), eps)


def make_gaussian_operator(m: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)).astype(np.float64)
    return normalize_columns(A)


def fill_amplitudes(n: int, support: set[int], rng: np.random.Generator):
    x = np.zeros(n, dtype=np.float64)
    support_list = sorted(support)
    amps = rng.uniform(0.5, 2.0, size=len(support_list)) * rng.choice(
        [-1.0, 1.0], size=len(support_list)
    )
    x[support_list] = amps
    return x


def iid_sparse_signal(n: int, k: int, rng: np.random.Generator):
    support = set(int(i) for i in rng.choice(n, size=k, replace=False))
    x = fill_amplitudes(n, support, rng)
    return x, support


def block_sparse_signal(n: int, k: int, block_size: int, rng: np.random.Generator):
    n_blocks = n // block_size
    blocks_needed = int(np.ceil(k / block_size))
    chosen_blocks = rng.choice(n_blocks, size=blocks_needed, replace=False)

    support = []
    for b in chosen_blocks:
        start = int(b * block_size)
        support.extend(range(start, min(start + block_size, n)))

    support = set(int(i) for i in support[:k])
    x = fill_amplitudes(n, support, rng)
    return x, support


def cluster_sparse_signal(n: int, k: int, radius: int, rng: np.random.Generator):
    support = set()
    n_centers = max(1, int(np.ceil(k / (2 * radius + 1))))

    centers = rng.choice(n, size=n_centers, replace=False)
    for c in centers:
        for j in range(int(c) - radius, int(c) + radius + 1):
            if 0 <= j < n:
                support.add(int(j))
            if len(support) >= k:
                break
        if len(support) >= k:
            break

    while len(support) < k:
        support.add(int(rng.integers(0, n)))

    support = set(sorted(support)[:k])
    x = fill_amplitudes(n, support, rng)
    return x, support


def markov_sparse_signal(n: int, k: int, rng: np.random.Generator):
    """
    Simple Markov-style support:
    once active, nearby continuation is more likely.
    """
    support = set()
    pos = int(rng.integers(0, n))

    while len(support) < k:
        support.add(pos)

        if rng.random() < 0.75:
            step = int(rng.choice([-2, -1, 1, 2]))
            pos = int(np.clip(pos + step, 0, n - 1))
        else:
            pos = int(rng.integers(0, n))

    x = fill_amplitudes(n, support, rng)
    return x, support


def make_signal(family: str, n: int, k: int, args, rng: np.random.Generator):
    if family == "iid_sparse":
        return iid_sparse_signal(n, k, rng)
    if family == "block_sparse":
        return block_sparse_signal(n, k, args.block_size, rng)
    if family == "cluster_sparse":
        return cluster_sparse_signal(n, k, args.cluster_radius, rng)
    if family == "markov_sparse":
        return markov_sparse_signal(n, k, rng)
    raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------
# Metrics and utilities
# ---------------------------------------------------------------------

def best_k_support(scores: np.ndarray, k: int) -> set[int]:
    k_eff = int(max(1, min(k, scores.size)))
    idx = np.argpartition(-np.abs(scores), k_eff - 1)[:k_eff]
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


def precision_recall(S_pred: set[int], S_true: set[int]):
    precision = len(S_pred & S_true) / max(len(S_pred), 1)
    recall = len(S_pred & S_true) / max(len(S_true), 1)
    return precision, recall


def summarize(vals):
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def eval_support(A, y, x_true, S_true, S_pred):
    x_hat = support_lstsq(A, y, S_pred)
    p, r = precision_recall(S_pred, S_true)
    return {
        "nrmse": nrmse(x_hat, x_true),
        "iou": iou(S_pred, S_true),
        "precision": p,
        "recall": r,
    }


# ---------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------

def naive_topk(A, y, k):
    return best_k_support(A.T @ y, k)


def omp(A, y, k):
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

    return set(int(i) for i in selected)


def cosamp(A, y, k, max_iters=30, tol=1e-10):
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

    return S_prev


def htp(A, y, k, max_iters=100):
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

    return S_prev


def smoothed_topk(A, y, k, radius=3):
    """
    Simple cluster-prior baseline:
    smooth correlation scores locally before top-k selection.
    """
    raw = np.abs(A.T @ y)
    smoothed = raw.copy()

    for j in range(raw.size):
        lo = max(0, j - radius)
        hi = min(raw.size, j + radius + 1)
        smoothed[j] = np.mean(raw[lo:hi])

    return best_k_support(smoothed, k)


def block_score_topk(A, y, k, block_size=5):
    """
    Simple block-prior baseline:
    score a block by total correlation energy, then select top coordinates
    inside the best blocks.
    """
    raw = np.abs(A.T @ y)
    n = raw.size
    n_blocks = int(np.ceil(n / block_size))

    block_scores = []
    for b in range(n_blocks):
        lo = b * block_size
        hi = min(n, lo + block_size)
        block_scores.append(float(np.sum(raw[lo:hi] ** 2)))

    block_order = np.argsort(block_scores)[::-1]

    selected = []
    for b in block_order:
        lo = int(b * block_size)
        hi = min(n, lo + block_size)
        coords = list(range(lo, hi))
        coords = sorted(coords, key=lambda j: raw[j], reverse=True)

        for j in coords:
            selected.append(j)
            if len(selected) >= k:
                return set(int(i) for i in selected)

    return set(int(i) for i in selected[:k])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    A = make_gaussian_operator(args.m, args.n, seed=args.seed)

    families = ["iid_sparse", "block_sparse", "cluster_sparse", "markov_sparse"]
    methods = [
        "naive",
        "omp",
        "cosamp",
        "htp",
        "smoothed_topk",
        "block_score_topk",
        "oracle",
    ]

    store = {
        fam: {
            method: {"nrmse": [], "iou": [], "precision": [], "recall": []}
            for method in methods
        }
        for fam in families
    }

    print("=" * 78)
    print("Structured-prior sparse recovery")
    print("=" * 78)
    print(f"n={args.n}, m={args.m}, k={args.k}, n_test={args.n_test}")
    print(f"noise_std={args.noise_std}")

    for fam in families:
        print(f"\nRunning family: {fam}")

        for _ in range(args.n_test):
            x_true, S_true = make_signal(fam, args.n, args.k, args, rng)
            y = A @ x_true

            if args.noise_std > 0:
                y = y + args.noise_std * rng.standard_normal(args.m)

            predicted = {
                "naive": naive_topk(A, y, args.k),
                "omp": omp(A, y, args.k),
                "cosamp": cosamp(A, y, args.k, max_iters=args.max_iters),
                "htp": htp(A, y, args.k),
                "smoothed_topk": smoothed_topk(A, y, args.k, radius=args.cluster_radius),
                "block_score_topk": block_score_topk(A, y, args.k, block_size=args.block_size),
                "oracle": S_true,
            }

            for method, S_pred in predicted.items():
                res = eval_support(A, y, x_true, S_true, S_pred)
                for key in ["nrmse", "iou", "precision", "recall"]:
                    store[fam][method][key].append(res[key])

    summary = {
        "config": vars(args),
        "families": families,
        "methods": methods,
        "summary": {},
        "interpretation": (
            "If prior-aware simple baselines such as smoothed_topk or "
            "block_score_topk improve over generic top-k/greedy methods on "
            "structured supports, this suggests that learned prior-aware "
            "support recovery may have a clearer advantage on structured priors."
        ),
    }

    for fam in families:
        summary["summary"][fam] = {}
        print("\n" + "-" * 78)
        print(f"Family: {fam}")
        print("-" * 78)
        print(f"{'method':<18} {'NRMSE':>18} {'IoU':>18}")

        for method in methods:
            summary["summary"][fam][method] = {
                metric: summarize(store[fam][method][metric])
                for metric in ["nrmse", "iou", "precision", "recall"]
            }

            nrm = summary["summary"][fam][method]["nrmse"]
            iou_s = summary["summary"][fam][method]["iou"]

            print(
                f"{method:<18} "
                f"{nrm['mean']:>8.4f} ± {nrm['std']:<7.4f} "
                f"{iou_s['mean']:>8.4f} ± {iou_s['std']:<7.4f}"
            )

    out_json = RESULTS_DIR / f"{args.out_prefix}.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_json}")

    # Plot NRMSE by family.
    fig, ax = plt.subplots(figsize=(10.0, 4.8))

    x = np.arange(len(families))
    width = 0.11

    for i, method in enumerate(methods):
        means = [summary["summary"][fam][method]["nrmse"]["mean"] for fam in families]
        ax.bar(x + (i - len(methods) / 2) * width, means, width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15)
    ax.set_ylabel("NRMSE")
    ax.set_title("Structured-prior recovery: NRMSE by signal family")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    out_png = FIGURES_DIR / f"{args.out_prefix}_nrmse.png"
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
