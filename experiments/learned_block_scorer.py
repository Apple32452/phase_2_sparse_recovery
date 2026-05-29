"""
learned_block_scorer.py

Learned block-level support recovery experiment.

Motivation:
    Previous structured-prior experiments showed that block_score_topk can beat
    CoSaMP in hard block-sparse regimes. This script tests whether a learned
    block scorer can improve or match that structured-prior baseline.

Main idea:
    Instead of scoring individual coordinates independently, predict which
    blocks are active, then select coordinates inside the predicted active
    blocks.

Methods:
    - naive top-k correlation + LS
    - CoSaMP + LS
    - block_score_topk + LS
    - learned_block_scorer + LS
    - oracle support + LS

Default hard regime:
    n = 256
    m = 96
    k = 40
    block_size = 5

Outputs:
    results/learned_block_scorer/<prefix>.json
    figures/learned_block_scorer/<prefix>_nrmse.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "learned_block_scorer"
FIGURES_DIR = ROOT / "figures" / "learned_block_scorer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--m", type=int, default=96)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--out-prefix", type=str, default="learned_block_scorer_m96_k40")
    return p.parse_args()


# ---------------------------------------------------------------------
# Operators and data
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

    amps = rng.uniform(0.5, 2.0, size=len(support_list))
    signs = rng.choice([-1.0, 1.0], size=len(support_list))
    x[support_list] = amps * signs

    return x


def block_sparse_signal(n: int, k: int, block_size: int, rng: np.random.Generator):
    """
    Generate an exactly k-sparse signal with full block structure.

    Uses only full blocks so the final partial block does not create
    supports smaller than k.
    """
    n_full_blocks = n // block_size
    blocks_needed = int(np.ceil(k / block_size))

    support = set()
    chosen_blocks = set()

    while len(support) < k:
        available = [b for b in range(n_full_blocks) if b not in chosen_blocks]
        if not available:
            break

        b = int(rng.choice(available))
        chosen_blocks.add(b)

        lo = b * block_size
        hi = lo + block_size

        for j in range(lo, hi):
            support.add(int(j))
            if len(support) >= k:
                break

    support = set(sorted(support)[:k])
    x = fill_amplitudes(n, support, rng)

    return x, support, chosen_blocks


def add_noise(y: np.ndarray, noise_std: float, rng: np.random.Generator):
    if noise_std <= 0:
        return y
    return y + noise_std * rng.standard_normal(y.shape)


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
        "support_size": len(S_pred),
    }


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def naive_topk(A, y, k):
    return best_k_support(A.T @ y, k)


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


def block_score_topk(A, y, k, block_size=5):
    """
    Hand-designed block prior:
    rank blocks by total correlation energy, then select coordinates
    inside the strongest blocks.
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
# Learned block scorer
# ---------------------------------------------------------------------

def block_features(A, y, block_size=5):
    """
    Create one feature vector per block.

    Features include:
        - block correlation energy
        - mean/max/std/sum of absolute correlations in block
        - neighboring block scores
        - simple coherence statistics inside block
    """
    raw = np.abs(A.T @ y)
    n = raw.size
    n_blocks = int(np.ceil(n / block_size))

    scale = max(float(np.max(raw)), 1e-12)
    raw_norm = raw / scale

    G = np.abs(A.T @ A)
    np.fill_diagonal(G, 0.0)

    block_energy = []
    block_mean = []
    block_max = []
    block_std = []
    block_sum = []
    block_coh_mean = []
    block_coh_max = []

    for b in range(n_blocks):
        lo = b * block_size
        hi = min(n, lo + block_size)

        vals = raw_norm[lo:hi]
        subG = G[lo:hi, lo:hi]

        block_energy.append(float(np.sum(vals ** 2)))
        block_mean.append(float(np.mean(vals)))
        block_max.append(float(np.max(vals)))
        block_std.append(float(np.std(vals)))
        block_sum.append(float(np.sum(vals)))

        if subG.size > 0:
            block_coh_mean.append(float(np.mean(subG)))
            block_coh_max.append(float(np.max(subG)))
        else:
            block_coh_mean.append(0.0)
            block_coh_max.append(0.0)

    block_energy = np.asarray(block_energy)
    block_mean = np.asarray(block_mean)
    block_max = np.asarray(block_max)
    block_std = np.asarray(block_std)
    block_sum = np.asarray(block_sum)
    block_coh_mean = np.asarray(block_coh_mean)
    block_coh_max = np.asarray(block_coh_max)

    feats = []

    for b in range(n_blocks):
        left = max(0, b - 1)
        right = min(n_blocks - 1, b + 1)

        neighbor_energy = block_energy[left:right + 1]
        neighbor_sum = block_sum[left:right + 1]

        feats.append(
            [
                block_energy[b],
                block_mean[b],
                block_max[b],
                block_std[b],
                block_sum[b],
                float(np.mean(neighbor_energy)),
                float(np.max(neighbor_energy)),
                float(np.sum(neighbor_energy)),
                float(np.mean(neighbor_sum)),
                float(np.max(neighbor_sum)),
                block_coh_mean[b],
                block_coh_max[b],
                b / max(n_blocks - 1, 1),
                float(np.linalg.norm(y)),
            ]
        )

    return np.asarray(feats, dtype=np.float64)


def train_block_scorer(A, args, rng):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from e

    X_all = []
    y_all = []

    n_blocks = int(np.ceil(args.n / args.block_size))

    for _ in range(args.n_train):
        x, S, active_blocks = block_sparse_signal(args.n, args.k, args.block_size, rng)
        y = add_noise(A @ x, args.noise_std, rng)

        X = block_features(A, y, args.block_size)
        labels = np.zeros(n_blocks, dtype=np.int64)

        for b in active_blocks:
            if 0 <= b < n_blocks:
                labels[b] = 1

        X_all.append(X)
        y_all.append(labels)

    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    return clf


def learned_block_support(clf, A, y, k, block_size=5):
    raw = np.abs(A.T @ y)
    n = raw.size
    n_blocks = int(np.ceil(n / block_size))
    blocks_needed = int(np.ceil(k / block_size))

    X = block_features(A, y, block_size)
    probs = clf.predict_proba(X)[:, 1]

    block_order = np.argsort(probs)[::-1]
    selected_blocks = block_order[:blocks_needed]

    selected = []
    for b in selected_blocks:
        lo = int(b * block_size)
        hi = min(n, lo + block_size)

        coords = list(range(lo, hi))
        coords = sorted(coords, key=lambda j: raw[j], reverse=True)

        for j in coords:
            selected.append(j)

    # If k is not divisible by block_size, prune within selected blocks by raw score.
    selected = sorted(set(selected), key=lambda j: raw[j], reverse=True)
    return set(int(i) for i in selected[:k])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    A = make_gaussian_operator(args.m, args.n, seed=args.seed)

    print("=" * 78)
    print("Learned block scorer")
    print("=" * 78)
    print(f"n={args.n}, m={args.m}, k={args.k}, block_size={args.block_size}")
    print(f"n_train={args.n_train}, n_test={args.n_test}, noise_std={args.noise_std}")

    print("\nTraining learned block scorer...")
    clf = train_block_scorer(A, args, rng)

    methods = [
        "naive",
        "cosamp",
        "block_score_topk",
        "learned_block_scorer",
        "oracle",
    ]

    store = {
        method: {"nrmse": [], "iou": [], "precision": [], "recall": [], "support_size": []}
        for method in methods
    }

    print("\nEvaluating on block-sparse test signals...")

    for _ in range(args.n_test):
        x_true, S_true, _ = block_sparse_signal(args.n, args.k, args.block_size, rng)
        y = add_noise(A @ x_true, args.noise_std, rng)

        predicted = {
            "naive": naive_topk(A, y, args.k),
            "cosamp": cosamp(A, y, args.k, max_iters=args.max_iters),
            "block_score_topk": block_score_topk(A, y, args.k, block_size=args.block_size),
            "learned_block_scorer": learned_block_support(clf, A, y, args.k, block_size=args.block_size),
            "oracle": S_true,
        }

        for method, S_pred in predicted.items():
            res = eval_support(A, y, x_true, S_true, S_pred)
            for metric in ["nrmse", "iou", "precision", "recall", "support_size"]:
                store[method][metric].append(res[metric])

    summary = {
        "config": vars(args),
        "methods": methods,
        "summary": {},
        "interpretation": (
            "learned_block_scorer predicts active blocks first, then selects "
            "coordinates inside the highest-probability blocks. It should be "
            "compared primarily against block_score_topk and CoSaMP."
        ),
    }

    print("\n" + "-" * 78)
    print("Block-sparse test results")
    print("-" * 78)
    print(f"{'method':<22} {'NRMSE':>18} {'IoU':>18}")

    for method in methods:
        summary["summary"][method] = {
            metric: summarize(store[method][metric])
            for metric in ["nrmse", "iou", "precision", "recall", "support_size"]
        }

        nrm = summary["summary"][method]["nrmse"]
        iou_s = summary["summary"][method]["iou"]

        print(
            f"{method:<22} "
            f"{nrm['mean']:>8.4f} ± {nrm['std']:<7.4f} "
            f"{iou_s['mean']:>8.4f} ± {iou_s['std']:<7.4f}"
        )

    out_json = RESULTS_DIR / f"{args.out_prefix}.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_json}")

    # Plot NRMSE
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    xs = np.arange(len(methods))
    means = [summary["summary"][m]["nrmse"]["mean"] for m in methods]
    stds = [summary["summary"][m]["nrmse"]["std"] for m in methods]

    ax.bar(xs, means, yerr=stds, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("NRMSE")
    ax.set_title(f"Learned block scorer: m={args.m}, k={args.k}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_png = FIGURES_DIR / f"{args.out_prefix}_nrmse.png"
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")

    # Print key gain numbers
    cosamp_nrmse = summary["summary"]["cosamp"]["nrmse"]["mean"]
    block_nrmse = summary["summary"]["block_score_topk"]["nrmse"]["mean"]
    learned_nrmse = summary["summary"]["learned_block_scorer"]["nrmse"]["mean"]

    print("\nKey gains:")
    print(f"  CoSaMP NRMSE              = {cosamp_nrmse:.4f}")
    print(f"  block_score_topk NRMSE    = {block_nrmse:.4f}")
    print(f"  learned_block_scorer NRMSE= {learned_nrmse:.4f}")
    print(f"  gain learned vs CoSaMP    = {cosamp_nrmse - learned_nrmse:+.4f}")
    print(f"  gain learned vs block     = {block_nrmse - learned_nrmse:+.4f}")


if __name__ == "__main__":
    main()
