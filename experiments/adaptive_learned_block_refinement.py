"""
adaptive_learned_block_refinement.py

Adaptive learned block-refinement experiment.

Motivation:
    Previous experiments showed that one-step learned block refinement is very
    strong, while multiple fixed refinement steps can drift and degrade.

Goal:
    Turn the empirical "one step is best" observation into an adaptive
    early-stopping rule.

Main idea:
    1. Train a learned block scorer.
    2. Initialize support from learned block probabilities.
    3. Propose residual-based block refinements.
    4. Accept a refinement only if it improves an unsupervised score:
           residual_norm + change_weight * support_change_penalty
    5. Stop automatically when no improvement is detected.

Methods:
    - naive top-k
    - CoSaMP
    - block_score_topk
    - learned_block_scorer
    - one_step_refinement
    - fixed_iterative_refinement
    - adaptive_refinement
    - oracle

Outputs:
    results/adaptive_learned_block_refinement/<prefix>.json
    figures/adaptive_learned_block_refinement/<prefix>_nrmse.png
    figures/adaptive_learned_block_refinement/<prefix>_accepted_steps.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "adaptive_learned_block_refinement"
FIGURES_DIR = ROOT / "figures" / "adaptive_learned_block_refinement"
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
    p.add_argument("--refine-iters", type=int, default=4)
    p.add_argument("--change-weight", type=float, default=0.05)
    p.add_argument("--min-improvement", type=float, default=1e-4)
    p.add_argument(
        "--out-prefix",
        type=str,
        default="adaptive_learned_block_refinement_m96_k40",
    )
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


def fill_amplitudes(n: int, support: set[int], rng: np.random.Generator):
    x = np.zeros(n, dtype=np.float64)
    support_list = sorted(support)
    amps = rng.uniform(0.5, 2.0, size=len(support_list))
    signs = rng.choice([-1.0, 1.0], size=len(support_list))
    x[support_list] = amps * signs
    return x


def block_sparse_signal(n: int, k: int, block_size: int, rng: np.random.Generator):
    """
    Generate an exactly k-sparse block-structured signal.

    Uses only full blocks to avoid the final partial-block issue when
    n is not divisible by block_size.
    """
    n_full_blocks = n // block_size
    blocks_needed = int(np.ceil(k / block_size))

    if blocks_needed > n_full_blocks:
        raise ValueError(
            f"Cannot choose {blocks_needed} full blocks from {n_full_blocks} blocks."
        )

    chosen_blocks = set(
        int(b) for b in rng.choice(n_full_blocks, size=blocks_needed, replace=False)
    )

    support = []
    for b in chosen_blocks:
        lo = b * block_size
        hi = lo + block_size
        support.extend(range(lo, hi))

    support = set(int(i) for i in sorted(support)[:k])
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
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
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


def support_change_penalty(S_new: set[int], S_old: set[int]) -> float:
    union = len(S_new | S_old)
    inter = len(S_new & S_old)
    support_iou = inter / max(union, 1)
    return float(1.0 - support_iou)


def unsupervised_support_score(
    A,
    y,
    support,
    prev_support=None,
    change_weight=0.05,
):
    """
    Test-time stopping score.

    Lower is better.

    Uses only:
        - residual norm
        - optional support-change penalty

    Does not use the true signal or true support.
    """
    support = set(int(i) for i in support)
    x_hat = support_lstsq(A, y, support)
    residual = y - A @ x_hat
    residual_score = np.linalg.norm(residual) / max(np.linalg.norm(y), 1e-12)

    if prev_support is None:
        penalty = 0.0
    else:
        penalty = support_change_penalty(support, set(prev_support))

    return float(residual_score + change_weight * penalty)


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
    rank blocks by total correlation energy, then select coordinates inside
    the strongest blocks.
    """
    raw = np.abs(A.T @ y)
    n = raw.size
    n_blocks = n // block_size

    block_scores = []
    for b in range(n_blocks):
        lo = b * block_size
        hi = lo + block_size
        block_scores.append(float(np.sum(raw[lo:hi] ** 2)))

    block_order = np.argsort(block_scores)[::-1]

    selected = []
    for b in block_order:
        lo = int(b * block_size)
        hi = lo + block_size
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
    One feature vector per full block.
    """
    raw = np.abs(A.T @ y)
    n = raw.size
    n_blocks = n // block_size

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
        hi = lo + block_size

        vals = raw_norm[lo:hi]
        subG = G[lo:hi, lo:hi]

        block_energy.append(float(np.sum(vals ** 2)))
        block_mean.append(float(np.mean(vals)))
        block_max.append(float(np.max(vals)))
        block_std.append(float(np.std(vals)))
        block_sum.append(float(np.sum(vals)))
        block_coh_mean.append(float(np.mean(subG)))
        block_coh_max.append(float(np.max(subG)))

    block_energy = np.asarray(block_energy)
    block_sum = np.asarray(block_sum)

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
        raise ImportError("Install scikit-learn with: pip install scikit-learn") from e

    X_all = []
    y_all = []
    n_blocks = args.n // args.block_size

    for _ in range(args.n_train):
        x, _, active_blocks = block_sparse_signal(args.n, args.k, args.block_size, rng)
        y = add_noise(A @ x, args.noise_std, rng)

        X = block_features(A, y, args.block_size)
        labels = np.zeros(n_blocks, dtype=np.int64)

        for b in active_blocks:
            if 0 <= b < n_blocks:
                labels[b] = 1

        X_all.append(X)
        y_all.append(labels)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )

    clf.fit(np.vstack(X_all), np.concatenate(y_all))
    return clf


def learned_block_probs(clf, A, y, block_size=5):
    X = block_features(A, y, block_size)
    return clf.predict_proba(X)[:, 1]


def support_from_blocks(A, y, blocks, k, block_size=5, score_vector=None):
    raw = np.abs(A.T @ y) if score_vector is None else np.abs(score_vector)
    n = raw.size

    selected = []
    for b in blocks:
        lo = int(b * block_size)
        hi = min(n, lo + block_size)
        coords = list(range(lo, hi))
        coords = sorted(coords, key=lambda j: raw[j], reverse=True)
        selected.extend(coords)

    selected = sorted(set(selected), key=lambda j: raw[j], reverse=True)
    return set(int(i) for i in selected[:k])


def learned_block_support(clf, A, y, k, block_size=5):
    probs = learned_block_probs(clf, A, y, block_size)
    blocks_needed = int(np.ceil(k / block_size))
    block_order = np.argsort(probs)[::-1]
    return support_from_blocks(A, y, block_order[:blocks_needed], k, block_size)


# ---------------------------------------------------------------------
# Fixed and adaptive refinement
# ---------------------------------------------------------------------

def propose_refined_support(
    clf,
    A,
    y,
    current_support,
    k,
    block_size=5,
    learned_probs=None,
):
    """
    One CoSaMP-style learned block refinement proposal.

    Uses:
        learned block probabilities,
        residual block energy,
        candidate least-squares amplitude energy.
    """
    n = A.shape[1]
    n_blocks = n // block_size
    blocks_needed = int(np.ceil(k / block_size))

    if learned_probs is None:
        learned_probs = learned_block_probs(clf, A, y, block_size)

    x_current = support_lstsq(A, y, current_support)
    residual = y - A @ x_current
    residual_corr = np.abs(A.T @ residual)

    residual_block_energy = np.zeros(n_blocks)

    for b in range(n_blocks):
        lo = b * block_size
        hi = lo + block_size
        residual_block_energy[b] = np.sum(residual_corr[lo:hi] ** 2)

    residual_block_energy /= max(np.max(residual_block_energy), 1e-12)

    current_blocks = set(int(j // block_size) for j in current_support)
    learned_candidates = set(np.argsort(learned_probs)[::-1][:blocks_needed])
    residual_candidates = set(np.argsort(residual_block_energy)[::-1][:blocks_needed])

    candidate_blocks = current_blocks | learned_candidates | residual_candidates

    candidate_support_full = support_from_blocks(
        A,
        y,
        candidate_blocks,
        min(len(candidate_blocks) * block_size, n),
        block_size,
    )

    x_candidate_full = support_lstsq(A, y, candidate_support_full)

    candidate_amp_energy = np.zeros(n_blocks)
    for b in range(n_blocks):
        lo = b * block_size
        hi = lo + block_size
        candidate_amp_energy[b] = np.sum(np.abs(x_candidate_full[lo:hi]) ** 2)

    candidate_amp_energy /= max(np.max(candidate_amp_energy), 1e-12)

    combined = (
        0.45 * learned_probs
        + 0.35 * candidate_amp_energy
        + 0.20 * residual_block_energy
    )

    next_blocks = set(np.argsort(combined)[::-1][:blocks_needed])

    next_support = support_from_blocks(
        A,
        y,
        next_blocks,
        k,
        block_size,
        score_vector=x_candidate_full,
    )

    return next_support


def fixed_iterative_learned_block_refinement(
    clf,
    A,
    y,
    k,
    block_size=5,
    refine_iters=4,
):
    """
    Fixed-number learned block refinement.
    """
    current_support = learned_block_support(clf, A, y, k, block_size)
    learned_probs = learned_block_probs(clf, A, y, block_size)

    for _ in range(refine_iters):
        next_support = propose_refined_support(
            clf,
            A,
            y,
            current_support,
            k,
            block_size,
            learned_probs=learned_probs,
        )

        if next_support == current_support:
            break

        current_support = next_support

    return current_support


def adaptive_learned_block_refinement(
    clf,
    A,
    y,
    k,
    block_size=5,
    max_refine_iters=4,
    change_weight=0.05,
    min_improvement=1e-4,
):
    """
    Adaptive learned block refinement.

    A candidate update is accepted only if it improves an unsupervised
    residual-plus-stability score.
    """
    current_support = learned_block_support(clf, A, y, k, block_size)
    learned_probs = learned_block_probs(clf, A, y, block_size)

    current_score = unsupervised_support_score(
        A,
        y,
        current_support,
        prev_support=None,
        change_weight=change_weight,
    )

    best_support = set(current_support)
    best_score = current_score
    accepted_steps = 0
    score_history = [current_score]

    for _ in range(max_refine_iters):
        next_support = propose_refined_support(
            clf,
            A,
            y,
            current_support,
            k,
            block_size,
            learned_probs=learned_probs,
        )

        next_score = unsupervised_support_score(
            A,
            y,
            next_support,
            prev_support=current_support,
            change_weight=change_weight,
        )

        score_history.append(next_score)
        improvement = current_score - next_score

        if improvement <= min_improvement:
            break

        current_support = next_support
        current_score = next_score
        accepted_steps += 1

        if current_score < best_score:
            best_score = current_score
            best_support = set(current_support)

    return best_support, accepted_steps, score_history


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    A = make_gaussian_operator(args.m, args.n, args.seed)

    print("=" * 78)
    print("Adaptive learned block refinement")
    print("=" * 78)
    print(f"n={args.n}, m={args.m}, k={args.k}, block_size={args.block_size}")
    print(f"n_train={args.n_train}, n_test={args.n_test}")
    print(f"max_refine_iters={args.refine_iters}")
    print(f"change_weight={args.change_weight}, min_improvement={args.min_improvement}")

    print("\nTraining learned block scorer...")
    clf = train_block_scorer(A, args, rng)

    methods = [
        "naive",
        "cosamp",
        "block_score_topk",
        "learned_block_scorer",
        "one_step_refinement",
        "fixed_iterative_refinement",
        "adaptive_refinement",
        "oracle",
    ]

    store = {
        method: {"nrmse": [], "iou": [], "precision": [], "recall": [], "support_size": []}
        for method in methods
    }

    adaptive_steps = []
    adaptive_score_histories = []

    print("\nEvaluating...")

    for _ in range(args.n_test):
        x_true, S_true, _ = block_sparse_signal(args.n, args.k, args.block_size, rng)
        y = add_noise(A @ x_true, args.noise_std, rng)

        learned_support = learned_block_support(clf, A, y, args.k, args.block_size)

        one_step_support = fixed_iterative_learned_block_refinement(
            clf,
            A,
            y,
            args.k,
            args.block_size,
            refine_iters=1,
        )

        fixed_support = fixed_iterative_learned_block_refinement(
            clf,
            A,
            y,
            args.k,
            args.block_size,
            refine_iters=args.refine_iters,
        )

        adaptive_support, n_steps, score_history = adaptive_learned_block_refinement(
            clf,
            A,
            y,
            args.k,
            args.block_size,
            max_refine_iters=args.refine_iters,
            change_weight=args.change_weight,
            min_improvement=args.min_improvement,
        )

        adaptive_steps.append(n_steps)
        adaptive_score_histories.append(score_history)

        predicted = {
            "naive": naive_topk(A, y, args.k),
            "cosamp": cosamp(A, y, args.k, max_iters=args.max_iters),
            "block_score_topk": block_score_topk(A, y, args.k, args.block_size),
            "learned_block_scorer": learned_support,
            "one_step_refinement": one_step_support,
            "fixed_iterative_refinement": fixed_support,
            "adaptive_refinement": adaptive_support,
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
        "adaptive_steps": {
            "mean": float(np.mean(adaptive_steps)),
            "std": float(np.std(adaptive_steps)),
            "median": float(np.median(adaptive_steps)),
            "values": adaptive_steps,
        },
        "adaptive_score_histories": adaptive_score_histories,
        "interpretation": (
            "adaptive_refinement accepts residual-based support updates only if "
            "they improve an unsupervised residual-plus-stability score. "
            "one_step_refinement is included because previous ablations showed "
            "that one step is often strongest."
        ),
    }

    print("\n" + "-" * 78)
    print("Results")
    print("-" * 78)
    print(f"{'method':<28} {'NRMSE':>18} {'IoU':>18} {'support':>10}")

    for method in methods:
        summary["summary"][method] = {
            metric: summarize(store[method][metric])
            for metric in ["nrmse", "iou", "precision", "recall", "support_size"]
        }

        nrm = summary["summary"][method]["nrmse"]
        iou_s = summary["summary"][method]["iou"]
        supp = summary["summary"][method]["support_size"]

        print(
            f"{method:<28} "
            f"{nrm['mean']:>8.4f} ± {nrm['std']:<7.4f} "
            f"{iou_s['mean']:>8.4f} ± {iou_s['std']:<7.4f} "
            f"{supp['mean']:>8.2f}"
        )

    print("\nAdaptive accepted steps:")
    print(
        f"  mean={summary['adaptive_steps']['mean']:.3f}, "
        f"std={summary['adaptive_steps']['std']:.3f}, "
        f"median={summary['adaptive_steps']['median']:.3f}"
    )

    out_json = RESULTS_DIR / f"{args.out_prefix}.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_json}")

    # Plot NRMSE
    fig, ax = plt.subplots(figsize=(10.0, 4.8))

    xs = np.arange(len(methods))
    means = [summary["summary"][m]["nrmse"]["mean"] for m in methods]
    stds = [summary["summary"][m]["nrmse"]["std"] for m in methods]

    ax.bar(xs, means, yerr=stds, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("NRMSE")
    ax.set_title(f"Adaptive learned block refinement: m={args.m}, k={args.k}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_png = FIGURES_DIR / f"{args.out_prefix}_nrmse.png"
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")

    # Plot accepted steps
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = np.arange(-0.5, args.refine_iters + 1.5, 1)
    ax.hist(adaptive_steps, bins=bins, rwidth=0.85)
    ax.set_xlabel("accepted adaptive refinement steps")
    ax.set_ylabel("count")
    ax.set_title("Adaptive refinement stopping behavior")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_steps_png = FIGURES_DIR / f"{args.out_prefix}_accepted_steps.png"
    fig.savefig(out_steps_png, dpi=180)
    print(f"Wrote {out_steps_png}")

    # Key gains
    cosamp_n = summary["summary"]["cosamp"]["nrmse"]["mean"]
    block_n = summary["summary"]["block_score_topk"]["nrmse"]["mean"]
    learned_n = summary["summary"]["learned_block_scorer"]["nrmse"]["mean"]
    one_step_n = summary["summary"]["one_step_refinement"]["nrmse"]["mean"]
    fixed_n = summary["summary"]["fixed_iterative_refinement"]["nrmse"]["mean"]
    adaptive_n = summary["summary"]["adaptive_refinement"]["nrmse"]["mean"]

    print("\nKey gains:")
    print(f"  CoSaMP NRMSE                    = {cosamp_n:.4f}")
    print(f"  block_score_topk NRMSE          = {block_n:.4f}")
    print(f"  learned_block_scorer NRMSE      = {learned_n:.4f}")
    print(f"  one_step_refinement NRMSE       = {one_step_n:.4f}")
    print(f"  fixed_iterative_refinement NRMSE= {fixed_n:.4f}")
    print(f"  adaptive_refinement NRMSE       = {adaptive_n:.4f}")
    print(f"  gain adaptive vs CoSaMP         = {cosamp_n - adaptive_n:+.4f}")
    print(f"  gain adaptive vs block_score    = {block_n - adaptive_n:+.4f}")
    print(f"  gain adaptive vs learned        = {learned_n - adaptive_n:+.4f}")
    print(f"  gain adaptive vs one_step       = {one_step_n - adaptive_n:+.4f}")


if __name__ == "__main__":
    main()
