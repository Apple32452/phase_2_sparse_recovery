"""
unknown_k.py

Unknown-sparsity experiment.

Motivation:
    Classical CoSaMP requires the true sparsity k as input. In realistic
    sparse-recovery problems, k is rarely known exactly. This experiment tests
    whether a simple learned cardinality predictor can improve recovery when k
    is unknown.

Experiment:
    - Fixed Gaussian sensing operator A.
    - True sparsity k is sampled from a range.
    - Train a RandomForest classifier to predict k from sorted correlation
      features |A^T y|.
    - Compare:
        1. CoSaMP with true k            (oracle-k baseline)
        2. CoSaMP with fixed k=25
        3. CoSaMP with fixed k=40
        4. CoSaMP with fixed k=55
        5. OMP with residual stopping   (unknown-k classical baseline)
        6. Learned-k predictor + CoSaMP

Outputs:
    results/unknown_k/unknown_k_results.json
    figures/unknown_k/unknown_k_nrmse.png
    figures/unknown_k/unknown_k_prediction.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "unknown_k"
FIGURES_DIR = ROOT / "figures" / "unknown_k"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--k-values", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70])
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument("--feature-dim", type=int, default=40)
    p.add_argument("--omp-stop-tol", type=float, default=1e-4)
    p.add_argument("--out-prefix", type=str, default="unknown_k")
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


def add_noise(y: np.ndarray, noise_std: float, rng: np.random.Generator):
    if noise_std <= 0:
        return y
    return y + noise_std * rng.standard_normal(y.shape)


# ---------------------------------------------------------------------
# Utilities and metrics
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


# ---------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------

def cosamp(A: np.ndarray, y: np.ndarray, k: int, max_iters: int = 30, tol: float = 1e-10):
    n = A.shape[1]
    k = int(max(1, min(k, n // 2)))

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
        S_prev = best_k_support(A.T @ y, k)

    return S_prev, support_lstsq(A, y, S_prev)


def omp_residual_stop(A: np.ndarray, y: np.ndarray, max_steps: int, stop_tol: float):
    residual = y.copy()
    y_norm = max(float(np.linalg.norm(y)), 1e-12)
    selected = []

    max_steps = int(min(max_steps, A.shape[1]))

    for _ in range(max_steps):
        scores = np.abs(A.T @ residual)

        if selected:
            scores[selected] = -np.inf

        j = int(np.argmax(scores))
        selected.append(j)

        x_s, *_ = np.linalg.lstsq(A[:, selected], y, rcond=None)
        residual = y - A[:, selected] @ x_s

        if float(np.linalg.norm(residual)) <= stop_tol * y_norm:
            break

    S = set(int(i) for i in selected)
    x_hat = support_lstsq(A, y, S)
    return S, x_hat


# ---------------------------------------------------------------------
# Learned k predictor
# ---------------------------------------------------------------------

def correlation_features(A: np.ndarray, y: np.ndarray, feature_dim: int):
    scores = np.sort(np.abs(A.T @ y))[::-1]
    top = scores[:feature_dim]

    if top.size < feature_dim:
        top = np.pad(top, (0, feature_dim - top.size))

    scale = max(top[0], 1e-12)
    top_norm = top / scale

    # Gaps between sorted correlations reveal an elbow around k.
    gaps = top_norm[:-1] - top_norm[1:]
    gap_feats = gaps[: min(20, gaps.size)]

    if gap_feats.size < 20:
        gap_feats = np.pad(gap_feats, (0, 20 - gap_feats.size))

    energy = np.cumsum(top_norm ** 2)
    energy = energy / max(energy[-1], 1e-12)
    energy_feats = energy[: min(20, energy.size)]

    if energy_feats.size < 20:
        energy_feats = np.pad(energy_feats, (0, 20 - energy_feats.size))

    return np.concatenate([
        top_norm,
        gap_feats,
        energy_feats,
        np.array([np.linalg.norm(y)], dtype=np.float64),
    ])


def train_k_predictor(A, args, rng):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for unknown_k.py. "
            "Install with: pip install scikit-learn"
        ) from e

    X_feat = []
    y_label = []

    for _ in range(args.n_train):
        k = int(rng.choice(args.k_values))
        x, _ = make_sparse_signal(args.n, k, rng)
        y = add_noise(A @ x, args.noise_std, rng)

        X_feat.append(correlation_features(A, y, args.feature_dim))
        y_label.append(k)

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(np.asarray(X_feat), np.asarray(y_label))

    return clf


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def eval_one_method(method_name, S_pred, x_hat, x_true, S_true, k_true):
    p, r = precision_recall(S_pred, S_true)
    return {
        "method": method_name,
        "nrmse": nrmse(x_hat, x_true),
        "iou": iou(S_pred, S_true),
        "precision": p,
        "recall": r,
        "selected_size": len(S_pred),
        "k_error": abs(len(S_pred) - k_true),
    }


def main():
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    A = make_gaussian_operator(args.m, args.n, seed=args.seed)

    print("=" * 78)
    print("Unknown-k sparse recovery experiment")
    print("=" * 78)
    print(f"n={args.n}, m={args.m}, k_values={args.k_values}")
    print(f"n_train={args.n_train}, n_test={args.n_test}, noise_std={args.noise_std}")

    print("\nTraining learned k predictor...")
    k_predictor = train_k_predictor(A, args, rng)

    methods = [
        "cosamp_true_k",
        "cosamp_fixed_25",
        "cosamp_fixed_40",
        "cosamp_fixed_55",
        "omp_residual_stop",
        "learned_k_cosamp",
    ]

    records = []
    pred_k_records = []

    for t in range(args.n_test):
        k_true = int(rng.choice(args.k_values))
        x_true, S_true = make_sparse_signal(args.n, k_true, rng)
        y = add_noise(A @ x_true, args.noise_std, rng)

        feat = correlation_features(A, y, args.feature_dim).reshape(1, -1)
        k_pred = int(k_predictor.predict(feat)[0])

        pred_k_records.append({"k_true": k_true, "k_pred": k_pred})

        # 1. CoSaMP with true k
        S, xh = cosamp(A, y, k_true, max_iters=args.max_iters)
        records.append({**eval_one_method("cosamp_true_k", S, xh, x_true, S_true, k_true), "k_true": k_true, "k_pred": k_true})

        # 2. Fixed k baselines
        for fixed_k in [25, 40, 55]:
            S, xh = cosamp(A, y, fixed_k, max_iters=args.max_iters)
            records.append({
                **eval_one_method(f"cosamp_fixed_{fixed_k}", S, xh, x_true, S_true, k_true),
                "k_true": k_true,
                "k_pred": fixed_k,
            })

        # 3. OMP residual stopping
        S, xh = omp_residual_stop(A, y, max_steps=args.m, stop_tol=args.omp_stop_tol)
        records.append({
            **eval_one_method("omp_residual_stop", S, xh, x_true, S_true, k_true),
            "k_true": k_true,
            "k_pred": len(S),
        })

        # 4. Learned-k + CoSaMP
        S, xh = cosamp(A, y, k_pred, max_iters=args.max_iters)
        records.append({
            **eval_one_method("learned_k_cosamp", S, xh, x_true, S_true, k_true),
            "k_true": k_true,
            "k_pred": k_pred,
        })

    # Summaries by method and true k
    out = {
        "config": vars(args),
        "records": records,
        "pred_k_records": pred_k_records,
        "summary_by_method": {},
        "summary_by_k": {},
        "interpretation": (
            "cosamp_true_k is an oracle-k baseline. If learned_k_cosamp "
            "outperforms fixed-k baselines, then learning cardinality helps "
            "when k is unknown."
        ),
    }

    for method in methods:
        rows = [r for r in records if r["method"] == method]
        out["summary_by_method"][method] = {
            "nrmse": summarize([r["nrmse"] for r in rows]),
            "iou": summarize([r["iou"] for r in rows]),
            "precision": summarize([r["precision"] for r in rows]),
            "recall": summarize([r["recall"] for r in rows]),
            "selected_size": summarize([r["selected_size"] for r in rows]),
            "k_error": summarize([r["k_error"] for r in rows]),
        }

    for k in args.k_values:
        out["summary_by_k"][str(k)] = {}
        for method in methods:
            rows = [r for r in records if r["method"] == method and r["k_true"] == k]
            out["summary_by_k"][str(k)][method] = {
                "nrmse": summarize([r["nrmse"] for r in rows]),
                "iou": summarize([r["iou"] for r in rows]),
                "k_error": summarize([r["k_error"] for r in rows]),
            }

    pred_errors = [abs(r["k_pred"] - r["k_true"]) for r in pred_k_records]
    out["k_predictor"] = {
        "mae": float(np.mean(pred_errors)),
        "accuracy": float(np.mean([r["k_pred"] == r["k_true"] for r in pred_k_records])),
    }

    print("\nSummary by method:")
    for method in methods:
        s = out["summary_by_method"][method]
        print(
            f"{method:<22} "
            f"NRMSE={s['nrmse']['mean']:.4f}±{s['nrmse']['std']:.4f}  "
            f"IoU={s['iou']['mean']:.4f}±{s['iou']['std']:.4f}  "
            f"k_error={s['k_error']['mean']:.2f}"
        )

    print("\nLearned k predictor:")
    print(f"  MAE={out['k_predictor']['mae']:.3f}")
    print(f"  accuracy={out['k_predictor']['accuracy']:.3f}")

    out_json = RESULTS_DIR / f"{args.out_prefix}_results.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_json}")

    # Plot NRMSE by true k
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    for method in methods:
        xs = []
        ys = []
        yerr = []
        for k in args.k_values:
            item = out["summary_by_k"][str(k)][method]["nrmse"]
            xs.append(k)
            ys.append(item["mean"])
            yerr.append(item["std"])

        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=method)

    ax.set_xlabel("true sparsity k")
    ax.set_ylabel("NRMSE")
    ax.set_title("Unknown-k recovery: NRMSE by true sparsity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()

    out_png = FIGURES_DIR / f"{args.out_prefix}_nrmse.png"
    fig.savefig(out_png, dpi=180)
    print(f"Wrote {out_png}")

    # Plot true k vs predicted k
    true_ks = np.array([r["k_true"] for r in pred_k_records])
    pred_ks = np.array([r["k_pred"] for r in pred_k_records])

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(true_ks, pred_ks, alpha=0.55)
    lo = min(args.k_values)
    hi = max(args.k_values)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("true k")
    ax.set_ylabel("predicted k")
    ax.set_title("Learned cardinality prediction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_pred_png = FIGURES_DIR / f"{args.out_prefix}_k_prediction.png"
    fig.savefig(out_pred_png, dpi=180)
    print(f"Wrote {out_pred_png}")


if __name__ == "__main__":
    main()
