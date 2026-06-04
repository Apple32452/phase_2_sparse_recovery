#!/usr/bin/env python3
"""
Noise robustness experiment for adaptive learned block refinement.

Goal:
  Test whether residual-stop adaptive learned block refinement remains strong
  under measurement noise.

Outputs:
  results/adaptive_learned_block_refinement/adaptive_noise_robustness.json
  figures/adaptive_learned_block_refinement/adaptive_noise_robustness_nrmse.png
  figures/adaptive_learned_block_refinement/adaptive_noise_robustness_gain_cosamp.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "adaptive_learned_block_refinement"
FIGURE_DIR = ROOT / "figures" / "adaptive_learned_block_refinement"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-12)


def make_blocks(n: int, block_size: int) -> List[np.ndarray]:
    """
    Use only full blocks. For n=256 and block_size=5, this creates 51 blocks
    covering coordinates 0,...,254. Coordinate 255 is unused.
    """
    return [
        np.arange(i, i + block_size)
        for i in range(0, n - block_size + 1, block_size)
    ]


def make_matrix(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, n))
    col_norms = np.linalg.norm(A, axis=0) + 1e-12
    A = A / col_norms
    return A


def sample_block_sparse_signal(
    n: int,
    blocks: List[np.ndarray],
    k: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    block_size = len(blocks[0])
    assert k % block_size == 0, "k must be a multiple of block_size"

    q = k // block_size
    active_blocks = rng.choice(len(blocks), size=q, replace=False).tolist()

    x = np.zeros(n)
    support_parts = []

    for b in active_blocks:
        idx = blocks[b]
        support_parts.append(idx)
        x[idx] = rng.normal(size=len(idx))

    support = np.concatenate(support_parts).astype(int)
    x = x / (np.linalg.norm(x) + 1e-12)

    return x, support, active_blocks


def add_measurement_noise(
    y_clean: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Relative measurement noise:
        ||noise||_2 approximately noise_std * ||y_clean||_2
    """
    if noise_std <= 0:
        return y_clean.copy()

    noise = rng.normal(size=y_clean.shape)
    noise = noise / (np.linalg.norm(noise) + 1e-12)
    noise = noise_std * np.linalg.norm(y_clean) * noise
    return y_clean + noise


def fit_on_support(A: np.ndarray, y: np.ndarray, support: np.ndarray, n: int) -> np.ndarray:
    support = np.asarray(sorted(set(support.tolist())), dtype=int)
    xhat = np.zeros(n)

    if len(support) == 0:
        return xhat

    coef, *_ = np.linalg.lstsq(A[:, support], y, rcond=None)
    xhat[support] = coef
    return xhat


def support_from_blocks(block_ids: List[int], blocks: List[np.ndarray]) -> np.ndarray:
    if len(block_ids) == 0:
        return np.array([], dtype=int)
    return np.concatenate([blocks[b] for b in block_ids]).astype(int)


def select_top_blocks(scores: np.ndarray, q: int) -> List[int]:
    return np.argsort(-scores)[:q].tolist()


def nrmse(xhat: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-12))


def support_iou(s_hat: np.ndarray, s_true: np.ndarray) -> float:
    a = set(s_hat.tolist())
    b = set(s_true.tolist())
    if len(a | b) == 0:
        return 1.0
    return float(len(a & b) / len(a | b))


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def naive_topk(A: np.ndarray, y: np.ndarray, k: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    corr = np.abs(A.T @ y)
    support = np.argsort(-corr)[:k]
    xhat = fit_on_support(A, y, support, n)
    return xhat, support


def cosamp(
    A: np.ndarray,
    y: np.ndarray,
    k: int,
    n: int,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    r = y.copy()
    support: np.ndarray = np.array([], dtype=int)
    xhat = np.zeros(n)

    last_residual = np.inf

    for _ in range(max_iter):
        proxy = np.abs(A.T @ r)
        omega = np.argsort(-proxy)[: min(2 * k, n)]

        merged = np.array(sorted(set(support.tolist()) | set(omega.tolist())), dtype=int)

        b = fit_on_support(A, y, merged, n)
        new_support = np.argsort(-np.abs(b))[:k]
        xhat = fit_on_support(A, y, new_support, n)

        r = y - A @ xhat
        residual = np.linalg.norm(r)

        support = np.array(sorted(new_support.tolist()), dtype=int)

        if abs(last_residual - residual) < tol:
            break
        last_residual = residual

    return xhat, support


def block_corr_scores(A: np.ndarray, y: np.ndarray, blocks: List[np.ndarray]) -> np.ndarray:
    c = A.T @ y
    scores = []
    for idx in blocks:
        scores.append(np.linalg.norm(c[idx]) / np.sqrt(len(idx)))
    return np.asarray(scores, dtype=float)


def block_score_topk(
    A: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    k: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    q = k // len(blocks[0])
    scores = block_corr_scores(A, y, blocks)
    block_ids = select_top_blocks(scores, q)
    support = support_from_blocks(block_ids, blocks)
    xhat = fit_on_support(A, y, support, n)
    return xhat, support, block_ids


# ---------------------------------------------------------------------
# Learned block scorer
# ---------------------------------------------------------------------

def block_features(
    A: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    residual: np.ndarray | None = None,
) -> np.ndarray:
    if residual is None:
        residual = y

    At_y = A.T @ y
    At_r = A.T @ residual

    feats = []

    for idx in blocks:
        AB = A[:, idx]
        cy = At_y[idx]
        cr = At_r[idx]

        gram = AB.T @ AB
        offdiag = gram - np.diag(np.diag(gram))

        feats.append([
            np.linalg.norm(cy),
            np.max(np.abs(cy)),
            np.mean(np.abs(cy)),
            np.linalg.norm(cr),
            np.max(np.abs(cr)),
            np.mean(np.abs(cr)),
            np.linalg.norm(AB, ord="fro"),
            np.mean(np.abs(offdiag)),
            np.max(np.abs(offdiag)),
        ])

    return np.asarray(feats, dtype=float)


def train_learned_block_scorer(
    A: np.ndarray,
    blocks: List[np.ndarray],
    n: int,
    k: int,
    n_train: int,
    noise_std: float,
    rng: np.random.Generator,
):
    X_all = []
    y_all = []

    for _ in range(n_train):
        x, _, active_blocks = sample_block_sparse_signal(n, blocks, k, rng)
        y_clean = A @ x
        y_obs = add_measurement_noise(y_clean, noise_std, rng)

        X = block_features(A, y_obs, blocks)
        labels = np.zeros(len(blocks), dtype=int)
        labels[active_blocks] = 1

        X_all.append(X)
        y_all.append(labels)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    if not HAS_SKLEARN:
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_scaled, y_all)

    return model, scaler


def predict_learned_scores(
    model,
    scaler,
    A: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    residual: np.ndarray | None = None,
) -> np.ndarray:
    X = block_features(A, y, blocks, residual=residual)

    if model is None or scaler is None:
        return block_corr_scores(A, y if residual is None else residual, blocks)

    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


def learned_block_topk(
    model,
    scaler,
    A: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    k: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    q = k // len(blocks[0])
    scores = predict_learned_scores(model, scaler, A, y, blocks)
    block_ids = select_top_blocks(scores, q)
    support = support_from_blocks(block_ids, blocks)
    xhat = fit_on_support(A, y, support, n)
    return xhat, support, block_ids


# ---------------------------------------------------------------------
# Adaptive learned block refinement
# ---------------------------------------------------------------------

def fit_on_block_ids(
    A: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    block_ids: List[int],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    support = support_from_blocks(block_ids, blocks)
    xhat = fit_on_support(A, y, support, n)
    rss = float(np.linalg.norm(y - A @ xhat) ** 2)
    return xhat, support, rss


def propose_best_one_swap(
    A: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    current_blocks: List[int],
    current_x: np.ndarray,
    model,
    scaler,
    n: int,
    max_drop: int = 4,
    max_add: int = 10,
) -> Tuple[List[int], np.ndarray, np.ndarray, float]:
    G = len(blocks)
    current_set = set(current_blocks)

    residual = y - A @ current_x

    residual_scores = block_corr_scores(A, residual, blocks)
    learned_scores = predict_learned_scores(model, scaler, A, y, blocks, residual=residual)

    proposal_scores = zscore(residual_scores) + 0.5 * zscore(learned_scores)

    coef_energy = np.zeros(G)
    for b in current_blocks:
        coef_energy[b] = np.linalg.norm(current_x[blocks[b]])

    drop_candidates = sorted(current_blocks, key=lambda b: coef_energy[b])[:max_drop]
    add_candidates = [
        b for b in np.argsort(-proposal_scores).tolist()
        if b not in current_set
    ][:max_add]

    best_blocks = current_blocks
    best_x = current_x
    best_support = support_from_blocks(current_blocks, blocks)
    best_rss = float(np.linalg.norm(y - A @ current_x) ** 2)

    for d in drop_candidates:
        for a in add_candidates:
            candidate = sorted((current_set - {d}) | {a})
            x_cand, s_cand, rss_cand = fit_on_block_ids(A, y, blocks, candidate, n)

            if rss_cand < best_rss:
                best_blocks = candidate
                best_x = x_cand
                best_support = s_cand
                best_rss = rss_cand

    return best_blocks, best_x, best_support, best_rss


def learned_refinement(
    A: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    init_blocks: List[int],
    model,
    scaler,
    k: int,
    n: int,
    max_refine_iters: int,
    min_rel_improvement: float = 1e-4,
    force_fixed_iters: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    current_blocks = sorted(init_blocks)
    current_x, current_support, current_rss = fit_on_block_ids(
        A, y, blocks, current_blocks, n
    )

    accepted_steps = []

    for t in range(max_refine_iters):
        candidate_blocks, candidate_x, candidate_support, candidate_rss = propose_best_one_swap(
            A=A,
            y=y,
            blocks=blocks,
            current_blocks=current_blocks,
            current_x=current_x,
            model=model,
            scaler=scaler,
            n=n,
        )

        improvement = current_rss - candidate_rss
        threshold = min_rel_improvement * max(current_rss, 1e-12)

        if force_fixed_iters or improvement > threshold:
            current_blocks = candidate_blocks
            current_x = candidate_x
            current_support = candidate_support
            current_rss = candidate_rss
            accepted_steps.append(t + 1)
        else:
            break

    return current_x, current_support, accepted_steps


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def evaluate_setting(
    m: int,
    k: int,
    noise_std: float,
    seed: int,
    n: int,
    block_size: int,
    n_train: int,
    n_test: int,
) -> Dict:
    rng = np.random.default_rng(seed)

    blocks = make_blocks(n, block_size)
    A = make_matrix(m, n, rng)

    model, scaler = train_learned_block_scorer(
        A=A,
        blocks=blocks,
        n=n,
        k=k,
        n_train=n_train,
        noise_std=noise_std,
        rng=rng,
    )

    method_values: Dict[str, Dict[str, List[float]]] = {}

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

    for method in methods:
        method_values[method] = {"nrmse": [], "iou": [], "accepted_steps": []}

    for _ in range(n_test):
        x, support_true, _ = sample_block_sparse_signal(n, blocks, k, rng)
        y_clean = A @ x
        y = add_measurement_noise(y_clean, noise_std, rng)

        # Naive
        xhat, shat = naive_topk(A, y, k, n)
        method_values["naive"]["nrmse"].append(nrmse(xhat, x))
        method_values["naive"]["iou"].append(support_iou(shat, support_true))

        # CoSaMP
        xhat, shat = cosamp(A, y, k, n)
        method_values["cosamp"]["nrmse"].append(nrmse(xhat, x))
        method_values["cosamp"]["iou"].append(support_iou(shat, support_true))

        # Block heuristic
        xhat, shat, block_ids_block = block_score_topk(A, y, blocks, k, n)
        method_values["block_score_topk"]["nrmse"].append(nrmse(xhat, x))
        method_values["block_score_topk"]["iou"].append(support_iou(shat, support_true))

        # Learned block scorer
        xhat, shat, block_ids_learned = learned_block_topk(model, scaler, A, y, blocks, k, n)
        method_values["learned_block_scorer"]["nrmse"].append(nrmse(xhat, x))
        method_values["learned_block_scorer"]["iou"].append(support_iou(shat, support_true))

        # Use learned support as initialization.
        init_blocks = block_ids_learned

        # One-step refinement
        xhat, shat, accepted = learned_refinement(
            A=A,
            y=y,
            blocks=blocks,
            init_blocks=init_blocks,
            model=model,
            scaler=scaler,
            k=k,
            n=n,
            max_refine_iters=1,
            min_rel_improvement=1e-4,
            force_fixed_iters=False,
        )
        method_values["one_step_refinement"]["nrmse"].append(nrmse(xhat, x))
        method_values["one_step_refinement"]["iou"].append(support_iou(shat, support_true))
        method_values["one_step_refinement"]["accepted_steps"].append(len(accepted))

        # Fixed iterative refinement
        xhat, shat, accepted = learned_refinement(
            A=A,
            y=y,
            blocks=blocks,
            init_blocks=init_blocks,
            model=model,
            scaler=scaler,
            k=k,
            n=n,
            max_refine_iters=4,
            min_rel_improvement=1e-4,
            force_fixed_iters=True,
        )
        method_values["fixed_iterative_refinement"]["nrmse"].append(nrmse(xhat, x))
        method_values["fixed_iterative_refinement"]["iou"].append(support_iou(shat, support_true))
        method_values["fixed_iterative_refinement"]["accepted_steps"].append(len(accepted))

        # Adaptive refinement
        xhat, shat, accepted = learned_refinement(
            A=A,
            y=y,
            blocks=blocks,
            init_blocks=init_blocks,
            model=model,
            scaler=scaler,
            k=k,
            n=n,
            max_refine_iters=4,
            min_rel_improvement=1e-4,
            force_fixed_iters=False,
        )
        method_values["adaptive_refinement"]["nrmse"].append(nrmse(xhat, x))
        method_values["adaptive_refinement"]["iou"].append(support_iou(shat, support_true))
        method_values["adaptive_refinement"]["accepted_steps"].append(len(accepted))

        # Oracle
        xhat = fit_on_support(A, y, support_true, n)
        method_values["oracle"]["nrmse"].append(nrmse(xhat, x))
        method_values["oracle"]["iou"].append(support_iou(support_true, support_true))

    summary = {
        "m": m,
        "k": k,
        "noise_std": noise_std,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "methods": {},
    }

    for method in methods:
        vals = method_values[method]
        summary["methods"][method] = {
            "nrmse_mean": float(np.mean(vals["nrmse"])),
            "nrmse_std": float(np.std(vals["nrmse"])),
            "iou_mean": float(np.mean(vals["iou"])),
            "iou_std": float(np.std(vals["iou"])),
        }

        if len(vals["accepted_steps"]) > 0:
            summary["methods"][method]["accepted_steps_mean"] = float(np.mean(vals["accepted_steps"]))
            summary["methods"][method]["accepted_steps_std"] = float(np.std(vals["accepted_steps"]))

    return summary


def aggregate_runs(runs: List[Dict]) -> List[Dict]:
    rows = []

    keys = sorted(set((r["m"], r["k"], r["noise_std"]) for r in runs))

    for m, k, noise_std in keys:
        subset = [r for r in runs if r["m"] == m and r["k"] == k and r["noise_std"] == noise_std]
        methods = subset[0]["methods"].keys()

        for method in methods:
            nrmse_vals = np.array([r["methods"][method]["nrmse_mean"] for r in subset], dtype=float)
            iou_vals = np.array([r["methods"][method]["iou_mean"] for r in subset], dtype=float)

            row = {
                "m": int(m),
                "k": int(k),
                "noise_std": float(noise_std),
                "method": method,
                "num_seeds": len(subset),
                "nrmse_mean": float(nrmse_vals.mean()),
                "nrmse_se": float(nrmse_vals.std(ddof=1) / np.sqrt(len(nrmse_vals))) if len(nrmse_vals) > 1 else 0.0,
                "iou_mean": float(iou_vals.mean()),
                "iou_se": float(iou_vals.std(ddof=1) / np.sqrt(len(iou_vals))) if len(iou_vals) > 1 else 0.0,
            }

            if "accepted_steps_mean" in subset[0]["methods"][method]:
                step_vals = np.array([
                    r["methods"][method]["accepted_steps_mean"]
                    for r in subset
                ], dtype=float)
                row["accepted_steps_mean"] = float(step_vals.mean())
                row["accepted_steps_se"] = float(step_vals.std(ddof=1) / np.sqrt(len(step_vals))) if len(step_vals) > 1 else 0.0

            rows.append(row)

    return rows


def get_row(rows: List[Dict], m: int, k: int, noise_std: float, method: str) -> Dict:
    for r in rows:
        if r["m"] == m and r["k"] == k and abs(r["noise_std"] - noise_std) < 1e-12 and r["method"] == method:
            return r
    raise KeyError((m, k, noise_std, method))


def make_plots(rows: List[Dict], settings: List[Tuple[int, int]], noise_levels: List[float]) -> None:
    methods_to_plot = [
        "cosamp",
        "block_score_topk",
        "learned_block_scorer",
        "one_step_refinement",
        "adaptive_refinement",
    ]

    # NRMSE plot
    plt.figure(figsize=(12, 7))

    for m, k in settings:
        for method in methods_to_plot:
            vals = [
                get_row(rows, m, k, noise, method)["nrmse_mean"]
                for noise in noise_levels
            ]
            label = f"{method}, m={m}, k={k}"
            plt.plot(noise_levels, vals, marker="o", label=label)

    plt.xlabel("relative noise level")
    plt.ylabel("NRMSE")
    plt.title("Noise robustness: NRMSE")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out = FIGURE_DIR / "adaptive_noise_robustness_nrmse.png"
    plt.savefig(out, dpi=200)
    plt.close()

    # Gain over CoSaMP
    plt.figure(figsize=(12, 7))

    for m, k in settings:
        gains = []
        for noise in noise_levels:
            cosamp_val = get_row(rows, m, k, noise, "cosamp")["nrmse_mean"]
            adaptive_val = get_row(rows, m, k, noise, "adaptive_refinement")["nrmse_mean"]
            gains.append(cosamp_val - adaptive_val)

        plt.plot(noise_levels, gains, marker="o", label=f"m={m}, k={k}")

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("relative noise level")
    plt.ylabel("NRMSE gain over CoSaMP")
    plt.title("Noise robustness: adaptive gain over CoSaMP")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out = FIGURE_DIR / "adaptive_noise_robustness_gain_cosamp.png"
    plt.savefig(out, dpi=200)
    plt.close()


def print_table(rows: List[Dict], settings: List[Tuple[int, int]], noise_levels: List[float]) -> None:
    print("\nNoise robustness aggregate")
    print("-" * 110)
    print(f"{'setting':<12} {'noise':>8} {'method':<28} {'NRMSE':>10} {'SE':>10} {'IoU':>10}")

    for m, k in settings:
        for noise in noise_levels:
            for method in [
                "cosamp",
                "block_score_topk",
                "learned_block_scorer",
                "one_step_refinement",
                "adaptive_refinement",
                "oracle",
            ]:
                r = get_row(rows, m, k, noise, method)
                print(
                    f"m={m},k={k:<4} "
                    f"{noise:>8.3f} "
                    f"{method:<28} "
                    f"{r['nrmse_mean']:>10.4f} "
                    f"{r['nrmse_se']:>10.4f} "
                    f"{r['iou_mean']:>10.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test first.")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=5)
    args = parser.parse_args()

    if args.quick:
        seeds = [0, 1]
        noise_levels = [0.0, 0.05]
        n_train = 200
        n_test = 80
    else:
        seeds = list(range(10))
        noise_levels = [0.0, 0.01, 0.05, 0.10]
        n_train = 1000
        n_test = 300

    settings = [
        (96, 40),
        (96, 55),
    ]

    print("=" * 80)
    print("Adaptive learned block refinement: noise robustness")
    print("=" * 80)
    print(f"settings={settings}")
    print(f"noise_levels={noise_levels}")
    print(f"seeds={seeds}")
    print(f"n_train={n_train}, n_test={n_test}")
    print(f"sklearn_available={HAS_SKLEARN}")
    print("=" * 80)

    runs = []

    for m, k in settings:
        for noise_std in noise_levels:
            for seed in seeds:
                print(f"\nRunning m={m}, k={k}, noise={noise_std}, seed={seed}", flush=True)

                run = evaluate_setting(
                    m=m,
                    k=k,
                    noise_std=noise_std,
                    seed=seed,
                    n=args.n,
                    block_size=args.block_size,
                    n_train=n_train,
                    n_test=n_test,
                )
                runs.append(run)

    rows = aggregate_runs(runs)

    payload = {
        "description": "Noise robustness for adaptive learned block refinement.",
        "settings": [{"m": m, "k": k} for m, k in settings],
        "noise_levels": noise_levels,
        "seeds": seeds,
        "n_train": n_train,
        "n_test": n_test,
        "runs": runs,
        "aggregate": rows,
    }

    out_json = RESULT_DIR / "adaptive_noise_robustness.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    make_plots(rows, settings, noise_levels)
    print_table(rows, settings, noise_levels)

    print("\nWrote:")
    print(f"  {out_json}")
    print(f"  {FIGURE_DIR / 'adaptive_noise_robustness_nrmse.png'}")
    print(f"  {FIGURE_DIR / 'adaptive_noise_robustness_gain_cosamp.png'}")


if __name__ == "__main__":
    main()