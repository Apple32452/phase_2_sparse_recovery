import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/optimization_baselines")
FIGURES_DIR = Path("figures/optimization_baselines")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_matrix(m, n, rng):
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    return A


def make_blocks(n, block_size):
    return [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]


def sample_block_sparse_signal(n, k, block_size, rng):
    blocks = make_blocks(n, block_size)
    q = math.ceil(k / block_size)
    active_blocks = rng.choice(len(blocks), size=q, replace=False)

    support = []
    for b in active_blocks:
        support.extend(blocks[b].tolist())

    support = np.array(support[:k])
    x = np.zeros(n)
    x[support] = rng.normal(size=len(support))
    return x, support


def fit_on_support(A, y, support, n):
    support = np.array(sorted(set(support)), dtype=int)
    xhat = np.zeros(n)

    if len(support) == 0:
        return xhat

    As = A[:, support]
    coef, *_ = np.linalg.lstsq(As, y, rcond=None)
    xhat[support] = coef
    return xhat


def nrmse(xhat, x):
    return np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-12)


def support_iou(xhat, support_true, k):
    pred = set(np.argsort(np.abs(xhat))[-k:].tolist())
    true = set(support_true.tolist())
    inter = len(pred & true)
    union = len(pred | true)
    return inter / max(union, 1)


def topk_refit(A, y, scores, k):
    n = A.shape[1]
    support = np.argsort(np.abs(scores))[-k:]
    return fit_on_support(A, y, support, n)


def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def ista(A, y, lam, n_iters=300):
    n = A.shape[1]
    x = np.zeros(n)
    L = np.linalg.norm(A, ord=2) ** 2
    step = 1.0 / (L + 1e-12)

    for _ in range(n_iters):
        grad = A.T @ (A @ x - y)
        x = soft_threshold(x - step * grad, lam * step)

    return x


def fista(A, y, lam, n_iters=300):
    n = A.shape[1]
    x = np.zeros(n)
    z = x.copy()
    t = 1.0

    L = np.linalg.norm(A, ord=2) ** 2
    step = 1.0 / (L + 1e-12)

    for _ in range(n_iters):
        x_old = x.copy()
        grad = A.T @ (A @ z - y)
        x = soft_threshold(z - step * grad, lam * step)

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new

    return x


def cosamp(A, y, k, n_iters=20):
    m, n = A.shape
    residual = y.copy()
    support = set()
    xhat = np.zeros(n)

    for _ in range(n_iters):
        proxy = A.T @ residual
        omega = set(np.argsort(np.abs(proxy))[-2 * k:].tolist())
        merged = np.array(sorted(support | omega), dtype=int)

        b = fit_on_support(A, y, merged, n)
        support = set(np.argsort(np.abs(b))[-k:].tolist())

        xhat = fit_on_support(A, y, support, n)
        residual = y - A @ xhat

    return xhat


def mean_se(vals):
    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, se


def run_setting(m, k, seeds, n=256, block_size=5, n_test=300):
    methods = ["cosamp", "ista", "fista"]
    all_vals = {method: {"nrmse": [], "iou": []} for method in methods}

    # Lambda grid for ISTA/FISTA.
    # We tune lightly on each trial by choosing the best reconstruction residual after top-k refit.
    lambdas = [0.001, 0.003, 0.01, 0.03, 0.1]

    for seed in seeds:
        rng = np.random.default_rng(seed)
        A = make_matrix(m, n, rng)

        for _ in range(n_test):
            x, support_true = sample_block_sparse_signal(n, k, block_size, rng)
            y = A @ x

            # CoSaMP
            xhat = cosamp(A, y, k)
            all_vals["cosamp"]["nrmse"].append(nrmse(xhat, x))
            all_vals["cosamp"]["iou"].append(support_iou(xhat, support_true, k))

            # ISTA with small lambda sweep + top-k refit
            best_xhat = None
            best_res = float("inf")
            for lam in lambdas:
                z = ista(A, y, lam)
                cand = topk_refit(A, y, z, k)
                res = np.linalg.norm(y - A @ cand)
                if res < best_res:
                    best_res = res
                    best_xhat = cand

            all_vals["ista"]["nrmse"].append(nrmse(best_xhat, x))
            all_vals["ista"]["iou"].append(support_iou(best_xhat, support_true, k))

            # FISTA with small lambda sweep + top-k refit
            best_xhat = None
            best_res = float("inf")
            for lam in lambdas:
                z = fista(A, y, lam)
                cand = topk_refit(A, y, z, k)
                res = np.linalg.norm(y - A @ cand)
                if res < best_res:
                    best_res = res
                    best_xhat = cand

            all_vals["fista"]["nrmse"].append(nrmse(best_xhat, x))
            all_vals["fista"]["iou"].append(support_iou(best_xhat, support_true, k))

    summary = {}
    for method in methods:
        nrmse_mean, nrmse_se = mean_se(all_vals[method]["nrmse"])
        iou_mean, iou_se = mean_se(all_vals[method]["iou"])
        summary[method] = {
            "nrmse_mean": nrmse_mean,
            "nrmse_se": nrmse_se,
            "iou_mean": iou_mean,
            "iou_se": iou_se,
        }

    return summary


def main():
    settings = [(96, 40), (96, 55)]
    seeds = list(range(10))

    results = {}

    for m, k in settings:
        print(f"Running optimization baselines: m={m}, k={k}")
        key = f"m={m},k={k}"
        results[key] = run_setting(m, k, seeds)

    out_json = RESULTS_DIR / "optimization_baselines_10seed.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_json}")

    print("\nOptimization baseline results")
    print("-" * 80)
    print(f"{'setting':12s} {'method':10s} {'NRMSE':>10s} {'SE':>10s} {'IoU':>10s}")
    for setting, table in results.items():
        for method, row in table.items():
            print(
                f"{setting:12s} {method:10s} "
                f"{row['nrmse_mean']:10.4f} "
                f"{row['nrmse_se']:10.4f} "
                f"{row['iou_mean']:10.4f}"
            )

    # Plot
    methods = ["cosamp", "ista", "fista"]
    x = np.arange(len(settings))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        means = []
        ses = []
        for m, k in settings:
            key = f"m={m},k={k}"
            means.append(results[key][method]["nrmse_mean"])
            ses.append(results[key][method]["nrmse_se"])

        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=ses,
            capsize=4,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}, k={k}" for m, k in settings])
    ax.set_ylabel("NRMSE")
    ax.set_title("Optimization baselines: CoSaMP vs ISTA/FISTA")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_fig = FIGURES_DIR / "optimization_baselines_10seed_nrmse.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
