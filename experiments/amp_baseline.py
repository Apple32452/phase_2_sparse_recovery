import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/amp_baseline")
FIGURES_DIR = Path("figures/amp_baseline")
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

    support = np.array(sorted(support[:k]), dtype=int)

    x = np.zeros(n)
    x[support] = rng.normal(size=len(support))
    return x, support


def fit_on_support(A, y, support, n, ridge=1e-8):
    support = np.array(sorted(set(support)), dtype=int)
    xhat = np.zeros(n)

    if len(support) == 0:
        return xhat

    As = A[:, support]

    try:
        coef, *_ = np.linalg.lstsq(As, y, rcond=1e-10)
    except np.linalg.LinAlgError:
        G = As.T @ As
        b = As.T @ y
        reg = ridge * np.trace(G).real / max(G.shape[0], 1)
        if reg <= 0 or not np.isfinite(reg):
            reg = ridge
        coef = np.linalg.solve(G + reg * np.eye(G.shape[0]), b)

    xhat[support] = coef
    return xhat


def nrmse(xhat, x):
    return float(np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-12))


def support_iou_from_xhat(xhat, support_true, k):
    pred = set(np.argsort(np.abs(xhat))[-k:].tolist())
    true = set(support_true.tolist())
    inter = len(pred & true)
    union = len(pred | true)
    return inter / max(union, 1)


def topk_refit(A, y, scores, k):
    n = A.shape[1]
    support = np.argsort(np.abs(scores))[-k:]
    return fit_on_support(A, y, support, n)


def soft_threshold(z, theta):
    return np.sign(z) * np.maximum(np.abs(z) - theta, 0.0)


def amp(A, y, lam=1.0, n_iters=60, damping=0.5):
    """
    Vanilla AMP with soft-threshold denoising.

    This is intended for Gaussian compressed sensing baselines.
    We later use top-k least-squares refitting for fair support-aware comparison.
    """
    m, n = A.shape
    delta = m / n

    x = np.zeros(n)
    z = y.copy()

    for _ in range(n_iters):
        sigma_hat = np.linalg.norm(z) / np.sqrt(m)
        theta = lam * sigma_hat

        pseudo = x + A.T @ z
        x_new = soft_threshold(pseudo, theta)

        div = np.mean(np.abs(pseudo) > theta)
        z_new = y - A @ x_new + (div / max(delta, 1e-12)) * z

        # Damping improves stability for finite n and column-normalized A.
        x = damping * x_new + (1.0 - damping) * x
        z = damping * z_new + (1.0 - damping) * z

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
            break

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

        if np.linalg.norm(residual) < 1e-10:
            break

    return xhat


def mean_se(vals):
    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, se


def evaluate_setting(m, k, seeds, n=256, block_size=5, n_test=300):
    methods = [
        "cosamp",
        "amp_raw",
        "amp_topk_refit",
        "oracle",
    ]

    values = {method: {"nrmse": [], "iou": []} for method in methods}

    # Lambda grid for AMP threshold. We choose the best top-k refit by residual.
    # This gives AMP a fair, strong support-aware version.
    amp_lams = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    for seed in seeds:
        print(f"  seed={seed}")
        rng = np.random.default_rng(seed)
        A = make_matrix(m, n, rng)

        for _ in range(n_test):
            x, support_true = sample_block_sparse_signal(n, k, block_size, rng)
            y = A @ x

            # CoSaMP
            xhat = cosamp(A, y, k)
            values["cosamp"]["nrmse"].append(nrmse(xhat, x))
            values["cosamp"]["iou"].append(support_iou_from_xhat(xhat, support_true, k))

            # AMP: select lambda by top-k refit residual
            best_raw = None
            best_refit = None
            best_res = float("inf")

            for lam in amp_lams:
                x_amp = amp(A, y, lam=lam, n_iters=60, damping=0.5)
                x_refit = topk_refit(A, y, x_amp, k)
                res = np.linalg.norm(y - A @ x_refit)

                if np.isfinite(res) and res < best_res:
                    best_res = res
                    best_raw = x_amp
                    best_refit = x_refit

            if best_raw is None:
                best_raw = np.zeros(n)
                best_refit = np.zeros(n)

            values["amp_raw"]["nrmse"].append(nrmse(best_raw, x))
            values["amp_raw"]["iou"].append(support_iou_from_xhat(best_raw, support_true, k))

            values["amp_topk_refit"]["nrmse"].append(nrmse(best_refit, x))
            values["amp_topk_refit"]["iou"].append(support_iou_from_xhat(best_refit, support_true, k))

            # Oracle support
            x_oracle = fit_on_support(A, y, support_true, n)
            values["oracle"]["nrmse"].append(nrmse(x_oracle, x))
            values["oracle"]["iou"].append(1.0)

    summary = {}

    for method in methods:
        nrmse_mean, nrmse_se = mean_se(values[method]["nrmse"])
        iou_mean, iou_se = mean_se(values[method]["iou"])

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
        print("=" * 80)
        print(f"AMP baseline: m={m}, k={k}")
        key = f"m={m},k={k}"
        results[key] = evaluate_setting(m, k, seeds)

    out_json = RESULTS_DIR / "amp_baseline_10seed.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_json}")

    print("\nAMP baseline results")
    print("-" * 90)
    print(f"{'setting':12s} {'method':18s} {'NRMSE':>10s} {'SE':>10s} {'IoU':>10s}")

    for setting, table in results.items():
        for method, row in table.items():
            print(
                f"{setting:12s} {method:18s} "
                f"{row['nrmse_mean']:10.4f} "
                f"{row['nrmse_se']:10.4f} "
                f"{row['iou_mean']:10.4f}"
            )

    # Reference adaptive numbers from your current 10-seed noiseless main result.
    adaptive_ref = {
        "m=96,k=40": {"nrmse_mean": 0.0037, "nrmse_se": 0.0006, "iou_mean": 0.9966},
        "m=96,k=55": {"nrmse_mean": 0.1087, "nrmse_se": 0.0038, "iou_mean": 0.9248},
    }

    plot_methods = ["cosamp", "amp_raw", "amp_topk_refit", "adaptive_refinement", "oracle"]
    labels = {
        "cosamp": "CoSaMP",
        "amp_raw": "AMP raw",
        "amp_topk_refit": "AMP top-k refit",
        "adaptive_refinement": "adaptive",
        "oracle": "oracle",
    }

    xloc = np.arange(len(settings))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, method in enumerate(plot_methods):
        means = []
        ses = []

        for m, k in settings:
            key = f"m={m},k={k}"

            if method == "adaptive_refinement":
                means.append(adaptive_ref[key]["nrmse_mean"])
                ses.append(adaptive_ref[key]["nrmse_se"])
            else:
                means.append(results[key][method]["nrmse_mean"])
                ses.append(results[key][method]["nrmse_se"])

        ax.bar(
            xloc + (i - 2) * width,
            means,
            width,
            yerr=ses,
            capsize=4,
            label=labels[method],
        )

    ax.set_xticks(xloc)
    ax.set_xticklabels([f"m={m}, k={k}" for m, k in settings])
    ax.set_ylabel("NRMSE")
    ax.set_title("AMP baseline vs adaptive refinement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_fig = FIGURES_DIR / "amp_baseline_10seed_nrmse.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")

    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
