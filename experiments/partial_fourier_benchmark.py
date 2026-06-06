import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/partial_fourier_benchmark")
FIGURES_DIR = Path("figures/partial_fourier_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_partial_fourier_matrix(m, n, rng):
    rows = rng.choice(n, size=m, replace=False)
    j = np.arange(n)
    A = np.exp(-2j * np.pi * rows[:, None] * j[None, :] / n) / np.sqrt(m)
    return A


def make_blocks(n, block_size):
    return [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]


def sample_block_sparse_signal(n, k, block_size, rng):
    blocks = make_blocks(n, block_size)
    q = k // block_size
    active_blocks = rng.choice(len(blocks), size=q, replace=False)

    support = []
    for b in active_blocks:
        support.extend(blocks[b].tolist())

    support = np.array(sorted(support), dtype=int)

    x = np.zeros(n)
    x[support] = rng.normal(size=len(support))

    return x, support


def fit_on_support(A, y, support, n, ridge=1e-8):
    """
    Stable least-squares refit on a candidate support.

    Uses ordinary least squares when possible. If the support is large or
    the system is ill-conditioned, falls back to ridge-stabilized least squares.
    This is important for partial Fourier designs when temporary CoSaMP
    supports can exceed the number of measurements.
    """
    support = np.array(sorted(set(support)), dtype=int)
    xhat = np.zeros(n, dtype=complex)

    if len(support) == 0:
        return xhat

    # Avoid impossible / extremely unstable oversized support fits.
    # Keep the largest requested support but cap numerical solve size.
    As = A[:, support]

    try:
        # Try standard least squares first.
        coef, *_ = np.linalg.lstsq(As, y, rcond=1e-10)
    except np.linalg.LinAlgError:
        # Ridge fallback: solve (A_S^* A_S + lambda I)c = A_S^* y.
        G = As.conj().T @ As
        b = As.conj().T @ y
        reg = ridge * np.trace(G).real / max(G.shape[0], 1)
        if reg <= 0 or not np.isfinite(reg):
            reg = ridge

        coef = np.linalg.solve(G + reg * np.eye(G.shape[0]), b)

    xhat[support] = coef
    return xhat


def nrmse(xhat, x):
    return float(np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-12))


def support_iou(xhat, support_true, k):
    pred = set(np.argsort(np.abs(xhat))[-k:].tolist())
    true = set(support_true.tolist())

    inter = len(pred & true)
    union = len(pred | true)

    return inter / max(union, 1)


def block_scores_from_vector(v, blocks):
    scores = []
    abs_v = np.abs(v)
    for block in blocks:
        scores.append(float(np.sum(abs_v[block])))
    return np.array(scores)


def support_from_blocks(blocks, block_ids):
    support = []
    for b in block_ids:
        support.extend(blocks[b].tolist())
    return np.array(sorted(support), dtype=int)


def block_score_topk(A, y, k, block_size):
    n = A.shape[1]
    blocks = make_blocks(n, block_size)
    q = k // block_size

    proxy = A.conj().T @ y
    scores = block_scores_from_vector(proxy, blocks)

    chosen_blocks = np.argsort(scores)[-q:]
    support = support_from_blocks(blocks, chosen_blocks)

    return fit_on_support(A, y, support, n)


def cosamp(A, y, k, n_iters=20):
    """
    Numerically stable CoSaMP for partial Fourier sensing.

    Standard CoSaMP forms a temporary support of size up to 3k.
    When k is large relative to m, this can exceed the number of measurements
    and create unstable least-squares problems. We cap the temporary support
    size to at most m - 1 for numerical stability.
    """
    m, n = A.shape
    residual = y.copy()
    support = set()
    xhat = np.zeros(n, dtype=complex)

    # CoSaMP usually uses 2k new candidates, but do not exceed m - 1.
    proxy_size = min(2 * k, max(1, m - 1))
    merged_cap = min(3 * k, max(1, m - 1))

    for _ in range(n_iters):
        proxy = A.conj().T @ residual

        omega = set(np.argsort(np.abs(proxy))[-proxy_size:].tolist())
        merged = list(support | omega)

        # If merged support is too large, keep only strongest proxy entries.
        if len(merged) > merged_cap:
            merged = sorted(
                merged,
                key=lambda j: np.abs(proxy[j]),
                reverse=True,
            )[:merged_cap]

        merged = np.array(sorted(merged), dtype=int)

        b = fit_on_support(A, y, merged, n)

        support = set(np.argsort(np.abs(b))[-k:].tolist())
        xhat = fit_on_support(A, y, support, n)

        residual = y - A @ xhat

        if np.linalg.norm(residual) < 1e-10:
            break

    return xhat


def adaptive_block_refinement(A, y, k, block_size, max_iters=4, min_improvement=1e-8):
    """
    Residual-verified block-swap refinement.

    Start from block-score top-k support. At each step, propose replacing one
    selected block with one high-residual unselected block. Accept only if the
    least-squares residual decreases.
    """
    n = A.shape[1]
    blocks = make_blocks(n, block_size)
    q = k // block_size

    proxy = A.conj().T @ y
    init_scores = block_scores_from_vector(proxy, blocks)
    active_blocks = set(np.argsort(init_scores)[-q:].tolist())

    support = support_from_blocks(blocks, active_blocks)
    xhat = fit_on_support(A, y, support, n)
    best_res = float(np.linalg.norm(y - A @ xhat) ** 2)

    accepted_steps = 0

    for _ in range(max_iters):
        residual = y - A @ xhat
        residual_proxy = A.conj().T @ residual
        residual_scores = block_scores_from_vector(residual_proxy, blocks)

        inactive_blocks = [b for b in range(len(blocks)) if b not in active_blocks]
        candidate_adds = sorted(
            inactive_blocks,
            key=lambda b: residual_scores[b],
            reverse=True,
        )[:10]

        active_list = list(active_blocks)

        # Drop the weakest currently selected block based on current proxy score.
        candidate_drops = sorted(
            active_list,
            key=lambda b: init_scores[b],
        )[:10]

        best_candidate = None
        best_candidate_res = best_res
        best_candidate_xhat = None

        for add_b in candidate_adds:
            for drop_b in candidate_drops:
                new_blocks = set(active_blocks)
                new_blocks.remove(drop_b)
                new_blocks.add(add_b)

                new_support = support_from_blocks(blocks, new_blocks)
                cand_xhat = fit_on_support(A, y, new_support, n)
                cand_res = float(np.linalg.norm(y - A @ cand_xhat) ** 2)

                if cand_res + min_improvement < best_candidate_res:
                    best_candidate_res = cand_res
                    best_candidate = new_blocks
                    best_candidate_xhat = cand_xhat

        if best_candidate is None:
            break

        active_blocks = best_candidate
        xhat = best_candidate_xhat
        best_res = best_candidate_res
        accepted_steps += 1

    return xhat, accepted_steps


def oracle(A, y, support_true, n):
    return fit_on_support(A, y, support_true, n)


def mean_se(vals):
    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, se


def run_setting(m, k, seeds, n=256, block_size=5, n_test=300):
    methods = [
        "cosamp",
        "block_score_topk",
        "adaptive_block_refinement",
        "oracle",
    ]

    vals = {method: {"nrmse": [], "iou": []} for method in methods}
    accepted_steps = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        A = make_partial_fourier_matrix(m, n, rng)

        for _ in range(n_test):
            x, support_true = sample_block_sparse_signal(n, k, block_size, rng)
            y = A @ x

            xhat = cosamp(A, y, k)
            vals["cosamp"]["nrmse"].append(nrmse(xhat, x))
            vals["cosamp"]["iou"].append(support_iou(xhat, support_true, k))

            xhat = block_score_topk(A, y, k, block_size)
            vals["block_score_topk"]["nrmse"].append(nrmse(xhat, x))
            vals["block_score_topk"]["iou"].append(support_iou(xhat, support_true, k))

            xhat, steps = adaptive_block_refinement(A, y, k, block_size)
            vals["adaptive_block_refinement"]["nrmse"].append(nrmse(xhat, x))
            vals["adaptive_block_refinement"]["iou"].append(support_iou(xhat, support_true, k))
            accepted_steps.append(steps)

            xhat = oracle(A, y, support_true, n)
            vals["oracle"]["nrmse"].append(nrmse(xhat, x))

            # Oracle uses the true support by definition, so its support IoU is exactly one.
            vals["oracle"]["iou"].append(1.0)

    summary = {}

    for method in methods:
        nrmse_mean, nrmse_se = mean_se(vals[method]["nrmse"])
        iou_mean, iou_se = mean_se(vals[method]["iou"])

        summary[method] = {
            "nrmse_mean": nrmse_mean,
            "nrmse_se": nrmse_se,
            "iou_mean": iou_mean,
            "iou_se": iou_se,
        }

    summary["accepted_steps"] = {
        "mean": float(np.mean(accepted_steps)),
        "se": float(np.std(accepted_steps, ddof=1) / np.sqrt(len(accepted_steps))),
    }

    return summary


def main():
    settings = [(96, 40), (96, 55), (112, 55), (128, 70)]
    seeds = list(range(10))

    results = {}

    for m, k in settings:
        print("=" * 80)
        print(f"Partial Fourier benchmark: m={m}, k={k}")
        key = f"m={m},k={k}"
        results[key] = run_setting(m, k, seeds)

    out_json = RESULTS_DIR / "partial_fourier_benchmark_10seed.json"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_json}")

    print("\nPartial Fourier benchmark results")
    print("-" * 100)
    print(f"{'setting':12s} {'method':30s} {'NRMSE':>10s} {'SE':>10s} {'IoU':>10s}")

    for setting, table in results.items():
        for method, row in table.items():
            if method == "accepted_steps":
                continue
            print(
                f"{setting:12s} {method:30s} "
                f"{row['nrmse_mean']:10.4f} "
                f"{row['nrmse_se']:10.4f} "
                f"{row['iou_mean']:10.4f}"
            )

    methods = [
        "cosamp",
        "block_score_topk",
        "adaptive_block_refinement",
        "oracle",
    ]

    x = np.arange(len(settings))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, method in enumerate(methods):
        means = []
        ses = []

        for m, k in settings:
            key = f"m={m},k={k}"
            means.append(results[key][method]["nrmse_mean"])
            ses.append(results[key][method]["nrmse_se"])

        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=ses,
            capsize=4,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}, k={k}" for m, k in settings])
    ax.set_ylabel("NRMSE")
    ax.set_title("Partial Fourier compressed sensing benchmark")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_fig = FIGURES_DIR / "partial_fourier_benchmark_10seed_nrmse.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")

    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
