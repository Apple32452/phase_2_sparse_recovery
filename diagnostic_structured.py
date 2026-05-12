"""
Diagnostic: is there headroom for a learned local-identifiability scorer?

We compare three sparse-recovery support selectors across operator families
that vary in their coherence structure:

  (1) Naive top-k of |A^T y|        -- one-shot coherence-based score
  (2) OMP                            -- sequential coherence-based greedy
  (3) OOMP (residual-minimizing)     -- sequential greedy with full
                                        projection-aware scoring
  (4) True-support oracle            -- ceiling

OMP picks j = argmax |A_{:,j}^T r_t|.  OOMP picks the j that maximally
reduces ||y - P_{span(A_{S \cup {j}})} y||.  These coincide when columns
are near-orthogonal (Gaussian) and diverge when columns share span
(block-coherent, spike-tail).  A learned local-identifiability scorer would
approximate something between OOMP and a Bayes-optimal selector; OOMP
gives a concrete, achievable upper bound on what coherence-only greedy
methods can be improved to.

If support-IoU(OOMP) >> support-IoU(OMP) on structured matrices but they
agree on Gaussian, the proposed research direction has measurable headroom.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Operator constructors
# ----------------------------------------------------------------------

def make_gaussian(m, n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A

def make_block_coherent(m, n, seed, n_blocks=8, intra_corr=0.85):
    """Columns within a block share most of their direction with an anchor."""
    rng = np.random.default_rng(seed)
    block_size = n // n_blocks
    A = np.zeros((m, n))
    for b in range(n_blocks):
        anchor = rng.standard_normal(m)
        anchor /= np.linalg.norm(anchor)
        for j in range(b * block_size, (b + 1) * block_size):
            noise = rng.standard_normal(m)
            noise -= (noise @ anchor) * anchor
            noise /= np.linalg.norm(noise)
            col = intra_corr * anchor + np.sqrt(1 - intra_corr**2) * noise
            A[:, j] = col / np.linalg.norm(col)
    return A

def make_spike_tail(m, n, seed, n_bad_pairs=40, bad_corr=0.97):
    """Gaussian backbone, plus a handful of near-collinear column pairs."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    for _ in range(n_bad_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        noise = rng.standard_normal(m)
        noise -= (noise @ A[:, i]) * A[:, i]
        noise /= np.linalg.norm(noise)
        A[:, j] = bad_corr * A[:, i] + np.sqrt(1 - bad_corr**2) * noise
        A[:, j] /= np.linalg.norm(A[:, j])
    return A

def mutual_coherence(A):
    G = A.T @ A
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))

# ----------------------------------------------------------------------
# Signal generation
# ----------------------------------------------------------------------

def gen_signal(n, k, seed, amp_lo=0.5, amp_hi=2.0):
    rng = np.random.default_rng(seed)
    S = rng.choice(n, size=k, replace=False)
    x = np.zeros(n)
    sign = rng.choice([-1.0, 1.0], size=k)
    amp = rng.uniform(amp_lo, amp_hi, size=k)
    x[S] = sign * amp
    return x, set(int(s) for s in S)

# ----------------------------------------------------------------------
# Selectors
# ----------------------------------------------------------------------

def naive_topk(A, y, k):
    scores = np.abs(A.T @ y)
    idx = np.argpartition(-scores, k - 1)[:k]
    return set(int(i) for i in idx)

def omp(A, y, k):
    n = A.shape[1]
    r = y.copy()
    selected = []
    for _ in range(k):
        scores = np.abs(A.T @ r)
        for s in selected:
            scores[s] = -np.inf
        j = int(np.argmax(scores))
        selected.append(j)
        A_S = A[:, selected]
        x_S, *_ = np.linalg.lstsq(A_S, y, rcond=None)
        r = y - A_S @ x_S
    return set(selected)

def oomp(A, y, k):
    """Residual-minimizing greedy via orthogonal projection updates."""
    m, n = A.shape
    A_perp = A.copy()
    y_perp = y.copy()
    selected = []
    for _ in range(k):
        col_norms = np.linalg.norm(A_perp, axis=0)
        col_norms = np.maximum(col_norms, 1e-12)
        inner = A_perp.T @ y_perp
        scores = (inner ** 2) / (col_norms ** 2)
        for s in selected:
            scores[s] = -np.inf
        j = int(np.argmax(scores))
        selected.append(j)
        q = A_perp[:, j] / col_norms[j]
        A_perp = A_perp - np.outer(q, q @ A_perp)
        y_perp = y_perp - q * (q @ y_perp)
    return set(selected)

def cosamp(A, y, k, max_iters=30):
    """Needell & Tropp 2009: add-and-prune support refinement."""
    n = A.shape[1]
    x = np.zeros(n)
    S_prev = set()
    for _ in range(max_iters):
        r = y - A @ x
        u = A.T @ r
        # identification: top 2k of proxy
        omega = set(int(i) for i in np.argpartition(-np.abs(u), 2 * k - 1)[:2 * k])
        # merge with current support
        T = sorted(omega | set(int(i) for i in np.nonzero(x)[0]))
        # signal estimation on merged support
        A_T = A[:, T]
        b_T, *_ = np.linalg.lstsq(A_T, y, rcond=None)
        b = np.zeros(n)
        b[T] = b_T
        # prune to k-sparse
        S_new = set(int(i) for i in np.argpartition(-np.abs(b), k - 1)[:k])
        # restrict amplitudes to pruned support
        S_list = sorted(S_new)
        x = np.zeros(n)
        x[S_list], *_ = np.linalg.lstsq(A[:, S_list], y, rcond=None)
        if S_new == S_prev:
            break
        S_prev = S_new
    return S_new

def iht(A, y, k, max_iters=300, step=None):
    """Blumensath & Davies 2009 hard-thresholding gradient descent."""
    n = A.shape[1]
    if step is None:
        spec = float(np.linalg.norm(A, 2))
        step = 0.95 / (spec ** 2)
    x = np.zeros(n)
    for _ in range(max_iters):
        x_new = x + step * A.T @ (y - A @ x)
        idx = np.argpartition(-np.abs(x_new), k - 1)[:k]
        x_thresh = np.zeros(n)
        x_thresh[idx] = x_new[idx]
        if np.allclose(x_thresh, x, atol=1e-8, rtol=1e-6):
            x = x_thresh
            break
        x = x_thresh
    return set(int(i) for i in np.nonzero(x)[0])

# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------

def iou(S_pred, S_true):
    inter = len(S_pred & S_true)
    union = len(S_pred | S_true)
    return inter / union if union > 0 else 0.0

def support_ls_nrmse(A, y, S_pred, x_true):
    if len(S_pred) == 0:
        return 1.0
    S_list = sorted(S_pred)
    A_S = A[:, S_list]
    x_S, *_ = np.linalg.lstsq(A_S, y, rcond=None)
    x_hat = np.zeros_like(x_true)
    x_hat[S_list] = x_S
    return float(np.linalg.norm(x_hat - x_true) / max(np.linalg.norm(x_true), 1e-12))

def evaluate(A, n_signals, k, signal_seed_base):
    method_names = ["naive_topk", "omp", "oomp", "cosamp", "iht", "oracle"]
    metrics = {m: {"iou": [], "nrmse": []} for m in method_names}
    selectors = [
        ("naive_topk", naive_topk),
        ("omp", omp),
        ("oomp", oomp),
        ("cosamp", cosamp),
        ("iht", iht),
    ]
    for s in range(n_signals):
        x, S_true = gen_signal(A.shape[1], k, signal_seed_base + s)
        y = A @ x
        for name, fn in selectors:
            S_pred = fn(A, y, k)
            metrics[name]["iou"].append(iou(S_pred, S_true))
            metrics[name]["nrmse"].append(support_ls_nrmse(A, y, S_pred, x))
        metrics["oracle"]["iou"].append(1.0)
        metrics["oracle"]["nrmse"].append(support_ls_nrmse(A, y, S_true, x))
    summary = {}
    for name, d in metrics.items():
        summary[name] = {
            "iou_mean": float(np.mean(d["iou"])),
            "iou_std": float(np.std(d["iou"])),
            "nrmse_mean": float(np.mean(d["nrmse"])),
            "nrmse_std": float(np.std(d["nrmse"])),
        }
    return summary

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    n, m, k = 256, 128, 25
    n_signals = 50

    families = [
        ("Gaussian (control)", make_gaussian(m, n, seed=0)),
        ("Block-coherent (8 blocks, 0.85)",
            make_block_coherent(m, n, seed=1, n_blocks=8, intra_corr=0.85)),
        ("Spike-tail (40 bad pairs, 0.97)",
            make_spike_tail(m, n, seed=2, n_bad_pairs=40, bad_corr=0.97)),
    ]

    results = {}
    for fam_name, A in families:
        mu = mutual_coherence(A)
        summary = evaluate(A, n_signals=n_signals, k=k, signal_seed_base=1000)
        results[fam_name] = {"mu": mu, "metrics": summary}
        print(f"\n[{fam_name}]  mu(A) = {mu:.4f}")
        print(f"  {'method':<14} {'IoU':>14}   {'NRMSE':>14}")
        for name in ["naive_topk", "omp", "oomp", "cosamp", "iht", "oracle"]:
            s = summary[name]
            print(f"  {name:<14} "
                  f"{s['iou_mean']:.3f} ± {s['iou_std']:.3f}    "
                  f"{s['nrmse_mean']:.3f} ± {s['nrmse_std']:.3f}")

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    method_order = ["naive_topk", "omp", "oomp", "cosamp", "iht", "oracle"]
    method_colors = {
        "naive_topk": "#bdbdbd",
        "omp": "#d95f02",
        "oomp": "#1b9e77",
        "cosamp": "#e7298a",
        "iht": "#66a61e",
        "oracle": "#7570b3",
    }
    fam_names = list(results.keys())
    fam_short = ["Gaussian", "Block-coh.", "Spike-tail"]
    x_pos = np.arange(len(fam_names))
    width = 0.13

    for i, name in enumerate(method_order):
        ious = [results[f]["metrics"][name]["iou_mean"] for f in fam_names]
        nrm = [results[f]["metrics"][name]["nrmse_mean"] for f in fam_names]
        offset = (i - (len(method_order) - 1) / 2) * width
        axes[0].bar(x_pos + offset, ious, width=width,
                    label=name, color=method_colors[name], edgecolor="black", linewidth=0.4)
        axes[1].bar(x_pos + offset, nrm, width=width,
                    label=name, color=method_colors[name], edgecolor="black", linewidth=0.4)

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(fam_short, fontsize=9)
    axes[0].set_ylabel("Support IoU")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Support recovery", fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.3, linewidth=0.5)
    axes[0].legend(fontsize=7, loc="lower right")

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(fam_short, fontsize=9)
    axes[1].set_ylabel("NRMSE (support-LS)")
    axes[1].set_title("Reconstruction error", fontsize=10)
    axes[1].grid(True, axis="y", alpha=0.3, linewidth=0.5)
    axes[1].legend(fontsize=7, loc="upper left")

    fig.tight_layout(pad=0.5)
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "diagnostic_structured.png"
    out_json = out_dir / "diagnostic_structured.json"
    fig.savefig(out_png, dpi=180)
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_png}")
    print(f"Wrote {out_json}")

    # ----- Verdict -----
    print("\n=== Headroom over OMP (IoU gain), per family ===")
    print(f"  {'family':<40}  {'OOMP':>8} {'CoSaMP':>8} {'IHT':>8}")
    for fam_name in fam_names:
        omp_iou = results[fam_name]["metrics"]["omp"]["iou_mean"]
        gaps = {
            m: results[fam_name]["metrics"][m]["iou_mean"] - omp_iou
            for m in ["oomp", "cosamp", "iht"]
        }
        print(f"  {fam_name:<40}  "
              f"{gaps['oomp']:+.3f}  {gaps['cosamp']:+.3f}  {gaps['iht']:+.3f}")

    print("\n=== Remaining gap to oracle (IoU), per family ===")
    print(f"  {'family':<40}  {'OMP':>8} {'CoSaMP':>8} {'IHT':>8}")
    for fam_name in fam_names:
        gaps = {
            m: 1.0 - results[fam_name]["metrics"][m]["iou_mean"]
            for m in ["omp", "cosamp", "iht"]
        }
        print(f"  {fam_name:<40}  "
              f"{gaps['omp']:.3f}    {gaps['cosamp']:.3f}    {gaps['iht']:.3f}")

if __name__ == "__main__":
    main()
