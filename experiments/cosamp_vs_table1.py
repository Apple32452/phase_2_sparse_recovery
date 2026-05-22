"""
Run CoSaMP on the abstract's exact Fourier and Gaussian test sets and
compare against the existing Table 1 numbers.

Uses the same operator construction (make_operators, seed=0) and the same
test signals (seed=42+999=1041) as phase_1/ista_comparison.py, so the
NRMSE numbers are directly comparable to the saved
phase_1/results/ista_comparison_T30_lam0.05.json.

This decides whether CoSaMP eats the abstract's main result.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TABLE1_JSON = ROOT / "phase_1" / "results" / "ista_comparison_T30_lam0.05.json"

# ----------------------------------------------------------------------
# Reproduce the abstract's operators and signals (numpy, no torch needed)
# ----------------------------------------------------------------------

def make_operators(n, m, seed=0):
    """Identical to phase_1/ista_comparison.py make_operators."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, m // 2, replace=False)
    F_complex = np.fft.fft(np.eye(n)) / np.sqrt(n)
    rows = np.concatenate([F_complex[idx].real, F_complex[idx].imag], axis=0)
    A_F = rows[:m].astype(np.float32)

    rng2 = np.random.RandomState(seed + 1)
    A_G = (rng2.randn(m, n) / np.sqrt(m)).astype(np.float32)
    return A_F, A_G

def make_signals(n, k, n_signals, amp_lo, amp_hi, seed):
    """Identical to phase_1/ista_comparison.py make_signals."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_signals, n), dtype=np.float32)
    S = np.zeros((n_signals, n), dtype=np.float32)
    for i in range(n_signals):
        supp = rng.choice(n, k, replace=False)
        amps = rng.uniform(amp_lo, amp_hi, k) * rng.choice([-1, 1], k)
        X[i, supp] = amps
        S[i, supp] = 1.0
    return X, S

# ----------------------------------------------------------------------
# Selectors
# ----------------------------------------------------------------------

def cosamp(A, y, k, max_iters=30):
    n = A.shape[1]
    x = np.zeros(n)
    S_prev = set()
    for _ in range(max_iters):
        r = y - A @ x
        u = A.T @ r
        omega = set(int(i) for i in np.argpartition(-np.abs(u), 2 * k - 1)[:2 * k])
        T = sorted(omega | set(int(i) for i in np.nonzero(x)[0]))
        A_T = A[:, T]
        b_T, *_ = np.linalg.lstsq(A_T, y, rcond=None)
        b = np.zeros(n)
        b[T] = b_T
        S_new = set(int(i) for i in np.argpartition(-np.abs(b), k - 1)[:k])
        S_list = sorted(S_new)
        x = np.zeros(n)
        x[S_list], *_ = np.linalg.lstsq(A[:, S_list], y, rcond=None)
        if S_new == S_prev:
            break
        S_prev = S_new
    return S_new, x

def htp(A, y, k, max_iters=100, step=None):
    """Hard Thresholding Pursuit: gradient step + top-k + LS refit."""
    n = A.shape[1]
    if step is None:
        spec = float(np.linalg.norm(A, 2))
        step = 0.95 / (spec ** 2)
    x = np.zeros(n)
    S_prev = set()
    for _ in range(max_iters):
        x_aux = x + step * A.T @ (y - A @ x)
        idx = np.argpartition(-np.abs(x_aux), k - 1)[:k]
        S_new = set(int(i) for i in idx)
        S_list = sorted(S_new)
        x = np.zeros(n)
        x[S_list], *_ = np.linalg.lstsq(A[:, S_list], y, rcond=None)
        if S_new == S_prev:
            break
        S_prev = S_new
    return S_new, x

# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------

def iou(S_pred, S_true):
    inter = len(S_pred & S_true)
    union = len(S_pred | S_true)
    return inter / union if union > 0 else 0.0

def evaluate_method(name, fn, A, X, S, k):
    n_signals = X.shape[0]
    nrmses = []
    ious = []
    for i in range(n_signals):
        x_true = X[i]
        S_true = set(int(j) for j in np.nonzero(S[i])[0])
        y = A @ x_true
        S_pred, x_hat = fn(A, y, k)
        denom = max(np.linalg.norm(x_true), 1e-8)
        nrmses.append(float(np.linalg.norm(x_hat - x_true) / denom))
        ious.append(iou(S_pred, S_true))
    return {
        "nrmse_mean": float(np.mean(nrmses)),
        "nrmse_std": float(np.std(nrmses)),
        "iou_mean": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    n, m, k = 256, 128, 25
    n_test = 500
    amp_lo, amp_hi = 0.5, 2.0
    test_seed = 42 + 999  # matches ista_comparison.py

    A_F, A_G = make_operators(n, m, seed=0)
    X_test, S_test = make_signals(n, k, n_test, amp_lo, amp_hi, seed=test_seed)

    print(f"n={n}, m={m}, k={k}, n_test={n_test}")
    print(f"mu(A_F) = {float(np.max(np.abs(A_F.T @ A_F - np.eye(n)))):.4f}")
    print(f"mu(A_G) = {float(np.max(np.abs(A_G.T @ A_G - np.eye(n)))):.4f}")

    selectors = [("CoSaMP", cosamp), ("HTP", htp)]

    fresh = {}
    for op_name, A in [("fourier", A_F), ("gaussian", A_G)]:
        fresh[op_name] = {}
        print(f"\n-- Running on {op_name.capitalize()} test set --")
        for sel_name, sel_fn in selectors:
            res = evaluate_method(sel_name, sel_fn, A, X_test, S_test, k)
            fresh[op_name][sel_name] = res
            print(f"  {sel_name:<8}  NRMSE = {res['nrmse_mean']:.4f} ± {res['nrmse_std']:.4f}   "
                  f"IoU = {res['iou_mean']:.4f} ± {res['iou_std']:.4f}")

    # ----- Compare to Table 1 -----
    with TABLE1_JSON.open() as f:
        table1 = json.load(f)
    methods = table1["methods"]

    print("\n" + "=" * 70)
    print(" Side-by-side with Table 1 (Fourier->Gaussian, n=256, m=128, k=25)")
    print("=" * 70)
    print(f"\n  {'method':<28}  {'Fourier NRMSE':>15}  {'Gaussian NRMSE':>16}")
    rows = [
        ("Raw ISTA",
            methods["raw_ista"]["fourier_nrmse"],
            methods["raw_ista"]["gaussian_nrmse"]),
        ("Naive top-k + LS",
            methods["naive_topk_ls"]["fourier_nrmse"],
            methods["naive_topk_ls"]["gaussian_nrmse"]),
        ("Learned detector + LS",
            methods["det_ls"]["fourier_nrmse"],
            methods["det_ls"]["gaussian_nrmse"]),
        ("LISTA, zero-shot",
            methods["lista_zero_shot"]["fourier_nrmse"],
            methods["lista_zero_shot"]["gaussian_nrmse"]),
        ("Oracle support + LS",
            methods["oracle_ls"]["fourier_nrmse"],
            methods["oracle_ls"]["gaussian_nrmse"]),
        ("CoSaMP  (NEW)",
            fresh["fourier"]["CoSaMP"]["nrmse_mean"],
            fresh["gaussian"]["CoSaMP"]["nrmse_mean"]),
        ("HTP     (NEW)",
            fresh["fourier"]["HTP"]["nrmse_mean"],
            fresh["gaussian"]["HTP"]["nrmse_mean"]),
    ]
    for name, fn, gn in rows:
        print(f"  {name:<28}  {fn:>15.4f}  {gn:>16.4f}")

    out = {
        "config": {"n": n, "m": m, "k": k, "n_test": n_test,
                   "amp_lo": amp_lo, "amp_hi": amp_hi, "test_seed": test_seed},
        "fresh_results": fresh,
        "table1_baselines": {
            name: {"fourier": fn, "gaussian": gn}
            for name, fn, gn in rows
        },
    }
    out_json = Path(__file__).resolve().parent / "cosamp_vs_table1.json"
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_json}")

    # ----- Verdict -----
    cosamp_g = fresh["gaussian"]["CoSaMP"]["nrmse_mean"]
    naive_g = methods["naive_topk_ls"]["gaussian_nrmse"]
    lista_g = methods["lista_zero_shot"]["gaussian_nrmse"]
    print("\n=== Verdict on Gaussian (zero-shot transfer target) ===")
    if cosamp_g <= 0.05:
        verdict = "EATS THE TABLE: CoSaMP achieves near-oracle NRMSE."
    elif cosamp_g < naive_g:
        verdict = "Wins by margin: CoSaMP beats current best (naive top-k + LS)."
    elif cosamp_g < lista_g:
        verdict = "Mixed: CoSaMP beats LISTA but not naive top-k + LS."
    else:
        verdict = "Saved: CoSaMP fails on this regime; learned methods retain headroom."
    print(f"  CoSaMP NRMSE                    : {cosamp_g:.4f}")
    print(f"  Naive top-k + LS NRMSE (Table 1): {naive_g:.4f}")
    print(f"  LISTA zero-shot NRMSE (Table 1) : {lista_g:.4f}")
    print(f"  CoSaMP vs naive top-k           : {(naive_g - cosamp_g) / naive_g * 100:+.1f}%")
    print(f"  CoSaMP vs LISTA zero-shot       : {(lista_g - cosamp_g) / lista_g * 100:+.1f}%")
    print(f"  -> {verdict}")

if __name__ == "__main__":
    main()
