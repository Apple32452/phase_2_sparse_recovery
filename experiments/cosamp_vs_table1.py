"""
cosamp_vs_table1.py

Compare CoSaMP and HTP against the Table-1-style strict sparse setting.

This version is cleaned for the reorganized repository:
  - Does not crash if the old phase_1 Table 1 JSON is missing.
  - Saves output to results/cosamp/cosamp_vs_table1.json.
  - Keeps the old Table 1 comparison if phase_1/results/ista_comparison_T30_lam0.05.json exists.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLE1_JSON = ROOT / "phase_1" / "results" / "ista_comparison_T30_lam0.05.json"
OUT_DIR = ROOT / "results" / "cosamp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Operators and signals
# ----------------------------------------------------------------------

def make_operators(n, m, seed=0):
    """Original phase-1-style operator construction."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, m // 2, replace=False)

    F_complex = np.fft.fft(np.eye(n)) / np.sqrt(n)
    rows = np.concatenate([F_complex[idx].real, F_complex[idx].imag], axis=0)
    A_F = rows[:m].astype(np.float32)

    rng2 = np.random.RandomState(seed + 1)
    A_G = (rng2.randn(m, n) / np.sqrt(m)).astype(np.float32)

    return A_F, A_G


def make_signals(n, k, n_signals, amp_lo, amp_hi, seed):
    rng = np.random.RandomState(seed)
    X = np.zeros((n_signals, n), dtype=np.float32)
    S = np.zeros((n_signals, n), dtype=np.float32)

    for i in range(n_signals):
        supp = rng.choice(n, k, replace=False)
        amps = rng.uniform(amp_lo, amp_hi, k) * rng.choice([-1, 1], k)
        X[i, supp] = amps
        S[i, supp] = 1.0

    return X, S


def mutual_coherence(A):
    col_norms = np.linalg.norm(A, axis=0, keepdims=True)
    A_unit = A / np.maximum(col_norms, 1e-12)
    G = A_unit.T @ A_unit
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))


# ----------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------

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
        omega = set(int(i) for i in np.argpartition(-np.abs(u), 2 * k - 1)[:2 * k])

        current_support = set(int(i) for i in np.nonzero(np.abs(x) > 0)[0])
        T = sorted(omega | current_support)

        b_T, *_ = np.linalg.lstsq(A[:, T], y, rcond=None)
        b = np.zeros(n)
        b[T] = b_T

        S_new = set(int(i) for i in np.argpartition(-np.abs(b), k - 1)[:k])
        S_list = sorted(S_new)

        x = np.zeros(n)
        x[S_list], *_ = np.linalg.lstsq(A[:, S_list], y, rcond=None)

        if S_new == S_prev and S_prev:
            break

        if np.isfinite(prev_res_norm):
            improvement = prev_res_norm - res_norm
            if improvement >= 0.0 and improvement <= tol * max(1.0, prev_res_norm):
                break

        prev_res_norm = res_norm
        S_prev = S_new

    return S_prev, x


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
# Metrics
# ----------------------------------------------------------------------

def iou(S_pred, S_true):
    inter = len(S_pred & S_true)
    union = len(S_pred | S_true)
    return inter / union if union > 0 else 0.0


def evaluate_method(fn, A, X, S, k):
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
    test_seed = 42 + 999

    A_F, A_G = make_operators(n, m, seed=0)
    X_test, S_test = make_signals(n, k, n_test, amp_lo, amp_hi, seed=test_seed)

    print(f"n={n}, m={m}, k={k}, n_test={n_test}")
    print(f"mu(A_F) = {mutual_coherence(A_F):.4f}")
    print(f"mu(A_G) = {mutual_coherence(A_G):.4f}")

    selectors = [("CoSaMP", cosamp), ("HTP", htp)]

    fresh = {}

    for op_name, A in [("fourier", A_F), ("gaussian", A_G)]:
        fresh[op_name] = {}
        print(f"\n-- Running on {op_name.capitalize()} test set --")

        for sel_name, sel_fn in selectors:
            res = evaluate_method(sel_fn, A, X_test, S_test, k)
            fresh[op_name][sel_name] = res

            print(
                f"  {sel_name:<8}  NRMSE = {res['nrmse_mean']:.4f} ± {res['nrmse_std']:.4f}   "
                f"IoU = {res['iou_mean']:.4f} ± {res['iou_std']:.4f}"
            )

    rows = [
        (
            "CoSaMP",
            fresh["fourier"]["CoSaMP"]["nrmse_mean"],
            fresh["gaussian"]["CoSaMP"]["nrmse_mean"],
        ),
        (
            "HTP",
            fresh["fourier"]["HTP"]["nrmse_mean"],
            fresh["gaussian"]["HTP"]["nrmse_mean"],
        ),
    ]

    table1 = None

    if TABLE1_JSON.exists():
        print(f"\nFound old Table 1 JSON: {TABLE1_JSON}")
        with TABLE1_JSON.open() as f:
            table1 = json.load(f)

        methods = table1["methods"]

        rows = [
            (
                "Raw ISTA",
                methods["raw_ista"]["fourier_nrmse"],
                methods["raw_ista"]["gaussian_nrmse"],
            ),
            (
                "Naive top-k + LS",
                methods["naive_topk_ls"]["fourier_nrmse"],
                methods["naive_topk_ls"]["gaussian_nrmse"],
            ),
            (
                "Learned detector + LS",
                methods["det_ls"]["fourier_nrmse"],
                methods["det_ls"]["gaussian_nrmse"],
            ),
            (
                "LISTA, zero-shot",
                methods["lista_zero_shot"]["fourier_nrmse"],
                methods["lista_zero_shot"]["gaussian_nrmse"],
            ),
            (
                "Oracle support + LS",
                methods["oracle_ls"]["fourier_nrmse"],
                methods["oracle_ls"]["gaussian_nrmse"],
            ),
        ] + rows
    else:
        print(f"\nWarning: old Table 1 JSON not found at:")
        print(f"  {TABLE1_JSON}")
        print("Skipping old Table 1 comparison and saving fresh CoSaMP/HTP results only.")

    print("\n" + "=" * 70)
    print(" Strict-sparse comparison, n=256, m=128, k=25")
    print("=" * 70)
    print(f"\n  {'method':<28}  {'Fourier NRMSE':>15}  {'Gaussian NRMSE':>16}")

    for name, fourier_nrmse, gaussian_nrmse in rows:
        print(f"  {name:<28}  {fourier_nrmse:>15.4f}  {gaussian_nrmse:>16.4f}")

    out = {
        "config": {
            "n": n,
            "m": m,
            "k": k,
            "n_test": n_test,
            "amp_lo": amp_lo,
            "amp_hi": amp_hi,
            "test_seed": test_seed,
            "table1_json": str(TABLE1_JSON),
            "table1_json_exists": TABLE1_JSON.exists(),
        },
        "fresh_results": fresh,
        "table1_comparison_rows": [
            {"method": name, "fourier_nrmse": fn, "gaussian_nrmse": gn}
            for name, fn, gn in rows
        ],
    }

    out_json = OUT_DIR / "cosamp_vs_table1.json"

    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote {out_json}")

    print("\n=== Verdict ===")
    cosamp_g = fresh["gaussian"]["CoSaMP"]["nrmse_mean"]

    if cosamp_g <= 0.05:
        print("CoSaMP achieves near-oracle recovery in this k=25 strict-sparse setting.")
        print("This confirms that below-transition strict-sparse recovery is not a good regime for claiming a learned advantage.")
    else:
        print("CoSaMP does not fully solve this setting; learned methods may retain headroom.")

    print(f"CoSaMP Gaussian NRMSE: {cosamp_g:.4f}")


if __name__ == "__main__":
    main()
