"""
Stress-test CoSaMP and the abstract's baselines across three perturbation
axes to find the regime in which classical refinement fails. If CoSaMP
breaks anywhere meaningful while the oracle still recovers nearly exactly,
that's the regime in which a learned operator-aware support detector has
genuine headroom.

Sweeps:
  1. Sparsity:           k in {15, 25, 40, 55, 70}            (m=128, noiseless)
  2. Measurement noise:  SNR in {inf, 30, 20, 15, 10, 5} dB   (k=25)
  3. Approximate spars.: tail_amp in {0, 0.02, 0.05, 0.1, 0.2} (k=25, noiseless)

Methods per cell: naive top-k + LS, OMP, CoSaMP, oracle.
Operators: partial Fourier and Gaussian (the abstract's two).
Signals per cell: 50.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Operators / signals (matches phase_1/ista_comparison.py construction)
# ----------------------------------------------------------------------

def make_operators(n, m, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, m // 2, replace=False)
    F = np.fft.fft(np.eye(n)) / np.sqrt(n)
    rows = np.concatenate([F[idx].real, F[idx].imag], axis=0)
    A_F = rows[:m].astype(np.float32)
    rng2 = np.random.RandomState(seed + 1)
    A_G = (rng2.randn(m, n) / np.sqrt(m)).astype(np.float32)
    return A_F, A_G

def gen_strict_signal(n, k, seed, amp_lo=0.5, amp_hi=2.0):
    rng = np.random.default_rng(seed)
    S = rng.choice(n, size=k, replace=False)
    x = np.zeros(n)
    x[S] = rng.uniform(amp_lo, amp_hi, k) * rng.choice([-1, 1], k)
    return x, set(int(s) for s in S)

def gen_compressible_signal(n, k, seed, tail_amp, amp_lo=0.5, amp_hi=2.0):
    """k large spikes plus a Gaussian tail on every other coordinate.
    True support = the k spikes; tail represents non-strict sparsity."""
    x, S_true = gen_strict_signal(n, k, seed, amp_lo, amp_hi)
    if tail_amp > 0:
        rng = np.random.default_rng(seed + 10_000)
        tail = tail_amp * rng.standard_normal(n)
        for j in S_true:
            tail[j] = 0.0
        x = x + tail
    return x, S_true

def add_measurement_noise(y, snr_db, seed):
    if not np.isfinite(snr_db):
        return y
    rng = np.random.default_rng(seed + 20_000)
    sig_pow = float(np.mean(y ** 2))
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    return y + np.sqrt(noise_pow) * rng.standard_normal(y.shape)

# ----------------------------------------------------------------------
# Selectors
# ----------------------------------------------------------------------

def naive_topk(A, y, k):
    scores = np.abs(A.T @ y)
    idx = np.argpartition(-scores, k - 1)[:k]
    return set(int(i) for i in idx)

def omp(A, y, k):
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
    return S_new

# ----------------------------------------------------------------------
# Eval helpers
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
    denom = max(np.linalg.norm(x_true), 1e-12)
    return float(np.linalg.norm(x_hat - x_true) / denom)

def evaluate_cell(A, signals, k):
    """signals is a list of (x_true, S_true, y) tuples."""
    out = {m: {"nrmse": [], "iou": []} for m in
           ["naive", "omp", "cosamp", "oracle"]}
    for x_true, S_true, y in signals:
        for name, fn in [("naive", naive_topk), ("omp", omp), ("cosamp", cosamp)]:
            S_pred = fn(A, y, k)
            out[name]["nrmse"].append(support_ls_nrmse(A, y, S_pred, x_true))
            out[name]["iou"].append(iou(S_pred, S_true))
        out["oracle"]["nrmse"].append(support_ls_nrmse(A, y, S_true, x_true))
        out["oracle"]["iou"].append(1.0)
    return {name: {"nrmse_mean": float(np.mean(d["nrmse"])),
                   "nrmse_std":  float(np.std(d["nrmse"])),
                   "iou_mean":   float(np.mean(d["iou"])),
                   "iou_std":    float(np.std(d["iou"]))}
            for name, d in out.items()}

# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------

def sweep_k(A_F, A_G, k_values, n, n_signals, seed_base):
    results = {}
    for k in k_values:
        signals = []
        for s in range(n_signals):
            x, S = gen_strict_signal(n, k, seed_base + s)
            signals.append((x, S, None))
        for op_name, A in [("fourier", A_F), ("gaussian", A_G)]:
            sigs_op = [(x, S, A @ x) for x, S, _ in signals]
            results.setdefault(op_name, {})[k] = evaluate_cell(A, sigs_op, k)
    return results

def sweep_noise(A_F, A_G, snr_values, n, k, n_signals, seed_base):
    results = {}
    base_signals = [gen_strict_signal(n, k, seed_base + s) for s in range(n_signals)]
    for snr in snr_values:
        snr_key = "inf" if not np.isfinite(snr) else int(snr)
        for op_name, A in [("fourier", A_F), ("gaussian", A_G)]:
            sigs_op = []
            for s, (x, S) in enumerate(base_signals):
                y = add_measurement_noise(A @ x, snr, seed_base + s)
                sigs_op.append((x, S, y))
            results.setdefault(op_name, {})[snr_key] = evaluate_cell(A, sigs_op, k)
    return results

def sweep_compressible(A_F, A_G, tail_amps, n, k, n_signals, seed_base):
    results = {}
    for tail in tail_amps:
        signals = []
        for s in range(n_signals):
            x, S = gen_compressible_signal(n, k, seed_base + s, tail_amp=tail)
            signals.append((x, S))
        for op_name, A in [("fourier", A_F), ("gaussian", A_G)]:
            sigs_op = [(x, S, A @ x) for x, S in signals]
            results.setdefault(op_name, {})[tail] = evaluate_cell(A, sigs_op, k)
    return results

# ----------------------------------------------------------------------
# Reporting / plotting
# ----------------------------------------------------------------------

def print_sweep(title, results, axis_label, axis_values):
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")
    for op_name in results:
        print(f"\n  [{op_name.upper()}]   {axis_label} -> NRMSE  (naive | omp | cosamp | oracle)")
        for v in axis_values:
            key = v if v in results[op_name] else (
                "inf" if not np.isfinite(v) else int(v) if isinstance(v, (int, float)) else v)
            r = results[op_name][key]
            print(f"    {axis_label}={str(v):<6} "
                  f"{r['naive']['nrmse_mean']:.3f}  | "
                  f"{r['omp']['nrmse_mean']:.3f}  | "
                  f"{r['cosamp']['nrmse_mean']:.3f}  | "
                  f"{r['oracle']['nrmse_mean']:.3f}")

def plot_sweeps(sweep_k_res, sweep_n_res, sweep_c_res, k_values,
                snr_values, tail_amps, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey="row")
    methods = ["naive", "omp", "cosamp", "oracle"]
    colors = {"naive": "#bdbdbd", "omp": "#d95f02",
              "cosamp": "#e7298a", "oracle": "#7570b3"}

    def lookup_axis(res, op, axis_vals, axis_kind):
        out = []
        for v in axis_vals:
            if axis_kind == "snr":
                key = "inf" if not np.isfinite(v) else int(v)
            else:
                key = v
            out.append(res[op][key])
        return out

    for col, (op_name, op_label) in enumerate(
            [("fourier", "Partial Fourier  (mu=0.75)"),
             ("gaussian", "Gaussian  (mu=0.37)")]):
        # column 0: k sweep, col 1: noise sweep, col 2: compressible sweep
        # but we have 3 sweeps for 2 operators -> 2 rows x 3 cols, with rows=op
        pass

    # Reorganize: rows = operators (Fourier, Gaussian), cols = sweeps (k, snr, tail)
    sweep_specs = [
        ("Sparsity sweep", sweep_k_res, k_values, "k", "k"),
        ("Noise sweep",   sweep_n_res, snr_values, "SNR (dB)", "snr"),
        ("Approx. sparsity sweep", sweep_c_res, tail_amps, "tail amplitude", "tail"),
    ]

    for row, (op_name, op_label) in enumerate(
            [("fourier", "Partial Fourier  (mu=0.75)"),
             ("gaussian", "Gaussian  (mu=0.37)")]):
        for col, (title, res, axis_vals, x_label, axis_kind) in enumerate(sweep_specs):
            ax = axes[row, col]
            for m in methods:
                cells = lookup_axis(res, op_name, axis_vals, axis_kind)
                ys = [c[m]["nrmse_mean"] for c in cells]
                if axis_kind == "snr":
                    xs = [40 if not np.isfinite(v) else v for v in axis_vals]  # cap inf at 40 dB
                    ax.set_xticks(xs)
                    ax.set_xticklabels(["inf" if not np.isfinite(v) else str(int(v))
                                        for v in axis_vals], fontsize=8)
                else:
                    xs = list(axis_vals)
                ax.plot(xs, ys, marker="o", color=colors[m], label=m, lw=1.6, ms=5)
            ax.set_title(f"{op_label} | {title}", fontsize=9)
            ax.set_xlabel(x_label, fontsize=9)
            if col == 0:
                ax.set_ylabel("NRMSE", fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_ylim(-0.02, 1.05)
            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(pad=0.6)
    fig.savefig(out_png, dpi=170)
    print(f"\nWrote {out_png}")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    n, m = 256, 128
    n_signals = 50
    seed_base = 5000

    A_F, A_G = make_operators(n, m, seed=0)

    k_values = [15, 25, 40, 55, 70]
    snr_values = [np.inf, 30, 20, 15, 10, 5]
    tail_amps = [0.0, 0.02, 0.05, 0.1, 0.2]

    print(f"n={n}, m={m}, n_signals={n_signals} per cell\n")

    print("Running sparsity sweep ...")
    res_k = sweep_k(A_F, A_G, k_values, n, n_signals, seed_base)
    print("Running noise sweep ...")
    res_n = sweep_noise(A_F, A_G, snr_values, n, k=25,
                        n_signals=n_signals, seed_base=seed_base + 1000)
    print("Running compressible-signal sweep ...")
    res_c = sweep_compressible(A_F, A_G, tail_amps, n, k=25,
                               n_signals=n_signals, seed_base=seed_base + 2000)

    print_sweep("Sparsity sweep (m=128, noiseless, strict-sparse)",
                res_k, "k", k_values)
    print_sweep("Noise sweep (m=128, k=25, strict-sparse)",
                res_n, "SNR", snr_values)
    print_sweep("Approx-sparsity sweep (m=128, k=25, noiseless)",
                res_c, "tail", tail_amps)

    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "cosamp_stress_test.png"
    out_json = out_dir / "cosamp_stress_test.json"

    plot_sweeps(res_k, res_n, res_c, k_values, snr_values, tail_amps, out_png)

    # Convert numpy keys (inf) to strings for JSON
    def jsonify(d):
        return {str(k): v for k, v in d.items()}
    payload = {
        "config": {"n": n, "m": m, "n_signals": n_signals,
                   "k_values": k_values,
                   "snr_values": ["inf" if not np.isfinite(v) else v for v in snr_values],
                   "tail_amps": tail_amps},
        "sweep_k": {op: jsonify(d) for op, d in res_k.items()},
        "sweep_noise": {op: jsonify(d) for op, d in res_n.items()},
        "sweep_compressible": {op: jsonify(d) for op, d in res_c.items()},
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote {out_json}")

    # ----- Verdict: where does CoSaMP fail? -----
    print("\n" + "=" * 78)
    print("  Cells where CoSaMP NRMSE > 0.10 AND oracle NRMSE < 0.05  (project lives here)")
    print("=" * 78)
    found_any = False
    for sweep_name, res, axis_vals, axis_label in [
        ("k", res_k, k_values, "k"),
        ("noise", res_n, snr_values, "SNR"),
        ("approx", res_c, tail_amps, "tail"),
    ]:
        for op_name in res:
            for v in axis_vals:
                key = (v if axis_label != "SNR" else
                       ("inf" if not np.isfinite(v) else int(v)))
                cell = res[op_name][key]
                cs = cell["cosamp"]["nrmse_mean"]
                orc = cell["oracle"]["nrmse_mean"]
                if cs > 0.10 and orc < 0.05:
                    found_any = True
                    print(f"  [{sweep_name:>6}]  {op_name:<8} {axis_label}={str(v):<5}  "
                          f"CoSaMP={cs:.3f}   oracle={orc:.3f}   gap={cs - orc:+.3f}")
    if not found_any:
        print("  (none — CoSaMP holds across every tested cell)")

if __name__ == "__main__":
    main()
