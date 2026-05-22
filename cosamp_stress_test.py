"""
cosamp_stress_test.py

Fixed/reproducible CoSaMP stress test for the sparse-recovery paper.

Main fixes versus the original script
-------------------------------------
1. Compressible-signal target is now consistent:
   - For strict sparse/noisy signals, the target support is the planted support.
   - For compressible signals, the target/oracle support is TopK(|x|), i.e.
     the best k-term support of the full signal after the tail is added.
   This matches the learned_compressible.py labeling convention.

2. Operator normalization is explicit and consistent:
   - Partial Fourier and Gaussian operators are both column-normalized by default.
   - Mutual coherence is computed from the normalized operators and printed/saved.
   - Plot titles use the measured coherence instead of hard-coded values.

3. Evaluation is more diagnostic:
   - NRMSE, IoU, precision, recall, missed target energy, false selected energy,
     and support condition number are reported.
   - The oracle is support-LS on the target support.

4. Reproducibility:
   - Uses argparse for seeds/configuration.
   - Saves config, operator diagnostics, and all sweep results to JSON.

Usage
-----
    python cosamp_stress_test.py

Optional:
    python cosamp_stress_test.py --n-signals 500 --seed-base 5000 --out-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoSaMP stress tests for sparse recovery.")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n-signals", type=int, default=50)
    parser.add_argument("--operator-seed", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--normalize-columns", action="store_true", default=True)
    parser.add_argument("--no-normalize-columns", dest="normalize_columns", action="store_false")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--out-prefix", type=str, default="cosamp_stress_test")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def normalize_columns(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return A with unit-norm columns."""
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    return A / np.maximum(norms, eps)


def best_k_support(x: np.ndarray, k: int) -> set[int]:
    """Indices of the k largest-magnitude entries of x."""
    if k <= 0:
        return set()
    k_eff = min(k, x.size)
    idx = np.argpartition(-np.abs(x), k_eff - 1)[:k_eff]
    return set(int(i) for i in idx)


def mutual_coherence(A: np.ndarray) -> float:
    """Maximum absolute off-diagonal Gram entry after column normalization."""
    A_unit = normalize_columns(A)
    G = A_unit.T @ A_unit
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))


def json_key(v):
    if isinstance(v, float) and not np.isfinite(v):
        return "inf"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return f"{float(v):g}"
    return str(v)


def safe_lstsq(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Numerically safe least-squares wrapper."""
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return x


# ----------------------------------------------------------------------
# Operators / signals
# ----------------------------------------------------------------------

def make_partial_fourier(n: int, m: int, seed: int) -> np.ndarray:
    """
    Real-valued partial Fourier operator.

    We sample m//2 complex Fourier rows and split real/imaginary parts,
    giving m real rows.
    """
    if m % 2 != 0:
        raise ValueError("m must be even for the real/imag partial Fourier construction.")

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, m // 2, replace=False)

    F = np.fft.fft(np.eye(n)) / np.sqrt(n)
    rows = np.concatenate([F[idx].real, F[idx].imag], axis=0)
    return rows[:m].astype(np.float32)


def make_gaussian(n: int, m: int, seed: int) -> np.ndarray:
    """Gaussian sensing operator."""
    rng = np.random.RandomState(seed)
    return rng.randn(m, n).astype(np.float32)


def make_operators(n: int, m: int, seed: int = 0, normalize: bool = True) -> Dict[str, np.ndarray]:
    """Construct the two sensing operators used in the stress test."""
    A_F = make_partial_fourier(n, m, seed=seed)
    A_G = make_gaussian(n, m, seed=seed + 1)

    if normalize:
        A_F = normalize_columns(A_F).astype(np.float32)
        A_G = normalize_columns(A_G).astype(np.float32)

    return {"fourier": A_F, "gaussian": A_G}


def gen_strict_signal(
    n: int,
    k: int,
    seed: int,
    amp_lo: float = 0.5,
    amp_hi: float = 2.0,
) -> Tuple[np.ndarray, set[int]]:
    """Generate an exactly k-sparse signal and its planted support."""
    rng = np.random.default_rng(seed)
    S = rng.choice(n, size=k, replace=False)
    x = np.zeros(n, dtype=np.float64)
    x[S] = rng.uniform(amp_lo, amp_hi, k) * rng.choice([-1, 1], k)
    return x, set(int(s) for s in S)


def gen_compressible_signal(
    n: int,
    k: int,
    seed: int,
    tail_amp: float,
    amp_lo: float = 0.5,
    amp_hi: float = 2.0,
) -> Tuple[np.ndarray, set[int], set[int]]:
    """
    Generate a compressible signal.

    Returns
    -------
    x:
        Full compressible signal.
    planted_support:
        Original k large spikes.
    target_support:
        Best k-term support TopK(|x|). This is the correct oracle target
        for compressible-signal NRMSE and matches the learned detector labels.
    """
    x, planted_support = gen_strict_signal(n, k, seed, amp_lo, amp_hi)

    if tail_amp > 0:
        rng = np.random.default_rng(seed + 10_000)
        tail = tail_amp * rng.standard_normal(n)
        for j in planted_support:
            tail[j] = 0.0
        x = x + tail

    target_support = best_k_support(x, k)
    return x, planted_support, target_support


def add_measurement_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add white Gaussian measurement noise at the requested SNR."""
    if not np.isfinite(snr_db):
        return y.copy()

    rng = np.random.default_rng(seed + 20_000)
    signal_power = float(np.mean(y ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return y + np.sqrt(noise_power) * rng.standard_normal(y.shape)


# ----------------------------------------------------------------------
# Classical support selectors
# ----------------------------------------------------------------------

def naive_topk(A: np.ndarray, y: np.ndarray, k: int) -> set[int]:
    """Top-k indices by absolute correlation |A^T y|."""
    scores = np.abs(A.T @ y)
    return best_k_support(scores, k)


def omp(A: np.ndarray, y: np.ndarray, k: int) -> set[int]:
    """Orthogonal Matching Pursuit with exactly k selections."""
    residual = y.copy()
    selected: List[int] = []

    for _ in range(k):
        scores = np.abs(A.T @ residual)
        if selected:
            scores[selected] = -np.inf

        j = int(np.argmax(scores))
        selected.append(j)

        A_S = A[:, selected]
        x_S = safe_lstsq(A_S, y)
        residual = y - A_S @ x_S

    return set(selected)


def cosamp(A: np.ndarray, y: np.ndarray, k: int, max_iters: int = 30, tol: float = 1e-10) -> set[int]:
    """
    CoSaMP support selection.

    Returns the final support of size k. Amplitudes are evaluated separately
    by support-restricted least squares for all methods.
    """
    n = A.shape[1]
    if 2 * k > n:
        raise ValueError(f"CoSaMP requires 2k <= n for this implementation, got k={k}, n={n}.")

    x = np.zeros(n, dtype=np.float64)
    prev_res_norm = np.inf
    S_prev: set[int] = set()

    for _ in range(max_iters):
        residual = y - A @ x
        res_norm = float(np.linalg.norm(residual))

        # Early stopping if the residual is already essentially zero.
        if res_norm < tol:
            break

        proxy = A.T @ residual
        omega = best_k_support(proxy, 2 * k)

        current_support = set(int(i) for i in np.nonzero(np.abs(x) > 0)[0])
        T = sorted(omega | current_support)

        A_T = A[:, T]
        b_T = safe_lstsq(A_T, y)

        b = np.zeros(n, dtype=np.float64)
        b[T] = b_T

        S_new = best_k_support(b, k)
        S_list = sorted(S_new)

        x = np.zeros(n, dtype=np.float64)
        x[S_list] = safe_lstsq(A[:, S_list], y)

        if S_new == S_prev:
            break
        if abs(prev_res_norm - res_norm) <= tol * max(1.0, prev_res_norm):
            break

        prev_res_norm = res_norm
        S_prev = S_new

    # If all iterations fail to select anything, fall back to proxy top-k.
    if not S_prev:
        return naive_topk(A, y, k)
    return S_prev


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def support_ls_reconstruction(A: np.ndarray, y: np.ndarray, support: Iterable[int], n: int) -> np.ndarray:
    """Least-squares reconstruction restricted to a support."""
    support_list = sorted(int(i) for i in support)
    x_hat = np.zeros(n, dtype=np.float64)

    if len(support_list) == 0:
        return x_hat

    A_S = A[:, support_list]
    x_S = safe_lstsq(A_S, y)
    x_hat[support_list] = x_S
    return x_hat


def support_ls_nrmse(A: np.ndarray, y: np.ndarray, support: Iterable[int], x_true: np.ndarray) -> float:
    """NRMSE after least-squares amplitude recovery on support."""
    x_hat = support_ls_reconstruction(A, y, support, x_true.size)
    denom = max(float(np.linalg.norm(x_true)), 1e-12)
    return float(np.linalg.norm(x_hat - x_true) / denom)


def support_iou(S_pred: set[int], S_target: set[int]) -> float:
    union = len(S_pred | S_target)
    if union == 0:
        return 1.0
    return len(S_pred & S_target) / union


def precision_recall(S_pred: set[int], S_target: set[int]) -> Tuple[float, float]:
    if len(S_pred) == 0:
        precision = 1.0 if len(S_target) == 0 else 0.0
    else:
        precision = len(S_pred & S_target) / len(S_pred)

    if len(S_target) == 0:
        recall = 1.0
    else:
        recall = len(S_pred & S_target) / len(S_target)

    return precision, recall


def support_condition_number(A: np.ndarray, support: Iterable[int]) -> float:
    support_list = sorted(int(i) for i in support)
    if len(support_list) == 0:
        return float("nan")

    A_S = A[:, support_list]
    try:
        return float(np.linalg.cond(A_S))
    except np.linalg.LinAlgError:
        return float("inf")


def support_energy_diagnostics(x_true: np.ndarray, S_pred: set[int], S_target: set[int]) -> Tuple[float, float]:
    """
    Energy diagnostics relative to the target support.

    missed_energy:
        Fraction of signal energy in target coordinates that were not selected.
    false_energy:
        Fraction of selected-support energy coming from coordinates outside target.
    """
    denom = max(float(np.linalg.norm(x_true) ** 2), 1e-12)

    missed = list(S_target - S_pred)
    false = list(S_pred - S_target)

    missed_energy = float(np.sum(x_true[missed] ** 2) / denom) if missed else 0.0
    false_energy = float(np.sum(x_true[false] ** 2) / denom) if false else 0.0

    return missed_energy, false_energy


def evaluate_cell(
    A: np.ndarray,
    signals: List[Tuple[np.ndarray, set[int], np.ndarray]],
    k: int,
    max_iters: int,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all methods on a cell.

    signals contains tuples:
        (x_true, target_support, y)
    """
    method_fns: Dict[str, Callable[[np.ndarray, np.ndarray, int], set[int]]] = {
        "naive": naive_topk,
        "omp": omp,
        "cosamp": lambda AA, yy, kk: cosamp(AA, yy, kk, max_iters=max_iters),
    }

    raw: Dict[str, Dict[str, List[float]]] = {
        name: {
            "nrmse": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "missed_energy": [],
            "false_energy": [],
            "cond": [],
        }
        for name in ["naive", "omp", "cosamp", "oracle"]
    }

    for x_true, S_target, y in signals:
        predicted_supports: Dict[str, set[int]] = {}

        for name, fn in method_fns.items():
            predicted_supports[name] = fn(A, y, k)

        # Oracle is least squares on the target support.
        predicted_supports["oracle"] = set(S_target)

        for name, S_pred in predicted_supports.items():
            nrmse = support_ls_nrmse(A, y, S_pred, x_true)
            iou = support_iou(S_pred, S_target)
            precision, recall = precision_recall(S_pred, S_target)
            missed_energy, false_energy = support_energy_diagnostics(x_true, S_pred, S_target)
            cond = support_condition_number(A, S_pred)

            raw[name]["nrmse"].append(nrmse)
            raw[name]["iou"].append(iou)
            raw[name]["precision"].append(precision)
            raw[name]["recall"].append(recall)
            raw[name]["missed_energy"].append(missed_energy)
            raw[name]["false_energy"].append(false_energy)
            raw[name]["cond"].append(cond)

    summary: Dict[str, Dict[str, float]] = {}
    for name, metrics in raw.items():
        summary[name] = {}
        for metric_name, values in metrics.items():
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                summary[name][f"{metric_name}_mean"] = float("nan")
                summary[name][f"{metric_name}_std"] = float("nan")
                summary[name][f"{metric_name}_median"] = float("nan")
            else:
                summary[name][f"{metric_name}_mean"] = float(np.mean(finite))
                summary[name][f"{metric_name}_std"] = float(np.std(finite))
                summary[name][f"{metric_name}_median"] = float(np.median(finite))

    return summary


# ----------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------

def sweep_k(
    operators: Dict[str, np.ndarray],
    k_values: List[int],
    n: int,
    n_signals: int,
    seed_base: int,
    max_iters: int,
) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
    results: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {name: {} for name in operators}

    for k in k_values:
        base_signals = []
        for s in range(n_signals):
            x, S_planted = gen_strict_signal(n, k, seed_base + s)
            # In strict sparse signals, planted support = best-k support.
            base_signals.append((x, S_planted))

        for op_name, A in operators.items():
            signals_op = [(x, S_target, A @ x) for x, S_target in base_signals]
            results[op_name][k] = evaluate_cell(A, signals_op, k, max_iters=max_iters)

    return results


def sweep_noise(
    operators: Dict[str, np.ndarray],
    snr_values: List[float],
    n: int,
    k: int,
    n_signals: int,
    seed_base: int,
    max_iters: int,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {name: {} for name in operators}

    base_signals = [gen_strict_signal(n, k, seed_base + s) for s in range(n_signals)]

    for snr in snr_values:
        snr_key = json_key(snr)
        for op_name, A in operators.items():
            signals_op = []
            for s, (x, S_target) in enumerate(base_signals):
                y_clean = A @ x
                y = add_measurement_noise(y_clean, snr, seed_base + s)
                signals_op.append((x, S_target, y))
            results[op_name][snr_key] = evaluate_cell(A, signals_op, k, max_iters=max_iters)

    return results


def sweep_compressible(
    operators: Dict[str, np.ndarray],
    tail_amps: List[float],
    n: int,
    k: int,
    n_signals: int,
    seed_base: int,
    max_iters: int,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {name: {} for name in operators}

    for tail_amp in tail_amps:
        tail_key = json_key(tail_amp)

        base_signals = []
        for s in range(n_signals):
            x, _S_planted, S_target = gen_compressible_signal(
                n=n,
                k=k,
                seed=seed_base + s,
                tail_amp=tail_amp,
            )
            # IMPORTANT: target is best-k support of the full compressible signal.
            base_signals.append((x, S_target))

        for op_name, A in operators.items():
            signals_op = [(x, S_target, A @ x) for x, S_target in base_signals]
            results[op_name][tail_key] = evaluate_cell(A, signals_op, k, max_iters=max_iters)

    return results


# ----------------------------------------------------------------------
# Reporting / plotting
# ----------------------------------------------------------------------

def print_operator_diagnostics(operators: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    print("\nOperator diagnostics")
    print("-" * 78)
    diagnostics = {}

    for name, A in operators.items():
        col_norms = np.linalg.norm(A, axis=0)
        mu = mutual_coherence(A)
        spec = float(np.linalg.norm(A, 2))
        diagnostics[name] = {
            "mu": mu,
            "spectral_norm": spec,
            "col_norm_min": float(np.min(col_norms)),
            "col_norm_max": float(np.max(col_norms)),
            "col_norm_mean": float(np.mean(col_norms)),
        }
        print(
            f"  {name:<8} mu={mu:.4f}  ||A||2={spec:.4f}  "
            f"col_norm=[{col_norms.min():.4f}, {col_norms.max():.4f}]"
        )

    return diagnostics


def print_sweep(
    title: str,
    results: Dict,
    axis_label: str,
    axis_values: Iterable,
) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")

    for op_name in results:
        print(f"\n  [{op_name.upper()}]   {axis_label} -> NRMSE  (naive | omp | cosamp | oracle)")
        for v in axis_values:
            key = v if v in results[op_name] else json_key(v)
            r = results[op_name][key]
            print(
                f"    {axis_label}={str(v):<8} "
                f"{r['naive']['nrmse_mean']:.3f}  | "
                f"{r['omp']['nrmse_mean']:.3f}  | "
                f"{r['cosamp']['nrmse_mean']:.3f}  | "
                f"{r['oracle']['nrmse_mean']:.3f}"
            )


def plot_sweeps(
    sweep_k_res: Dict,
    sweep_n_res: Dict,
    sweep_c_res: Dict,
    k_values: List[int],
    snr_values: List[float],
    tail_amps: List[float],
    op_diagnostics: Dict[str, Dict[str, float]],
    out_png: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey="row")

    methods = ["naive", "omp", "cosamp", "oracle"]
    colors = {
        "naive": "#bdbdbd",
        "omp": "#d95f02",
        "cosamp": "#e7298a",
        "oracle": "#7570b3",
    }

    sweep_specs = [
        ("Sparsity sweep", sweep_k_res, k_values, "k", "k"),
        ("Noise sweep", sweep_n_res, snr_values, "SNR (dB)", "snr"),
        ("Approx. sparsity sweep", sweep_c_res, tail_amps, "tail amplitude", "tail"),
    ]

    row_specs = [("fourier", "Partial Fourier"), ("gaussian", "Gaussian")]

    for row, (op_name, op_label) in enumerate(row_specs):
        mu = op_diagnostics[op_name]["mu"]

        for col, (title, res, axis_vals, x_label, axis_kind) in enumerate(sweep_specs):
            ax = axes[row, col]

            if axis_kind == "snr":
                xs = [40 if not np.isfinite(v) else float(v) for v in axis_vals]
                ax.set_xticks(xs)
                ax.set_xticklabels(["inf" if not np.isfinite(v) else str(int(v)) for v in axis_vals], fontsize=8)
            else:
                xs = [float(v) for v in axis_vals]

            for method in methods:
                ys = []
                for v in axis_vals:
                    key = json_key(v) if axis_kind in {"snr", "tail"} else v
                    ys.append(res[op_name][key][method]["nrmse_mean"])
                ax.plot(xs, ys, marker="o", color=colors[method], label=method, lw=1.6, ms=5)

            ax.set_title(f"{op_label} (mu={mu:.2f}) | {title}", fontsize=9)
            ax.set_xlabel(x_label, fontsize=9)
            if col == 0:
                ax.set_ylabel("NRMSE", fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_ylim(-0.02, 1.10)

            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(pad=0.6)
    fig.savefig(out_png, dpi=dpi)
    print(f"\nWrote {out_png}")


def save_json(payload: Dict, out_json: Path) -> None:
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    print(f"Wrote {out_json}")


def print_failure_cells(
    res_k: Dict,
    res_n: Dict,
    res_c: Dict,
    k_values: List[int],
    snr_values: List[float],
    tail_amps: List[float],
) -> None:
    print("\n" + "=" * 78)
    print("  Cells where CoSaMP NRMSE > 0.10 AND oracle NRMSE < 0.05")
    print("  These are support-identification failure regimes.")
    print("=" * 78)

    found_any = False

    sweeps = [
        ("k", res_k, k_values, "k"),
        ("noise", res_n, snr_values, "SNR"),
        ("approx", res_c, tail_amps, "tail"),
    ]

    for sweep_name, res, axis_vals, axis_label in sweeps:
        for op_name in res:
            for v in axis_vals:
                key = v if axis_label == "k" else json_key(v)
                cell = res[op_name][key]
                cs = cell["cosamp"]["nrmse_mean"]
                oracle = cell["oracle"]["nrmse_mean"]

                if cs > 0.10 and oracle < 0.05:
                    found_any = True
                    print(
                        f"  [{sweep_name:>6}]  {op_name:<8} {axis_label}={str(v):<8}  "
                        f"CoSaMP={cs:.3f}   oracle={oracle:.3f}   gap={cs - oracle:+.3f}"
                    )

    if not found_any:
        print("  (none — CoSaMP holds across every tested cell)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    n = args.n
    m = args.m
    n_signals = args.n_signals
    seed_base = args.seed_base

    k_values = [15, 25, 40, 55, 70]
    snr_values = [np.inf, 30, 20, 15, 10, 5]
    tail_amps = [0.0, 0.02, 0.05, 0.1, 0.2]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{args.out_prefix}.png"
    out_json = out_dir / f"{args.out_prefix}.json"

    print(f"n={n}, m={m}, n_signals={n_signals} per cell")
    print(f"operator_seed={args.operator_seed}, seed_base={seed_base}")
    print(f"column_normalization={args.normalize_columns}")

    operators = make_operators(
        n=n,
        m=m,
        seed=args.operator_seed,
        normalize=args.normalize_columns,
    )
    op_diagnostics = print_operator_diagnostics(operators)

    print("\nRunning sparsity sweep ...")
    res_k = sweep_k(
        operators=operators,
        k_values=k_values,
        n=n,
        n_signals=n_signals,
        seed_base=seed_base,
        max_iters=args.max_iters,
    )

    print("Running noise sweep ...")
    res_n = sweep_noise(
        operators=operators,
        snr_values=snr_values,
        n=n,
        k=25,
        n_signals=n_signals,
        seed_base=seed_base + 1000,
        max_iters=args.max_iters,
    )

    print("Running compressible-signal sweep ...")
    res_c = sweep_compressible(
        operators=operators,
        tail_amps=tail_amps,
        n=n,
        k=25,
        n_signals=n_signals,
        seed_base=seed_base + 2000,
        max_iters=args.max_iters,
    )

    print_sweep(
        "Sparsity sweep (m=128, noiseless, strict-sparse)",
        res_k,
        "k",
        k_values,
    )

    print_sweep(
        "Noise sweep (m=128, k=25, strict-sparse)",
        res_n,
        "SNR",
        snr_values,
    )

    print_sweep(
        "Approx-sparsity sweep (m=128, k=25, noiseless; target=TopK(|x|))",
        res_c,
        "tail",
        tail_amps,
    )

    plot_sweeps(
        sweep_k_res=res_k,
        sweep_n_res=res_n,
        sweep_c_res=res_c,
        k_values=k_values,
        snr_values=snr_values,
        tail_amps=tail_amps,
        op_diagnostics=op_diagnostics,
        out_png=out_png,
        dpi=args.dpi,
    )

    payload = {
        "config": {
            "n": n,
            "m": m,
            "n_signals": n_signals,
            "operator_seed": args.operator_seed,
            "seed_base": seed_base,
            "max_iters": args.max_iters,
            "normalize_columns": args.normalize_columns,
            "k_values": k_values,
            "snr_values": ["inf" if not np.isfinite(v) else v for v in snr_values],
            "tail_amps": tail_amps,
            "compressible_target": "TopK(abs(x_full))",
            "strict_sparse_target": "planted_support",
        },
        "operator_diagnostics": op_diagnostics,
        "sweep_k": {op: {str(k): v for k, v in d.items()} for op, d in res_k.items()},
        "sweep_noise": res_n,
        "sweep_compressible": res_c,
    }

    save_json(payload, out_json)

    print_failure_cells(
        res_k=res_k,
        res_n=res_n,
        res_c=res_c,
        k_values=k_values,
        snr_values=snr_values,
        tail_amps=tail_amps,
    )


if __name__ == "__main__":
    main()
