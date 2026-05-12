"""
Secure the compressible-regime win against single-shot artifacts.

Runs three batches:
  T1  Multi-seed robustness:   5 operator seeds x 3 init seeds = 15 runs,
                              training mode = extended Unif[0.1, 0.4]
                              (the variant that produced the headline win).
  T2  Wider training:          1 run, training mode = Unif[0, 0.4],
                              tests whether the win survives when the
                              detector also covers the strict-sparse
                              region we previously excluded.
  T3  Bootstrap CIs:           Computed offline from per-signal NRMSE
                              arrays already saved by each run.

Each individual run is a subprocess call to learned_compressible.py with
--save-per-signal so we can do paired analyses afterwards.

Outputs:
  - secure_compressible_summary.json   aggregate stats per (batch, tail)
  - prints a verdict table to stdout
"""

import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "learned_compressible.py"

# T1 sweep grid
OP_SEEDS = [0, 1, 2, 3, 4]
INIT_SEEDS = [42, 123, 7]

EVAL_TAILS = [0.1, 0.2, 0.3, 0.4]

def run_one(mode, tag, op_seed, init_seed, train_lo=None, train_hi=None,
            tail_eval=None):
    """Invoke learned_compressible.py once; return parsed JSON dict."""
    out_json = HERE / f"learned_compressible_{tag}.json"
    cmd = [sys.executable, str(SCRIPT),
           "--mode", mode,
           "--tag", tag,
           "--op-seed", str(op_seed),
           "--init-seed", str(init_seed),
           "--save-per-signal"]
    if train_lo is not None:
        cmd += ["--tail-train-lo", str(train_lo)]
    if train_hi is not None:
        cmd += ["--tail-train-hi", str(train_hi)]
    if tail_eval is not None:
        cmd += ["--tail-eval"] + [str(t) for t in tail_eval]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  FAILED tag={tag}  op={op_seed}  init={init_seed}")
        print(proc.stdout[-1000:])
        print(proc.stderr[-1000:])
        raise RuntimeError(f"run failed: {tag}")
    with out_json.open() as f:
        data = json.load(f)
    return data, dt

def per_signal_delta(run_data, tail):
    """CoSaMP - learned NRMSE per test signal at given tail."""
    r = run_data["results"][str(tail)]
    cs = np.array(r["cosamp"]["nrmse_per_signal"])
    ln = np.array(r["learned"]["nrmse_per_signal"])
    return cs - ln

def paired_bootstrap_ci(deltas, n_boot=2000, alpha=0.05, rng_seed=0):
    """Paired bootstrap 95% CI on the mean of deltas."""
    rng = np.random.default_rng(rng_seed)
    n = len(deltas)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = float(np.mean(deltas[idx]))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi

# ----------------------------------------------------------------------
# T1: multi-seed robustness on Unif[0.1, 0.4]
# ----------------------------------------------------------------------

print("=" * 78)
print(f"  T1: multi-seed robustness sweep "
      f"({len(OP_SEEDS)} op-seeds x {len(INIT_SEEDS)} init-seeds = "
      f"{len(OP_SEEDS) * len(INIT_SEEDS)} runs)")
print("=" * 78)

t1_runs = []
for op_seed, init_seed in product(OP_SEEDS, INIT_SEEDS):
    tag = f"t1_op{op_seed}_init{init_seed}"
    print(f"  running tag={tag} ...", end=" ", flush=True)
    data, dt = run_one("extended", tag, op_seed, init_seed,
                       train_lo=0.1, train_hi=0.4, tail_eval=EVAL_TAILS)
    print(f"done in {dt:.1f}s")
    t1_runs.append({"op_seed": op_seed, "init_seed": init_seed, "data": data})

# Per-tail aggregate: mean / std of delta across runs; pooled per-signal CI
t1_summary = {}
print("\n" + "-" * 78)
print(f"  {'tail':>6} {'CoSaMP':>10} {'learned':>10} {'delta':>10} "
      f"{'std(runs)':>10} {'CI_lo':>10} {'CI_hi':>10} {'wins/15':>10}")
print("-" * 78)
for tail in EVAL_TAILS:
    cs_means = []
    ln_means = []
    deltas_per_run = []
    pooled_deltas = []
    n_wins = 0
    for r in t1_runs:
        d = r["data"]["results"][str(tail)]
        cs_means.append(d["cosamp"]["nrmse_mean"])
        ln_means.append(d["learned"]["nrmse_mean"])
        deltas = per_signal_delta(r["data"], tail)
        deltas_per_run.append(float(np.mean(deltas)))
        pooled_deltas.extend(deltas.tolist())
        if np.mean(deltas) > 0:
            n_wins += 1
    pooled_deltas = np.array(pooled_deltas)
    ci_lo, ci_hi = paired_bootstrap_ci(pooled_deltas)
    t1_summary[tail] = {
        "cosamp_mean":  float(np.mean(cs_means)),
        "learned_mean": float(np.mean(ln_means)),
        "delta_mean_runs": float(np.mean(deltas_per_run)),
        "delta_std_runs":  float(np.std(deltas_per_run)),
        "delta_min_runs":  float(np.min(deltas_per_run)),
        "delta_max_runs":  float(np.max(deltas_per_run)),
        "delta_per_run":   deltas_per_run,
        "ci95_lo":         ci_lo,
        "ci95_hi":         ci_hi,
        "n_wins":          n_wins,
        "n_runs":          len(t1_runs),
    }
    print(f"  {tail:>6.2f} "
          f"{np.mean(cs_means):>10.4f} {np.mean(ln_means):>10.4f} "
          f"{np.mean(deltas_per_run):>+10.4f} {np.std(deltas_per_run):>10.4f} "
          f"{ci_lo:>+10.4f} {ci_hi:>+10.4f}  {n_wins:>5}/{len(t1_runs)}")

# ----------------------------------------------------------------------
# T2: wider training distribution Unif[0, 0.4]
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("  T2: wider training distribution Unif[0, 0.4]")
print("=" * 78)
print("  (single op_seed=0, init_seed=42; tests whether high-tail win "
      "survives when training also covers strict-sparse)")

T2_TAILS = [0.0, 0.1, 0.2, 0.3, 0.4]
data_t2, dt_t2 = run_one("extended", "t2_wide_0_04", op_seed=0, init_seed=42,
                          train_lo=0.0, train_hi=0.4, tail_eval=T2_TAILS)
print(f"  done in {dt_t2:.1f}s")
print()
print(f"  {'tail':>6} {'CoSaMP':>10} {'learned':>10} {'delta':>10} "
      f"{'CI_lo':>10} {'CI_hi':>10}")
print("-" * 78)
t2_summary = {}
for tail in T2_TAILS:
    d = data_t2["results"][str(tail)]
    cs = d["cosamp"]["nrmse_mean"]
    ln = d["learned"]["nrmse_mean"]
    deltas = per_signal_delta(data_t2, tail)
    delta = float(np.mean(deltas))
    ci_lo, ci_hi = paired_bootstrap_ci(deltas)
    t2_summary[tail] = {
        "cosamp_mean": cs, "learned_mean": ln,
        "delta": delta, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
    }
    print(f"  {tail:>6.2f} {cs:>10.4f} {ln:>10.4f} {delta:>+10.4f} "
          f"{ci_lo:>+10.4f} {ci_hi:>+10.4f}")

# ----------------------------------------------------------------------
# Save aggregate
# ----------------------------------------------------------------------

out = HERE / "secure_compressible_summary.json"
with out.open("w") as f:
    json.dump({
        "T1": {"op_seeds": OP_SEEDS, "init_seeds": INIT_SEEDS,
               "train_mode": "Unif[0.1, 0.4]",
               "per_tail": {str(t): v for t, v in t1_summary.items()}},
        "T2": {"train_mode": "Unif[0, 0.4]",
               "op_seed": 0, "init_seed": 42,
               "per_tail": {str(t): v for t, v in t2_summary.items()}},
    }, f, indent=2)
print(f"\nWrote {out}")

# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("  VERDICT")
print("=" * 78)
print("\n  T1 (Unif[0.1, 0.4], 15 runs):")
for tail in EVAL_TAILS:
    s = t1_summary[tail]
    sign = "WIN" if s["delta_mean_runs"] > 0.01 else (
        "TIE" if abs(s["delta_mean_runs"]) <= 0.01 else "LOSS")
    robust = "robust" if s["n_wins"] >= max(1, int(0.8 * s["n_runs"])) else (
        "mixed" if s["n_wins"] >= s["n_runs"] / 2 else "fragile")
    ci_excludes_zero = s["ci95_lo"] > 0 or s["ci95_hi"] < 0
    sig = "stat-sig" if ci_excludes_zero else "not stat-sig"
    print(f"    tail={tail}  delta={s['delta_mean_runs']:+.4f} "
          f"+/- {s['delta_std_runs']:.4f}  "
          f"wins={s['n_wins']}/{s['n_runs']} ({robust})  "
          f"CI95=[{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}] ({sig})  "
          f"-> {sign}")

print("\n  T2 (Unif[0, 0.4], 1 run):")
for tail in T2_TAILS:
    s = t2_summary[tail]
    sign = "WIN" if s["delta"] > 0.01 else (
        "TIE" if abs(s["delta"]) <= 0.01 else "LOSS")
    sig = "stat-sig" if (s["ci95_lo"] > 0 or s["ci95_hi"] < 0) else "not stat-sig"
    print(f"    tail={tail}  delta={s['delta']:+.4f}  "
          f"CI95=[{s['ci95_lo']:+.4f}, {s['ci95_hi']:+.4f}] ({sig})  "
          f"-> {sign}")
