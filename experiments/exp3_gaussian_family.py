"""
exp3_gaussian_family.py
=======================
Within-Family Gaussian Transfer — Variable Measurement Count

Primary research question (Section 9 of research_strategy_summary.tex):
    Can a reconstruction model exploit shared sensing geometry to generalize
    across unseen operators within a structured operator family?

Setup:
    Operator family  : Gaussian N(0, 1/m)   [single family, multiple m values]
    Training m       : {64, 128, 192}        2 instances each  -> 6 training ops
    Test (seen m)    : {64, 128, 192}        1 held-out instance each -> 3 ops
    Test (new m)     : {96, 160}             2 instances each -> 4 ops
    n=256, k=25, T=30, lam=0.05

Rationale:
    Experiments 1 and 1b used partial Fourier operators, which are near-
    isometries (sn_max≈1) for any m. No spectral diversity → conditioning
    has nothing to exploit.

    Gaussian operators at different m have genuinely different Marchenko-
    Pastur spectra. A^T A is (n x n) with rank ≤ m, so (n-m) eigenvalues
    are exactly zero. The nonzero eigenvalues follow the MP distribution
    on [(1-√γ)^2, (1+√γ)^2] where γ = m/n:
        m= 64  (γ=0.25): m nonzero eigs in [0.25, 2.25],  192 zeros
        m=128  (γ=0.50): m nonzero eigs in [0.09, 2.91],  128 zeros
        m=192  (γ=0.75): m nonzero eigs in [0.02, 3.48],   64 zeros
    The descriptor is the full 256-dim sorted eigenvalue vector; zero
    padding at the tail directly encodes m, and the nonzero portion
    encodes the spectral shape.

    CondLISTA encoder is now trained on Gaussian eigenvalue descriptors at all
    3 training m values. At test time on m=96 and m=160, it interpolates
    between seen spectra — this is in-distribution interpolation, NOT the
    cross-family extrapolation that caused Exp 2 to fail.

    Exp 2 failure root cause: encoder trained only on Fourier spectra
    (concentrated near 0.5), then queried with Gaussian spectra (broad,
    out-of-distribution). The encoder produced garbage modulations.

    Exp 3 fix: train encoder on multiple Gaussian m values so it learns
    the spectral signature of the Gaussian family and can interpolate.

Methods:
    1. ista_topk_ls    -- ISTA(T=30, per-op alpha) + top-k + LS  [0 params]
    2. shared_lista    -- shared per-step (alpha_t, lam_t)        [2*T=60 params]
    3. cond_lista      -- eigenvalue-conditioned (alpha_t, lam_t) [~10k params]

Key questions:
    Q1: Does CondLISTA beat SharedLISTA at new-m zero-shot?
        (tests whether eigenvalue encoding enables within-family interpolation)
    Q2: Does CondLISTA beat SharedLISTA at seen-m new instance?
        (tests whether conditioning helps even within a seen m value)
    Q3: How quickly does each method adapt with few new-m samples?

Usage:
    python exp3_gaussian_family.py
    python exp3_gaussian_family.py --device cuda

Outputs:
    results_exp3/exp3_results.json
    results_exp3/exp3_summary.png
    results_exp3/exp3_adapt_curves.png
"""

import argparse
import copy
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ──────────────────────────────────────────────────────────────
# 1. ARGUMENTS
# ──────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",          type=int,   default=256)
    p.add_argument("--k",          type=int,   default=25)
    p.add_argument("--amp_lo",     type=float, default=0.5)
    p.add_argument("--amp_hi",     type=float, default=2.0)
    p.add_argument("--T",          type=int,   default=30)
    p.add_argument("--lam",        type=float, default=0.05)
    p.add_argument("--n_train",    type=int,   default=4000)
    p.add_argument("--n_test",     type=int,   default=500)
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--ft_epochs",  type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--d_embed",    type=int,   default=32)
    p.add_argument("--ft_budgets", type=int,   nargs="+",
                   default=[0, 10, 25, 50, 100, 200])
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",    type=str,   default="results_exp3")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2. OPERATORS
# ──────────────────────────────────────────────────────────────

def make_gaussian_op(n, m, seed, device, label=""):
    """
    Gaussian N(0, 1/m) sensing operator.
    Returns dict with A, AtA, eigvals (descending), sn_max, alpha, m, label.
    """
    rng = np.random.RandomState(seed)
    A   = torch.tensor(
        rng.randn(m, n).astype(np.float32) / np.sqrt(m),
        device=device
    )
    with torch.no_grad():
        AtA     = A.T @ A                           # (n, n)
        eigvals = torch.linalg.eigvalsh(AtA)        # (n,) ascending
        sn_max  = math.sqrt(max(eigvals[-1].item(), 1e-8))
        alpha   = 1.0 / (sn_max ** 2)
        eigv    = eigvals.flip(0)                   # (n,) descending
    return {
        "A": A, "AtA": AtA, "eigvals": eigv,
        "sn_max": sn_max, "alpha": alpha,
        "m": m, "label": label,
    }


def build_operators(args, device):
    """
    Seed scheme: seed = m * 1000 + instance_id
      instance_id=0,1  -> training
      instance_id=2    -> test (seen m, held-out instance)
      instance_id=0,1  applied to new-m values -> test (new m)
    """
    train_m    = [64, 128, 192]
    test_new_m = [96, 160]

    train_ops      = []
    test_seen_ops  = []
    test_new_ops   = []
    global_sn_max  = 0.0

    print("  Training operators (Gaussian family):")
    for m in train_m:
        for inst in range(2):
            seed = m * 1000 + inst
            lbl  = f"G_train_m{m}_i{inst}"
            op   = make_gaussian_op(args.n, m, seed, device, lbl)
            train_ops.append(op)
            global_sn_max = max(global_sn_max, op["sn_max"])
            print(f"    {lbl}: m={m}  sn_max={op['sn_max']:.4f}  "
                  f"eig_max={op['eigvals'][0]:.3f}  "
                  f"eig_min={op['eigvals'][-1]:.5f}  "
                  f"eig_std={op['eigvals'].std().item():.4f}")

    print("  Test operators — seen m (held-out instance):")
    for m in train_m:
        seed = m * 1000 + 2
        lbl  = f"G_test_seenm{m}"
        op   = make_gaussian_op(args.n, m, seed, device, lbl)
        test_seen_ops.append(op)
        global_sn_max = max(global_sn_max, op["sn_max"])
        print(f"    {lbl}: m={m}  sn_max={op['sn_max']:.4f}  "
              f"eig_max={op['eigvals'][0]:.3f}  "
              f"eig_min={op['eigvals'][-1]:.5f}")

    print("  Test operators — new m value (interpolation test):")
    for m in test_new_m:
        for inst in range(2):
            seed = m * 1000 + inst
            lbl  = f"G_test_newm{m}_i{inst}"
            op   = make_gaussian_op(args.n, m, seed, device, lbl)
            test_new_ops.append(op)
            global_sn_max = max(global_sn_max, op["sn_max"])
            print(f"    {lbl}: m={m}  sn_max={op['sn_max']:.4f}  "
                  f"eig_max={op['eigvals'][0]:.3f}  "
                  f"eig_min={op['eigvals'][-1]:.5f}")

    global_alpha = 1.0 / (global_sn_max ** 2)
    print(f"\n  global sn_max={global_sn_max:.4f}   "
          f"global_alpha={global_alpha:.6f}")
    return train_ops, test_seen_ops, test_new_ops, global_sn_max, global_alpha


# ──────────────────────────────────────────────────────────────
# 3. DATA
# ──────────────────────────────────────────────────────────────

def make_signals(n, k, n_signals, amp_lo, amp_hi, seed, device):
    rng = np.random.RandomState(seed)
    X   = np.zeros((n_signals, n), dtype=np.float32)
    S   = np.zeros((n_signals, n), dtype=np.float32)
    for i in range(n_signals):
        supp       = rng.choice(n, k, replace=False)
        amps       = rng.uniform(amp_lo, amp_hi, k) * rng.choice([-1, 1], k)
        X[i, supp] = amps
        S[i, supp] = 1.0
    return (torch.tensor(X, device=device),
            torch.tensor(S, device=device))


# ──────────────────────────────────────────────────────────────
# 4. ISTA HELPERS
# ──────────────────────────────────────────────────────────────

def soft_threshold(x, lam):
    return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)


@torch.no_grad()
def ista_unroll(A, y, alpha, lam, T):
    """Unrolled ISTA using per-operator step size alpha."""
    x = torch.zeros(y.shape[0], A.shape[1], device=A.device)
    for _ in range(T):
        residual = x @ A.T - y
        x = soft_threshold(x - alpha * (residual @ A), lam)
    return x


# ──────────────────────────────────────────────────────────────
# 5. MODELS
# ──────────────────────────────────────────────────────────────

class SharedLISTA(nn.Module):
    """
    Standard LISTA: shared per-step (alpha_t, lam_t) with no operator info.
    Trained on mixed-m Gaussian data, learns a compromise schedule.
    Baseline: can it transfer to new m values without conditioning?
    Parameters: 2*T = 60.

    Per-operator sn_max is passed at forward time so that alpha_t is clamped
    to 2/||A||_2^2 (the correct gradient convergence bound) for each operator
    independently. This removes the need for a global alpha_max that is either
    too loose for large-m or too conservative for small-m.
    """
    def __init__(self, T, alpha_init, lam_init):
        super().__init__()
        self.T     = T
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))

    def forward(self, A, y, op_sn_max):
        """op_sn_max: largest singular value of A (scalar)."""
        alpha_max = 2.0 / (op_sn_max ** 2)   # convergence bound for this A
        x = torch.zeros(y.shape[0], A.shape[1], device=y.device)
        for t in range(self.T):
            residual = x @ A.T - y
            alpha_t  = self.alpha[t].clamp(min=1e-6, max=alpha_max)
            x = soft_threshold(x - alpha_t * (residual @ A),
                               self.lam[t].clamp(min=0.0))
        return x


class CondLISTA(nn.Module):
    """
    LISTA conditioned on sorted eigenvalues of A^T A.

    The 256-dim eigenvalue descriptor encodes both the m/n ratio (number of
    nonzero eigenvalues = m) and the Marchenko-Pastur spectral shape. The
    encoder maps this to per-step modulations of alpha and lam.

    Zero-initialized modulation: at epoch 0, CondLISTA == SharedLISTA.
    The encoder is trained on Gaussian descriptors at m in {64, 128, 192},
    so at m=96 and m=160 it interpolates rather than extrapolates.

    Key difference from Exp 2: the encoder is IN-DISTRIBUTION at test time.

    Per-operator sn_max passed at forward time ensures correct convergence
    bounds for each operator independently (same fix as SharedLISTA).
    """
    def __init__(self, T, alpha_init, lam_init, n, d_embed=32):
        super().__init__()
        self.T     = T
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))
        self.op_encoder = nn.Sequential(
            nn.Linear(n, d_embed), nn.Tanh(),
            nn.Linear(d_embed, d_embed), nn.Tanh(),
        )
        self.alpha_mod = nn.Linear(d_embed, T, bias=False)
        self.lam_mod   = nn.Linear(d_embed, T, bias=False)
        nn.init.zeros_(self.alpha_mod.weight)
        nn.init.zeros_(self.lam_mod.weight)

    def forward(self, A, y, eigvals, op_sn_max):
        """
        A         : (m, n)
        y         : (B, m)
        eigvals   : (n,) sorted eigenvalues of A^T A, descending
        op_sn_max : largest singular value of A (scalar)
        """
        alpha_max   = 2.0 / (op_sn_max ** 2)    # per-operator convergence bound
        e           = self.op_encoder(eigvals)   # (d_embed,)
        alpha_delta = self.alpha_mod(e)           # (T,)
        lam_delta   = self.lam_mod(e)             # (T,)
        x = torch.zeros(y.shape[0], A.shape[1], device=y.device)
        for t in range(self.T):
            residual = x @ A.T - y
            alpha_t  = (self.alpha[t] + alpha_delta[t]).clamp(
                min=1e-6, max=alpha_max)
            lam_t    = (self.lam[t] + lam_delta[t]).clamp(min=0.0)
            x = soft_threshold(x - alpha_t * (residual @ A), lam_t)
        return x


# ──────────────────────────────────────────────────────────────
# 6. METRICS
# ──────────────────────────────────────────────────────────────

def raw_nrmse(x_hat, x_true):
    num   = (x_hat - x_true).norm(dim=1)
    denom = x_true.norm(dim=1).clamp(min=1e-8)
    return (num / denom).mean().item()


def topk_ls_nrmse(scores, x_true, A, k, device):
    """Top-k support from ISTA iterate + LS amplitude correction."""
    batch, n = x_true.shape
    y        = x_true @ A.T
    topk_idx = torch.topk(scores.abs(), k, dim=1).indices
    errs = []
    for i in range(batch):
        supp = topk_idx[i]
        A_s  = A[:, supp]
        try:
            x_s, _, _, _ = torch.linalg.lstsq(A_s, y[i].unsqueeze(1),
                                               rcond=None)
            x_hat = torch.zeros(n, device=device)
            x_hat[supp] = x_s.squeeze()
        except Exception:
            x_hat = torch.zeros(n, device=device)
        errs.append(
            ((x_hat - x_true[i]).norm() /
             x_true[i].norm().clamp(min=1e-8)).item()
        )
    return float(np.mean(errs))


def oracle_ls_nrmse(x_true, A, S_true, device):
    """Oracle: known support → LS → NRMSE."""
    batch, n = x_true.shape
    y        = x_true @ A.T
    errs = []
    for i in range(batch):
        supp = S_true[i].bool()
        A_s  = A[:, supp]
        try:
            x_s, _, _, _ = torch.linalg.lstsq(A_s, y[i].unsqueeze(1),
                                               rcond=None)
            x_hat = torch.zeros(n, device=device)
            x_hat[supp] = x_s.squeeze()
        except Exception:
            x_hat = torch.zeros(n, device=device)
        errs.append(
            ((x_hat - x_true[i]).norm() /
             x_true[i].norm().clamp(min=1e-8)).item()
        )
    return float(np.mean(errs))


# ──────────────────────────────────────────────────────────────
# 7. TRAINING
# ──────────────────────────────────────────────────────────────

def train_model(model, model_type, train_ops, X_train, X_test, args, dev,
                label):
    """
    Trains on all training operators (mixed m).
    Each mini-batch randomly selects one operator, generates measurements,
    and updates shared parameters.
    Per-operator sn_max is passed to the forward so each operator gets
    its own convergence-safe alpha clamp.
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + (1 if model_type == "shared" else 2))
    history   = []

    print(f"\n── {label} training (mixed-m Gaussian: m={{64,128,192}}) ───")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            X_b = X_train[perm[start: start + args.batch_size]]
            op  = train_ops[int(rng.randint(len(train_ops)))]
            y_b = X_b @ op["A"].T
            if model_type == "shared":
                pred = model(op["A"], y_b, op["sn_max"])
            else:
                pred = model(op["A"], y_b, op["eigvals"], op["sn_max"])
            loss = nn.functional.mse_loss(pred, X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                avg = _avg_nrmse(model, model_type, train_ops, X_test)
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  train-avg-NRMSE={avg:.4f}")
    return history


@torch.no_grad()
def _avg_nrmse(model, model_type, ops, X_test):
    vals = []
    for op in ops:
        y = X_test @ op["A"].T
        if model_type == "shared":
            x_hat = model(op["A"], y, op["sn_max"])
        else:
            x_hat = model(op["A"], y, op["eigvals"], op["sn_max"])
        vals.append(raw_nrmse(x_hat, X_test))
    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────
# 8. EVALUATION
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_op(op, x_true, S_true, args, dev,
            shared_lista=None, cond_lista=None):
    A   = op["A"]
    y   = x_true @ A.T
    xT  = ista_unroll(A, y, op["alpha"], args.lam, args.T)
    res = {}
    res["oracle_nrmse"]    = oracle_ls_nrmse(x_true, A, S_true, dev)
    res["ista_topk_nrmse"] = topk_ls_nrmse(xT, x_true, A, args.k, dev)
    if shared_lista is not None:
        res["shared_lista_nrmse"] = raw_nrmse(
            shared_lista(A, y, op["sn_max"]), x_true)
    if cond_lista is not None:
        res["cond_lista_nrmse"] = raw_nrmse(
            cond_lista(A, y, op["eigvals"], op["sn_max"]), x_true)
    return res


# ──────────────────────────────────────────────────────────────
# 9. ADAPTATION CURVES
# ──────────────────────────────────────────────────────────────

def adapt_curve(model_init, op, X_pool, X_test, args, dev,
                model_type, label):
    """Fine-tune on N samples from new operator, report NRMSE curve."""
    A    = op["A"]
    eigv = op["eigvals"]

    sn_max = op["sn_max"]

    def _infer(m):
        with torch.no_grad():
            y = X_test @ A.T
            if model_type == "shared":
                return raw_nrmse(m(A, y, sn_max), X_test)
            else:
                return raw_nrmse(m(A, y, eigv, sn_max), X_test)

    results = {0: _infer(model_init)}

    for N in args.ft_budgets:
        if N == 0:
            continue
        N_actual  = min(N, X_pool.shape[0])
        X_ft      = X_pool[:N_actual]
        model_ft  = copy.deepcopy(model_init)
        optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)

        for _ in range(args.ft_epochs):
            model_ft.train()
            perm = torch.randperm(N_actual, device=dev)
            for start in range(0, N_actual, args.batch_size):
                X_b = X_ft[perm[start: start + args.batch_size]]
                y_b = X_b @ A.T
                if model_type == "shared":
                    pred = model_ft(A, y_b, sn_max)
                else:
                    pred = model_ft(A, y_b, eigv, sn_max)
                loss = nn.functional.mse_loss(pred, X_b)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model_ft.eval()
        nrmse = _infer(model_ft)
        results[N] = nrmse
        print(f"    [{label}] N={N_actual:>4}  NRMSE={nrmse:.4f}")
    return results


# ──────────────────────────────────────────────────────────────
# 10. VERDICT
# ──────────────────────────────────────────────────────────────

def print_verdict(seen_res, new_m_res, adapt_seen, adapt_new, args):
    def avg(result_list, key):
        v = [r[key] for r in result_list if key in r]
        return float(np.mean(v)) if v else float("nan")

    print("\n" + "=" * 68)
    print("  EXP 3 VERDICT — Within-Family Gaussian Transfer (Variable m)")
    print("=" * 68)

    print("\n  [TEST: SEEN m {64,128,192} — held-out instance, zero-shot]")
    for key, name in [("ista_topk_nrmse",    "ISTA top-k+LS"),
                      ("shared_lista_nrmse", "SharedLISTA"),
                      ("cond_lista_nrmse",   "CondLISTA")]:
        v = avg(seen_res, key)
        if not math.isnan(v):
            print(f"    {name:<24}: {v:.4f}")

    print("\n  [TEST: NEW m {96,160} — unseen m value, zero-shot]")
    vals_new = {}
    for key, name in [("ista_topk_nrmse",    "ISTA top-k+LS"),
                      ("shared_lista_nrmse", "SharedLISTA"),
                      ("cond_lista_nrmse",   "CondLISTA")]:
        v = avg(new_m_res, key)
        vals_new[key] = v
        if not math.isnan(v):
            print(f"    {name:<24}: {v:.4f}")

    shared_zs = vals_new.get("shared_lista_nrmse", float("nan"))
    cond_zs   = vals_new.get("cond_lista_nrmse",   float("nan"))
    gap = shared_zs - cond_zs

    print()
    if gap > 0.005:
        print(f"  [Q1 PASS] CondLISTA beats SharedLISTA at new-m zero-shot:")
        print(f"            SharedLISTA={shared_zs:.4f}  "
              f"CondLISTA={cond_zs:.4f}  (Δ={gap:.4f})")
        print("            Eigenvalue encoding enables within-family m-interpolation.")
    elif abs(gap) <= 0.005:
        print(f"  [Q1 INCONCLUSIVE] CondLISTA ≈ SharedLISTA at new-m:")
        print(f"            SharedLISTA={shared_zs:.4f}  CondLISTA={cond_zs:.4f}")
        print("            SharedLISTA adapts to new m without needing conditioning,")
        print("            OR the encoder learns nothing useful beyond the base schedule.")
    else:
        print(f"  [Q1 NEGATIVE] CondLISTA worse than SharedLISTA at new-m:")
        print(f"            SharedLISTA={shared_zs:.4f}  "
              f"CondLISTA={cond_zs:.4f}  (gap={gap:.4f})")

    budgets = sorted(next(iter(adapt_new["shared"].values())).keys())
    print(f"\n  [ADAPTATION — new-m avg NRMSE vs fine-tuning budget]")
    print(f"  {'Budget':>7}  {'SharedLISTA':>12}  {'CondLISTA':>10}")
    for b in budgets:
        def mb(mkey):
            v = [adapt_new[mkey][op["label"]].get(b, float("nan"))
                 for op in adapt_new["_ops"]]
            return float(np.mean(v))
        print(f"  {b:>7}  {mb('shared'):>12.4f}  {mb('cond'):>10.4f}")

    print("\n  [ADAPTATION — seen-m avg NRMSE vs fine-tuning budget]")
    print(f"  {'Budget':>7}  {'SharedLISTA':>12}  {'CondLISTA':>10}")
    budgets_s = sorted(next(iter(adapt_seen["shared"].values())).keys())
    for b in budgets_s:
        def mb_s(mkey):
            v = [adapt_seen[mkey][op["label"]].get(b, float("nan"))
                 for op in adapt_seen["_ops"]]
            return float(np.mean(v))
        print(f"  {b:>7}  {mb_s('shared'):>12.4f}  {mb_s('cond'):>10.4f}")

    print("=" * 68)


# ──────────────────────────────────────────────────────────────
# 11. PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_summary(seen_res, new_m_res, args, out_path):
    """Bar chart comparing methods on seen-m and new-m test sets."""
    methods = ["ista_topk_nrmse", "shared_lista_nrmse", "cond_lista_nrmse"]
    labels  = ["ISTA\ntop-k+LS", "Shared\nLISTA", "Cond\nLISTA"]
    colors  = ["steelblue", "forestgreen", "crimson"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (res_dict, title) in zip(axes, [
        (seen_res,  "Seen m {64,128,192}\n(held-out instance, zero-shot)"),
        (new_m_res, "New m {96,160}\n(unseen m value, zero-shot)"),
    ]):
        vals = []
        for m in methods:
            v = [r[m] for r in res_dict.values() if m in r]
            vals.append(float(np.mean(v)) if v else float("nan"))

        bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Exp 3: Within-Family Gaussian Transfer — Variable m\n"
        f"n={args.n}, k={args.k}, T={args.T}, λ={args.lam}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Summary plot -> {out_path}")
    plt.close()


def plot_adapt_curves(adapt_new, adapt_seen, args, out_path):
    """Adaptation curves for new-m and seen-m operators."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (adapt, title, ops_key) in zip(axes, [
        (adapt_new,  "New m {96, 160}", "_ops"),
        (adapt_seen, "Seen m {64,128,192}", "_ops"),
    ]):
        ops     = adapt[ops_key]
        budgets = sorted(next(iter(adapt["shared"].values())).keys())

        def mean_curve(mkey):
            return [
                float(np.mean([adapt[mkey][op["label"]].get(b, float("nan"))
                                for op in ops]))
                for b in budgets
            ]

        for mkey, color, marker, lbl in [
            ("shared", "forestgreen", "o-", "SharedLISTA"),
            ("cond",   "crimson",     "s-", "CondLISTA"),
        ]:
            ys = mean_curve(mkey)
            ax.plot(budgets, ys, marker, color=color, lw=2.0, ms=7,
                    label=lbl)

        ax.set_xlabel("Fine-tuning samples (N)")
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Exp 3: Adaptation Curves — SharedLISTA vs CondLISTA",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Adaptation curves plot -> {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 12. MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args = get_args()
    dev  = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device  : {dev}")
    print(f"n={args.n}  k={args.k}  T={args.T}  lam={args.lam}")
    print(f"Training m: {{64, 128, 192}}  (2 instances each -> 6 train ops)")
    print(f"Test seen m: {{64, 128, 192}}  (1 held-out instance -> 3 ops)")
    print(f"Test new m:  {{96, 160}}       (2 instances each -> 4 ops)")

    # ── Build operators ───────────────────────────────────────
    print("\n── Building operators ───────────────────────────────────")
    (train_ops, test_seen_ops, test_new_ops,
     global_sn_max, global_alpha) = build_operators(args, dev)

    # ── Signals ───────────────────────────────────────────────
    X_train, _ = make_signals(
        args.n, args.k, args.n_train,
        args.amp_lo, args.amp_hi, seed=args.seed, device=dev
    )
    X_test, S_test = make_signals(
        args.n, args.k, args.n_test,
        args.amp_lo, args.amp_hi, seed=args.seed + 999, device=dev
    )

    # ── Build models ──────────────────────────────────────────
    # Initialise at the smallest safe step (global worst case) so all
    # operators are stable from epoch 0. Per-operator alpha_max in forward
    # lets each operator use its own convergence bound during training.
    alpha_init = global_alpha              # 1 / global_sn_max^2
    lam_init   = args.lam * alpha_init    # matches ISTA soft-threshold scale

    shared_lista = SharedLISTA(args.T, alpha_init, lam_init).to(dev)
    cond_lista   = CondLISTA(
        args.T, alpha_init, lam_init, args.n, d_embed=args.d_embed
    ).to(dev)

    n_shared = sum(p.numel() for p in shared_lista.parameters())
    n_cond   = sum(p.numel() for p in cond_lista.parameters())
    print(f"\nParameters: SharedLISTA={n_shared}  CondLISTA={n_cond}")

    # ── Train both models on mixed-m Gaussian family ──────────
    train_model(shared_lista, "shared", train_ops, X_train, X_test,
                args, dev, "SharedLISTA")
    shared_lista.eval()

    train_model(cond_lista, "cond", train_ops, X_train, X_test,
                args, dev, "CondLISTA")
    cond_lista.eval()

    # ── Evaluate zero-shot ────────────────────────────────────
    print("\n── Zero-shot evaluation ─────────────────────────────────")
    seen_res  = {}
    new_m_res = {}

    print("\n  Test: seen-m (held-out instance):")
    for op in test_seen_ops:
        with torch.no_grad():
            r = eval_op(op, X_test, S_test, args, dev,
                        shared_lista, cond_lista)
        seen_res[op["label"]] = r
        print(f"    {op['label']}: "
              f"ista={r['ista_topk_nrmse']:.4f}  "
              f"shared={r['shared_lista_nrmse']:.4f}  "
              f"cond={r['cond_lista_nrmse']:.4f}")

    print("\n  Test: new-m (unseen m value):")
    for op in test_new_ops:
        with torch.no_grad():
            r = eval_op(op, X_test, S_test, args, dev,
                        shared_lista, cond_lista)
        new_m_res[op["label"]] = r
        print(f"    {op['label']}: "
              f"ista={r['ista_topk_nrmse']:.4f}  "
              f"shared={r['shared_lista_nrmse']:.4f}  "
              f"cond={r['cond_lista_nrmse']:.4f}")

    # ── Adaptation curves ─────────────────────────────────────
    print("\n── Adaptation curves (new-m operators) ─────────────────")
    adapt_new = {"shared": {}, "cond": {}, "_ops": test_new_ops}
    for op in test_new_ops:
        lbl = op["label"]
        print(f"\n  {lbl} (m={op['m']}):")
        print(f"    SharedLISTA:")
        adapt_new["shared"][lbl] = adapt_curve(
            shared_lista, op, X_train, X_test, args, dev,
            "shared", "SharedLISTA"
        )
        print(f"    CondLISTA:")
        adapt_new["cond"][lbl] = adapt_curve(
            cond_lista, op, X_train, X_test, args, dev,
            "cond", "CondLISTA"
        )

    print("\n── Adaptation curves (seen-m operators) ─────────────────")
    adapt_seen = {"shared": {}, "cond": {}, "_ops": test_seen_ops}
    for op in test_seen_ops:
        lbl = op["label"]
        print(f"\n  {lbl} (m={op['m']}):")
        print(f"    SharedLISTA:")
        adapt_seen["shared"][lbl] = adapt_curve(
            shared_lista, op, X_train, X_test, args, dev,
            "shared", "SharedLISTA"
        )
        print(f"    CondLISTA:")
        adapt_seen["cond"][lbl] = adapt_curve(
            cond_lista, op, X_train, X_test, args, dev,
            "cond", "CondLISTA"
        )

    # ── Verdict ───────────────────────────────────────────────
    print_verdict(
        list(seen_res.values()), list(new_m_res.values()),
        adapt_seen, adapt_new, args
    )

    # ── Save JSON ─────────────────────────────────────────────
    def ser_res(d):
        return {k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in d.items()}

    def ser_curves(c):
        return {
            k: {lbl: {str(b): float(v) for b, v in curve.items()}
                for lbl, curve in op_curves.items()}
            for k, op_curves in c.items() if k != "_ops"
        }

    out_json = os.path.join(args.out_dir, "exp3_results.json")
    with open(out_json, "w") as fh:
        json.dump({
            "args":           vars(args),
            "global_sn_max":  float(global_sn_max),
            "global_alpha":   float(global_alpha),
            "n_params":       {"shared_lista": n_shared, "cond_lista": n_cond},
            "seen_results":   ser_res(seen_res),
            "new_m_results":  ser_res(new_m_res),
            "adapt_new":      ser_curves(adapt_new),
            "adapt_seen":     ser_curves(adapt_seen),
        }, fh, indent=2)
    print(f"\nResults JSON -> {out_json}")

    # ── Plots ─────────────────────────────────────────────────
    plot_summary(
        seen_res, new_m_res, args,
        os.path.join(args.out_dir, "exp3_summary.png")
    )
    plot_adapt_curves(
        adapt_new, adapt_seen, args,
        os.path.join(args.out_dir, "exp3_adapt_curves.png")
    )


if __name__ == "__main__":
    main()
