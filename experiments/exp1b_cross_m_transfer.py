"""
exp1b_cross_m_transfer.py

Cross-acceleration transfer: train on m=128 (2x undersampling),
test zero-shot on m=64 (4x undersampling).

Motivation: the Exp 1 homogeneous-mask experiment was too easy — all masks
shared the same m, so the LISTA step sizes were identically optimal across
all operators and conditioning provided no signal. This experiment creates
genuine operator diversity by varying the measurement count m, which:
  - Changes the optimal step size and threshold schedule
  - Changes the information content of b_freq (64 vs 32 active frequencies)
  - Maps directly to the MRI use case (train at 2x, deploy at 4x acceleration)

Methods:
  1. ista_baseline  -- ISTA top-k+LS on the actual operator  [0 learned params]
  2. shared_lista   -- LISTA trained on m=128 only, zero-shot on m=64
  3. cond_lista     -- CondLISTA trained on m=128, zero-shot on m=64 via b_freq
  4. pm_lista_m64   -- PerMask LISTA trained directly on m=64  [oracle upper bound]

pm_lista_m64 answers: what is achievable with m=64 training data?
The gap shared_lista -> pm_lista_m64 quantifies the cost of missing target-domain data.
The gap shared_lista -> cond_lista quantifies what conditioning recovers for free.

Adaptation curves: given N samples from m=64, how fast does each method converge
to the pm_lista_m64 performance ceiling?

Key question:
  Does CondLISTA zero-shot NRMSE < Shared LISTA zero-shot NRMSE on m=64 operators?
  -> If yes: b_freq conditioning carries cross-acceleration info (measurement count
     is encoded as sum(b_freq) = m//2 — directly tells the model how undersampled it is).
  -> If no: the learned step-size schedule is the bottleneck, and conditioning the
     encoder alone is insufficient. Would need to change architecture.

Usage:
    python exp1b_cross_m_transfer.py
    python exp1b_cross_m_transfer.py --device cuda

Outputs:
    results_exp1b/exp1b_results.json
    results_exp1b/exp1b_summary.png
    results_exp1b/exp1b_adapt_curves.png
"""

import argparse
import copy
import json
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
    p.add_argument("--n",               type=int,   default=256)
    p.add_argument("--m_seen",          type=int,   default=128,
                   help="Measurements for training operators (2x acceleration)")
    p.add_argument("--m_unseen",        type=int,   default=64,
                   help="Measurements for test operators (4x acceleration)")
    p.add_argument("--k",               type=int,   default=25)
    p.add_argument("--amp_lo",          type=float, default=0.5)
    p.add_argument("--amp_hi",          type=float, default=2.0)
    p.add_argument("--T",               type=int,   default=30)
    p.add_argument("--lam",             type=float, default=0.05)
    p.add_argument("--n_seen",          type=int,   default=5,
                   help="Number of training operators (m=m_seen)")
    p.add_argument("--n_unseen",        type=int,   default=5,
                   help="Number of test operators (m=m_unseen)")
    p.add_argument("--n_train",         type=int,   default=4000)
    p.add_argument("--n_test",          type=int,   default=500)
    p.add_argument("--epochs",          type=int,   default=150)
    p.add_argument("--ft_epochs",       type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--d_embed",         type=int,   default=32,
                   help="CondLISTA operator-embedding dimension")
    p.add_argument("--ft_budgets",      type=int,   nargs="+",
                   default=[0, 10, 25, 50, 100, 200])
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--seen_seed_base",  type=int,   default=100,
                   help="Seen operator i uses seed = seen_seed_base + i*100")
    p.add_argument("--unseen_seed_base",type=int,   default=600,
                   help="Unseen operator i uses seed = unseen_seed_base + i*100")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",         type=str,   default="results_exp1b")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2. OPERATOR GENERATION
# ──────────────────────────────────────────────────────────────

def make_partial_fourier(n, m, mask_seed, device):
    """
    Generate one partial Fourier operator (m, n) and its frequency indicator.

    b_freq: binary (n,) vector — 1 at each sampled DFT frequency bin.
    sum(b_freq) = m//2 = n_freqs, which directly encodes the acceleration factor.

    For m=128: sum(b_freq) = 64 (2x acceleration)
    For m=64:  sum(b_freq) = 32 (4x acceleration)
    The CondLISTA encoder sees this difference and can use it to adapt.
    """
    rng      = np.random.RandomState(mask_seed)
    n_freqs  = m // 2
    freq_idx = np.sort(rng.choice(n, n_freqs, replace=False))
    F_complex = np.fft.fft(np.eye(n)) / np.sqrt(n)
    rows = np.concatenate(
        [F_complex[freq_idx].real, F_complex[freq_idx].imag], axis=0
    )
    A = torch.tensor(rows[:m], dtype=torch.float32, device=device)

    b_freq = torch.zeros(n, dtype=torch.float32, device=device)
    b_freq[freq_idx] = 1.0

    sn = torch.linalg.norm(A, ord=2).item() ** 2
    with torch.no_grad():
        AtA = A.T @ A
    return {"A": A, "b_freq": b_freq, "freq_idx": freq_idx, "sn": sn, "AtA": AtA}


def build_operator_sets(args, device):
    """
    Returns:
        seen_ops   : list of n_seen operators with m = m_seen
        unseen_ops : list of n_unseen operators with m = m_unseen
        sn_max     : float, max spectral norm^2 across all operators
        alpha      : float, LISTA step size = 1/sn_max
    """
    seen_ops, unseen_ops = [], []
    sn_max = 0.0

    print("  Seen operators (m={})".format(args.m_seen))
    for i in range(args.n_seen):
        op = make_partial_fourier(
            args.n, args.m_seen,
            mask_seed=args.seen_seed_base + i * 100,
            device=device
        )
        op["mask_id"] = i
        op["m"]       = args.m_seen
        sn_max        = max(sn_max, op["sn"])
        seen_ops.append(op)
        print(f"    mask {i}  sn={op['sn']:.4f}  "
              f"sum(b_freq)={int(op['b_freq'].sum().item())} "
              f"(={args.m_seen//2} freqs)")

    print("  Unseen operators (m={})".format(args.m_unseen))
    for i in range(args.n_unseen):
        op = make_partial_fourier(
            args.n, args.m_unseen,
            mask_seed=args.unseen_seed_base + i * 100,
            device=device
        )
        op["mask_id"] = args.n_seen + i
        op["m"]       = args.m_unseen
        sn_max        = max(sn_max, op["sn"])
        unseen_ops.append(op)
        print(f"    mask {i}  sn={op['sn']:.4f}  "
              f"sum(b_freq)={int(op['b_freq'].sum().item())} "
              f"(={args.m_unseen//2} freqs)")

    alpha = 1.0 / sn_max
    print(f"  sn_max={sn_max:.4f}   alpha={alpha:.6f}")
    return seen_ops, unseen_ops, sn_max, alpha


# ──────────────────────────────────────────────────────────────
# 3. DATA GENERATION
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
    x = torch.zeros(y.shape[0], A.shape[1], device=A.device)
    for _ in range(T):
        residual = x @ A.T - y
        x = soft_threshold(x - alpha * (residual @ A), lam)
    return x


# ──────────────────────────────────────────────────────────────
# 5. MODELS
# ──────────────────────────────────────────────────────────────

class LISTA(nn.Module):
    """
    Learned ISTA: per-step scalar alpha and lam.
    Dimension-agnostic: works for any m since the gradient step scales naturally.
    Parameters: 2*T.
    """
    def __init__(self, T, alpha_init, lam_init, sn_max):
        super().__init__()
        self.T         = T
        self.alpha_max = float(2.0 / sn_max)
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))

    def forward(self, A, y):
        x = torch.zeros(y.shape[0], A.shape[1], device=y.device)
        for t in range(self.T):
            residual = x @ A.T - y
            alpha_t  = self.alpha[t].clamp(min=1e-6, max=self.alpha_max)
            x = soft_threshold(
                x - alpha_t * (residual @ A),
                self.lam[t].clamp(min=0.0)
            )
        return x


class CondLISTA(nn.Module):
    """
    LISTA conditioned on b_freq in {0,1}^n.

    Why conditioning should work here (unlike Exp 1):
      sum(b_freq) = m//2 encodes the acceleration factor directly.
      Seen: sum=64 (m=128, 2x accel) -> one optimal (alpha_t, lam_t) schedule
      Unseen: sum=32 (m=64, 4x accel) -> different optimal schedule (more
      undersampled, noisier iterate, needs different thresholding)

    The encoder maps b_freq -> d_embed features; the modulation layers
    produce per-step additive corrections to alpha and lam.
    Zero-initialized modulation: starts as plain LISTA, learns corrections.
    """
    def __init__(self, T, alpha_init, lam_init, sn_max, n, d_embed=32):
        super().__init__()
        self.T         = T
        self.alpha_max = float(2.0 / sn_max)
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))

        self.op_encoder = nn.Sequential(
            nn.Linear(n, d_embed),
            nn.Tanh(),
        )
        self.alpha_mod = nn.Linear(d_embed, T, bias=False)
        self.lam_mod   = nn.Linear(d_embed, T, bias=False)
        nn.init.zeros_(self.alpha_mod.weight)
        nn.init.zeros_(self.lam_mod.weight)

    def forward(self, A, y, b_freq):
        e           = self.op_encoder(b_freq)   # (d_embed,)
        alpha_delta = self.alpha_mod(e)          # (T,)
        lam_delta   = self.lam_mod(e)            # (T,)
        x = torch.zeros(y.shape[0], A.shape[1], device=y.device)
        for t in range(self.T):
            residual = x @ A.T - y
            alpha_t  = (self.alpha[t] + alpha_delta[t]).clamp(min=1e-6, max=self.alpha_max)
            lam_t    = (self.lam[t]   + lam_delta[t]).clamp(min=0.0)
            x = soft_threshold(x - alpha_t * (residual @ A), lam_t)
        return x

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# 6. METRICS
# ──────────────────────────────────────────────────────────────

def raw_nrmse(x_hat, x_true):
    num   = (x_hat - x_true).norm(dim=1)
    denom = x_true.norm(dim=1).clamp(min=1e-8)
    return (num / denom).mean().item()


def topk_ls_nrmse(scores, x_true, A, k, device):
    batch, n = x_true.shape
    y        = x_true @ A.T
    topk_idx = torch.topk(scores.abs(), k, dim=1).indices
    errs = []
    for i in range(batch):
        supp = topk_idx[i]
        A_s  = A[:, supp]
        try:
            x_s, _, _, _ = torch.linalg.lstsq(A_s, y[i].unsqueeze(1), rcond=None)
            x_hat = torch.zeros(n, device=device)
            x_hat[supp] = x_s.squeeze()
        except Exception:
            x_hat = torch.zeros(n, device=device)
        errs.append(
            ((x_hat - x_true[i]).norm() / x_true[i].norm().clamp(min=1e-8)).item()
        )
    return float(np.mean(errs))


def oracle_ls_nrmse(x_true, A, S_true, device):
    batch, n = x_true.shape
    y        = x_true @ A.T
    errs = []
    for i in range(batch):
        supp = S_true[i].bool()
        A_s  = A[:, supp]
        try:
            x_s, _, _, _ = torch.linalg.lstsq(A_s, y[i].unsqueeze(1), rcond=None)
            x_hat = torch.zeros(n, device=device)
            x_hat[supp] = x_s.squeeze()
        except Exception:
            x_hat = torch.zeros(n, device=device)
        errs.append(
            ((x_hat - x_true[i]).norm() / x_true[i].norm().clamp(min=1e-8)).item()
        )
    return float(np.mean(errs))


# ──────────────────────────────────────────────────────────────
# 7. TRAINING
# ──────────────────────────────────────────────────────────────

def _train_loop(model, ops, X_train, X_test, args, dev,
                model_type="lista", label="", rng_seed=1):
    """
    Shared training loop.  model_type: "lista" | "cond_lista"
    For each batch: pick a random op from ops, generate y, train.
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(rng_seed)
    history   = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            idx    = perm[start: start + args.batch_size]
            X_b    = X_train[idx]
            op     = ops[int(rng.randint(len(ops)))]
            y      = X_b @ op["A"].T
            if model_type == "cond_lista":
                pred = model(op["A"], y, op["b_freq"])
            else:
                pred = model(op["A"], y)
            loss = nn.functional.mse_loss(pred, X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            model.eval()
            avg = _eval_avg_nrmse(model, ops, X_test, args, dev, model_type)
            history.append(avg)
            print(f"  [{label}] Epoch {epoch+1:>4}  avg-NRMSE={avg:.4f}")
    return history


@torch.no_grad()
def _eval_avg_nrmse(model, ops, X_test, args, dev, model_type):
    vals = []
    for op in ops:
        y = X_test @ op["A"].T
        if model_type == "cond_lista":
            pred = model(op["A"], y, op["b_freq"])
        else:
            pred = model(op["A"], y)
        vals.append(raw_nrmse(pred, X_test))
    return float(np.mean(vals))


def train_shared_lista(lista, seen_ops, X_train, X_test, args, dev):
    print("\n── Shared LISTA (trained on m={}) ─────────────────────".format(
        args.m_seen))
    return _train_loop(lista, seen_ops, X_train, X_test, args, dev,
                       model_type="lista", label="SharedLISTA", rng_seed=args.seed+1)


def train_cond_lista(cond_lista, seen_ops, X_train, X_test, args, dev):
    print("\n── CondLISTA (trained on m={}) ─────────────────────────".format(
        args.m_seen))
    return _train_loop(cond_lista, seen_ops, X_train, X_test, args, dev,
                       model_type="cond_lista", label="CondLISTA", rng_seed=args.seed+2)


def train_pm_lista_m_unseen(unseen_ops, X_train, X_test, alpha, args, dev, sn_max):
    """
    Oracle: train one LISTA per unseen (m=m_unseen) operator.
    These models are trained with actual m_unseen data — they represent
    the performance ceiling for the unseen acceleration factor.
    """
    print("\n── PerMask LISTA (oracle, trained on m={}) ─────────────".format(
        args.m_unseen))
    models = []
    for op in unseen_ops:
        mid   = op["mask_id"]
        lista = LISTA(args.T, alpha, args.lam, sn_max).to(dev)
        print(f"\n  Mask {mid} (m={args.m_unseen}):")
        _train_loop(lista, [op], X_train, X_test, args, dev,
                    model_type="lista",
                    label=f"PM_m{args.m_unseen}_mask{mid}",
                    rng_seed=args.seed + 100 + mid)
        lista.eval()
        models.append(lista)
    return models


# ──────────────────────────────────────────────────────────────
# 8. EVALUATION
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_on_operator(op, x_true, S_true, alpha, args, dev,
                     shared_lista=None, cond_lista=None, pm_lista=None):
    A, b_freq = op["A"], op["b_freq"]
    y  = x_true @ A.T
    xT = ista_unroll(A, y, alpha, args.lam, args.T)
    res = {}

    res["oracle_nrmse"]    = oracle_ls_nrmse(x_true, A, S_true, dev)
    res["ista_topk_nrmse"] = topk_ls_nrmse(xT, x_true, A, args.k, dev)

    if shared_lista is not None:
        res["shared_lista_nrmse"] = raw_nrmse(shared_lista(A, y), x_true)
    if cond_lista is not None:
        res["cond_lista_nrmse"] = raw_nrmse(cond_lista(A, y, b_freq), x_true)
    if pm_lista is not None:
        res["pm_lista_nrmse"] = raw_nrmse(pm_lista(A, y), x_true)
    return res


# ──────────────────────────────────────────────────────────────
# 9. ADAPTATION CURVES
# ──────────────────────────────────────────────────────────────

def adapt_curve(model_init, op, X_pool, X_test, alpha, args, dev,
                model_type="lista", label=""):
    """Fine-tune model_init on N m_unseen samples; return budget -> NRMSE dict."""
    A, b_freq = op["A"], op["b_freq"]

    base = copy.deepcopy(model_init).eval()
    with torch.no_grad():
        if model_type == "cond_lista":
            zs = raw_nrmse(base(A, X_test @ A.T, b_freq), X_test)
        else:
            zs = raw_nrmse(base(A, X_test @ A.T), X_test)
    results = {0: zs}

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
                idx = perm[start: start + args.batch_size]
                X_b = X_ft[idx]
                y   = X_b @ A.T
                if model_type == "cond_lista":
                    pred = model_ft(A, y, b_freq)
                else:
                    pred = model_ft(A, y)
                loss = nn.functional.mse_loss(pred, X_b)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model_ft.eval()
        with torch.no_grad():
            if model_type == "cond_lista":
                nrmse = raw_nrmse(model_ft(A, X_test @ A.T, b_freq), X_test)
            else:
                nrmse = raw_nrmse(model_ft(A, X_test @ A.T), X_test)
        results[N] = nrmse
        print(f"    [{label}] N={N_actual:>4}  NRMSE={nrmse:.4f}")
    return results


def build_adapt_curves(unseen_ops, shared_lista, cond_lista,
                       X_train, X_test, alpha, args, dev):
    print("\n── Adaptation curves on m={} operators ─────────────────".format(
        args.m_unseen))
    curves = {"shared_lista": {}, "cond_lista": {}}
    for op in unseen_ops:
        mid = op["mask_id"]
        print(f"\n  Mask {mid} (m={args.m_unseen}):")
        print("    shared_lista:")
        curves["shared_lista"][mid] = adapt_curve(
            shared_lista, op, X_train, X_test, alpha, args, dev,
            model_type="lista", label="shared"
        )
        print("    cond_lista:")
        curves["cond_lista"][mid] = adapt_curve(
            cond_lista, op, X_train, X_test, alpha, args, dev,
            model_type="cond_lista", label="cond"
        )
    return curves


# ──────────────────────────────────────────────────────────────
# 10. PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_summary(seen_res, unseen_res, oracle_seen, oracle_unseen,
                 args, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Seen panel — only methods that were actually evaluated on seen masks
    methods_seen   = ["ista_topk_nrmse", "shared_lista_nrmse", "cond_lista_nrmse"]
    labels_seen    = ["ISTA top-k+LS", "Shared LISTA", "Cond LISTA"]
    colors_seen    = ["steelblue", "forestgreen", "crimson"]

    # Unseen panel — all four methods
    methods_unseen = ["ista_topk_nrmse", "shared_lista_nrmse",
                      "cond_lista_nrmse", "pm_lista_nrmse"]
    labels_unseen  = ["ISTA top-k+LS", "Shared LISTA\n(m=128 trained)",
                      "Cond LISTA\n(m=128 trained)", "PerMask LISTA\n(m=64 oracle)"]
    colors_unseen  = ["steelblue", "forestgreen", "crimson", "darkorange"]

    for ax, (methods, labels, colors, results, oracle, title) in zip(axes, [
        (methods_seen,   labels_seen,   colors_seen,   seen_res,
         oracle_seen,   f"Seen masks (m={args.m_seen}, avg NRMSE)"),
        (methods_unseen, labels_unseen, colors_unseen, unseen_res,
         oracle_unseen, f"Unseen masks (m={args.m_unseen}, zero-shot avg NRMSE)"),
    ]):
        vals = [
            float(np.mean([r[m] for r in results.values() if m in r]))
            for m in methods
        ]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.axhline(oracle, color="black", ls=":", lw=1.5,
                   label=f"oracle LS={oracle:.3f}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Exp 1b: Cross-Acceleration Transfer  |  "
        f"Train m={args.m_seen} (2×)  →  Test m={args.m_unseen} (4×)\n"
        f"n={args.n}, k={args.k}, T={args.T}, lam={args.lam}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSummary plot -> {out_path}")
    plt.close()


def plot_adapt_curves(curves, pm_ceiling, ista_baseline,
                      oracle_unseen, args, out_path):
    """
    Adaptation curves for all unseen masks.
    pm_ceiling: dict mask_id -> NRMSE for the oracle per-mask m_unseen model.
    ista_baseline: dict mask_id -> ISTA top-k NRMSE for unseen masks.
    """
    unseen_ids = sorted(curves["shared_lista"].keys())
    n_unseen   = len(unseen_ids)
    ncols = min(n_unseen, 3)
    nrows = (n_unseen + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    style = {
        "shared_lista": ("forestgreen", "o-", f"Shared LISTA (m={args.m_seen} init)"),
        "cond_lista":   ("crimson",     "^-", f"Cond LISTA (m={args.m_seen} init)"),
    }
    budgets = sorted({b for mc in curves["shared_lista"].values() for b in mc})

    for idx, mid in enumerate(unseen_ids):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]

        for method, (color, marker, mlabel) in style.items():
            if mid not in curves[method]:
                continue
            curve = curves[method][mid]
            xs = [b for b in budgets if b in curve]
            ys = [curve[b] for b in xs]
            ax.plot(xs, ys, marker, color=color, ms=6, lw=2, label=mlabel)

        if mid in pm_ceiling:
            ax.axhline(pm_ceiling[mid], color="darkorange",
                       ls="--", lw=1.8,
                       label=f"Oracle m={args.m_unseen}={pm_ceiling[mid]:.3f}")
        ax.axhline(ista_baseline[mid], color="steelblue",
                   ls="--", lw=1.4,
                   label=f"ISTA={ista_baseline[mid]:.3f}")
        ax.axhline(oracle_unseen, color="black",
                   ls=":", lw=1.2,
                   label=f"LS oracle={oracle_unseen:.3f}")
        ax.set_xlabel("Adaptation samples (m={})".format(args.m_unseen))
        ax.set_ylabel("NRMSE")
        ax.set_title(f"Unseen mask {mid}  (m={args.m_unseen})",
                     fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_unseen, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle(
        f"Exp 1b: Adaptation on m={args.m_unseen} operators\n"
        f"(all models pre-trained on m={args.m_seen} only)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Adaptation curves plot -> {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 11. VERDICT
# ──────────────────────────────────────────────────────────────

def print_verdict(seen_res, unseen_res, adapt_curves, pm_ceiling, args):
    print("\n" + "=" * 66)
    print("  EXP 1b VERDICT — Cross-Acceleration Transfer")
    print(f"  Train: m={args.m_seen} (2×)   Test: m={args.m_unseen} (4×)")
    print("=" * 66)

    def avg(results, key):
        vals = [r[key] for r in results.values() if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    # Seen
    print(f"\n  [SEEN MASKS — m={args.m_seen}]")
    print(f"    ISTA top-k+LS  : {avg(seen_res, 'ista_topk_nrmse'):.4f}")
    print(f"    Shared LISTA   : {avg(seen_res, 'shared_lista_nrmse'):.4f}")
    print(f"    Cond LISTA     : {avg(seen_res, 'cond_lista_nrmse'):.4f}")

    # Unseen zero-shot
    ista_u   = avg(unseen_res, "ista_topk_nrmse")
    shared_u = avg(unseen_res, "shared_lista_nrmse")
    cond_u   = avg(unseen_res, "cond_lista_nrmse")
    pm_u     = avg(unseen_res, "pm_lista_nrmse")
    print(f"\n  [UNSEEN MASKS — m={args.m_unseen}, zero-shot]")
    print(f"    ISTA top-k+LS       : {ista_u:.4f}  [parameter-free ceiling]")
    print(f"    Shared LISTA        : {shared_u:.4f}  [no conditioning]")
    print(f"    Cond LISTA          : {cond_u:.4f}  [b_freq conditioned]")
    print(f"    PerMask LISTA m=64  : {pm_u:.4f}  [oracle — trained on m=64]")

    # Gap analysis
    print()
    perf_gap_total = shared_u - pm_u
    perf_gap_cond  = shared_u - cond_u
    if perf_gap_total > 0:
        pct_recovered = 100 * perf_gap_cond / perf_gap_total if perf_gap_total > 1e-6 else 0
        print(f"  Total gap (shared -> oracle m=64): {perf_gap_total:+.4f}")
        print(f"  Gap recovered by conditioning:     {perf_gap_cond:+.4f}"
              f"  ({pct_recovered:.1f}% of total gap)")

    print()
    if cond_u < shared_u - 0.005:
        print("  PASS: CondLISTA zero-shot beats Shared LISTA on m=64 operators.")
        print("  -> b_freq conditioning carries cross-acceleration information.")
        print("     Proceed to Exp 2 (attention block test).")
    elif abs(cond_u - shared_u) <= 0.005:
        print("  INCONCLUSIVE: CondLISTA ties Shared LISTA zero-shot.")
        print("  -> Simple scalar modulation insufficient for cross-m transfer.")
        print("     Consider: per-coordinate lam modulation, or A^T A-based conditioning.")
    else:
        print("  NEGATIVE: Shared LISTA beats CondLISTA zero-shot.")
        print("  -> Conditioning is actively hurting. Check encoder or training.")

    # Adaptation analysis
    print(f"\n  [ADAPTATION — avg NRMSE by budget]")
    budgets = sorted(next(iter(adapt_curves["shared_lista"].values())).keys())
    print(f"    {'Budget':>8}  {'Shared':>9}  {'Cond':>9}  {'Oracle':>9}")
    for b in budgets:
        sh = np.mean([adapt_curves["shared_lista"][mid].get(b, float("nan"))
                      for mid in adapt_curves["shared_lista"]])
        co = np.mean([adapt_curves["cond_lista"][mid].get(b, float("nan"))
                      for mid in adapt_curves["cond_lista"]])
        pm = np.mean(list(pm_ceiling.values()))
        print(f"    {b:>8}  {sh:>9.4f}  {co:>9.4f}  {pm:>9.4f}")

    # Find crossover: N where shared/cond matches oracle
    for method, label in [("shared_lista", "Shared"), ("cond_lista", "Cond")]:
        for b in budgets:
            avg_b = np.mean([adapt_curves[method][mid].get(b, float("nan"))
                             for mid in adapt_curves[method]])
            if avg_b <= pm_u + 0.002:
                print(f"  {label} matches oracle at N={b}")
                break
    print("=" * 66)


# ──────────────────────────────────────────────────────────────
# 12. MAIN
# ──────────────────────────────────────────────────────────────

def main():
    args = get_args()
    dev  = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device       : {dev}")
    print(f"n={args.n}  k={args.k}  T={args.T}  lam={args.lam}")
    print(f"SEEN:   {args.n_seen} masks at m={args.m_seen}  (2x acceleration)")
    print(f"UNSEEN: {args.n_unseen} masks at m={args.m_unseen}  (4x acceleration)")
    print(f"n_train={args.n_train}  n_test={args.n_test}  epochs={args.epochs}")

    # ── Build operators ───────────────────────────────────────
    print("\n── Building operators ───────────────────────────────────")
    seen_ops, unseen_ops, sn_max, alpha = build_operator_sets(args, dev)

    # ── Signals ───────────────────────────────────────────────
    X_train, _       = make_signals(args.n, args.k, args.n_train,
                                     args.amp_lo, args.amp_hi,
                                     seed=args.seed, device=dev)
    X_test,  S_test  = make_signals(args.n, args.k, args.n_test,
                                     args.amp_lo, args.amp_hi,
                                     seed=args.seed + 999, device=dev)

    # ── Train shared LISTA on m=128 ───────────────────────────
    shared_lista = LISTA(args.T, alpha, args.lam, sn_max).to(dev)
    print(f"\nShared LISTA: {sum(p.numel() for p in shared_lista.parameters())} params")
    train_shared_lista(shared_lista, seen_ops, X_train, X_test, args, dev)
    shared_lista.eval()

    # ── Train CondLISTA on m=128 ──────────────────────────────
    cond_lista = CondLISTA(args.T, alpha, args.lam, sn_max,
                           args.n, d_embed=args.d_embed).to(dev)
    print(f"\nCondLISTA: {cond_lista.n_params()} params  "
          f"(+{cond_lista.n_params() - 2*args.T} over LISTA)")
    train_cond_lista(cond_lista, seen_ops, X_train, X_test, args, dev)
    cond_lista.eval()

    # ── Oracle: per-mask LISTA trained on m=64 ────────────────
    pm_lista_m64 = train_pm_lista_m_unseen(
        unseen_ops, X_train, X_test, alpha, args, dev, sn_max
    )
    for m in pm_lista_m64:
        m.eval()

    # ── Evaluate ──────────────────────────────────────────────
    print("\n── Evaluating all methods ────────────────────────────────")

    seen_res = {}
    print(f"\n  Seen masks (m={args.m_seen}):")
    for op in seen_ops:
        mid = op["mask_id"]
        res = eval_on_operator(op, X_test, S_test, alpha, args, dev,
                               shared_lista=shared_lista,
                               cond_lista=cond_lista)
        seen_res[mid] = res
        print(f"  Mask {mid}: oracle={res['oracle_nrmse']:.4f}  "
              f"ista={res['ista_topk_nrmse']:.4f}  "
              f"shared={res['shared_lista_nrmse']:.4f}  "
              f"cond={res['cond_lista_nrmse']:.4f}")

    unseen_res = {}
    print(f"\n  Unseen masks (m={args.m_unseen}, zero-shot):")
    for i, op in enumerate(unseen_ops):
        mid = op["mask_id"]
        res = eval_on_operator(op, X_test, S_test, alpha, args, dev,
                               shared_lista=shared_lista,
                               cond_lista=cond_lista,
                               pm_lista=pm_lista_m64[i])
        unseen_res[mid] = res
        print(f"  Mask {mid}: oracle={res['oracle_nrmse']:.4f}  "
              f"ista={res['ista_topk_nrmse']:.4f}  "
              f"shared={res['shared_lista_nrmse']:.4f}  "
              f"cond={res['cond_lista_nrmse']:.4f}  "
              f"pm_m64={res['pm_lista_nrmse']:.4f}")

    # ── Adaptation curves ─────────────────────────────────────
    adapt_curves = build_adapt_curves(
        unseen_ops, shared_lista, cond_lista,
        X_train, X_test, alpha, args, dev
    )

    # Oracle ceiling for plotting
    pm_ceiling = {
        op["mask_id"]: unseen_res[op["mask_id"]]["pm_lista_nrmse"]
        for op in unseen_ops
    }

    # ── Verdict ───────────────────────────────────────────────
    print_verdict(seen_res, unseen_res, adapt_curves, pm_ceiling, args)

    # ── Save JSON ─────────────────────────────────────────────
    def ser(r):
        out = {}
        for k, v in r.items():
            if isinstance(v, float):
                out[k] = v
            elif isinstance(v, (int, np.floating)):
                out[k] = float(v)
        return out

    serialisable_curves = {
        method: {
            str(mid): {str(b): float(v) for b, v in curve.items()}
            for mid, curve in mask_curves.items()
        }
        for method, mask_curves in adapt_curves.items()
    }

    out_json = os.path.join(args.out_dir, "exp1b_results.json")
    with open(out_json, "w") as fh:
        json.dump({
            "args":           vars(args),
            "alpha":          float(alpha),
            "sn_max":         float(sn_max),
            "n_params_lista": sum(p.numel() for p in shared_lista.parameters()),
            "n_params_cond":  cond_lista.n_params(),
            "oracle_seen":    float(np.mean([r["oracle_nrmse"]
                                             for r in seen_res.values()])),
            "oracle_unseen":  float(np.mean([r["oracle_nrmse"]
                                             for r in unseen_res.values()])),
            "seen_results":   {str(k): ser(v) for k, v in seen_res.items()},
            "unseen_results": {str(k): ser(v) for k, v in unseen_res.items()},
            "adapt_curves":   serialisable_curves,
        }, fh, indent=2)
    print(f"\nResults JSON -> {out_json}")

    # ── Plots ─────────────────────────────────────────────────
    oracle_seen   = float(np.mean([r["oracle_nrmse"] for r in seen_res.values()]))
    oracle_unseen = float(np.mean([r["oracle_nrmse"] for r in unseen_res.values()]))
    ista_baseline = {op["mask_id"]: unseen_res[op["mask_id"]]["ista_topk_nrmse"]
                     for op in unseen_ops}

    plot_summary(seen_res, unseen_res,
                 oracle_seen, oracle_unseen, args,
                 os.path.join(args.out_dir, "exp1b_summary.png"))
    plot_adapt_curves(adapt_curves, pm_ceiling, ista_baseline,
                      oracle_unseen, args,
                      os.path.join(args.out_dir, "exp1b_adapt_curves.png"))


if __name__ == "__main__":
    main()
