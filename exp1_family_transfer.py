"""
exp1_family_transfer.py

Tier-1 de-risking: intra-family transfer across partial Fourier sampling masks.

Family: n=256, m=128 partial Fourier with 10 random masks.
  Seen (train):   masks 0--4
  Unseen (test):  masks 5--9

Methods:
  1. ista_baseline   -- ISTA(T=30,lam=0.05) + naive top-k + LS  [0 learned params]
  2. per_mask_lista  -- one LISTA trained per seen mask independently
  3. shared_lista    -- single LISTA trained jointly on all seen masks
  4. cond_lista      -- shared LISTA + frequency-mask conditioning

Evaluation:
  - Seen masks:       NRMSE on held-out test signals
  - Unseen zero-shot: NRMSE without adaptation
  - Unseen + adapt:   NRMSE after N in {0,10,25,50,100,200} samples per mask

Key question: does conditioning on the frequency mask enable better zero-shot
intra-family transfer than an unconditioned shared baseline?

Usage:
    python exp1_family_transfer.py
    python exp1_family_transfer.py --device cuda --epochs 200

Outputs:
    results_exp1/exp1_results.json
    results_exp1/exp1_summary.png
    results_exp1/exp1_adapt_curves.png
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
    p.add_argument("--n",              type=int,   default=256)
    p.add_argument("--m",              type=int,   default=128)
    p.add_argument("--k",              type=int,   default=25)
    p.add_argument("--amp_lo",         type=float, default=0.5)
    p.add_argument("--amp_hi",         type=float, default=2.0)
    p.add_argument("--T",              type=int,   default=30)
    p.add_argument("--lam",            type=float, default=0.05)
    p.add_argument("--n_masks",        type=int,   default=10,
                   help="Total number of masks (seen + unseen)")
    p.add_argument("--n_seen",         type=int,   default=5,
                   help="Number of seen/training masks")
    p.add_argument("--n_train",        type=int,   default=4000,
                   help="Training signals (shared pool, resampled per mask)")
    p.add_argument("--n_test",         type=int,   default=500,
                   help="Test signals (same signals evaluated on each mask)")
    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--ft_epochs",      type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=128)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--d_embed",        type=int,   default=32,
                   help="CondLISTA operator-embedding dimension")
    p.add_argument("--ft_budgets",     type=int,   nargs="+",
                   default=[0, 10, 25, 50, 100, 200])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--mask_seed_base", type=int,   default=100,
                   help="Mask i uses seed = mask_seed_base + i*100")
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",        type=str,   default="results_exp1")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2. OPERATOR / MASK GENERATION
# ──────────────────────────────────────────────────────────────

def make_partial_fourier(n, m, mask_seed, device):
    """
    Generate one partial Fourier operator and its frequency-indicator vector.

    Returns:
        A        : (m, n) float32 tensor  — the sensing matrix
        b_freq   : (n,)   float32 tensor  — binary, 1 at sampled frequency bins
        freq_idx : (m//2,) int array      — sampled DFT row indices
    """
    rng = np.random.RandomState(mask_seed)
    n_freqs  = m // 2                               # each complex row -> 2 real rows
    freq_idx = np.sort(rng.choice(n, n_freqs, replace=False))

    F_complex = np.fft.fft(np.eye(n)) / np.sqrt(n) # (n, n) complex DFT
    rows_real = F_complex[freq_idx].real            # (n_freqs, n)
    rows_imag = F_complex[freq_idx].imag            # (n_freqs, n)
    rows      = np.concatenate([rows_real, rows_imag], axis=0)  # (m, n)
    A = torch.tensor(rows[:m], dtype=torch.float32, device=device)

    b_freq = torch.zeros(n, dtype=torch.float32, device=device)
    b_freq[freq_idx] = 1.0

    return A, b_freq, freq_idx


def make_mask_family(n, m, n_masks, mask_seed_base, device):
    """
    Build a family of n_masks partial Fourier operators.

    Returns:
        operators : list of dicts, each with keys 'A', 'b_freq', 'freq_idx', 'mask_id'
        sn_max    : float — max spectral norm squared across all operators (for LISTA)
    """
    operators = []
    sn_max    = 0.0
    for i in range(n_masks):
        seed = mask_seed_base + i * 100
        A, b_freq, freq_idx = make_partial_fourier(n, m, seed, device)
        sn_i   = torch.linalg.norm(A, ord=2).item() ** 2
        sn_max = max(sn_max, sn_i)
        operators.append({
            "A": A, "b_freq": b_freq, "freq_idx": freq_idx,
            "mask_id": i, "sn": sn_i
        })
    return operators, sn_max


# ──────────────────────────────────────────────────────────────
# 3. DATA GENERATION
# ──────────────────────────────────────────────────────────────

def make_signals(n, k, n_signals, amp_lo, amp_hi, seed, device):
    rng = np.random.RandomState(seed)
    X   = np.zeros((n_signals, n), dtype=np.float32)
    S   = np.zeros((n_signals, n), dtype=np.float32)
    for i in range(n_signals):
        supp     = rng.choice(n, k, replace=False)
        amps     = rng.uniform(amp_lo, amp_hi, k) * rng.choice([-1, 1], k)
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
    Learned ISTA: per-step scalar step size and threshold.
    alpha clamped to [eps, 2/sn_max] for stability across all operators.
    Parameters: 2*T (e.g., 60 for T=30).
    """
    def __init__(self, T, alpha_init, lam_init, sn_max):
        super().__init__()
        self.T         = T
        self.alpha_max = float(2.0 / sn_max)
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))

    def forward(self, A, y):
        """A: (m, n),  y: (B, m)  ->  x_hat: (B, n)"""
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
    LISTA conditioned on the frequency-mask indicator vector b_freq in {0,1}^n.

    Architecture:
        op_encoder : Linear(n, d_embed) -> Tanh  (shared across steps)
        alpha_mod  : Linear(d_embed, T, bias=False)  (scalar delta per step)
        lam_mod    : Linear(d_embed, T, bias=False)  (scalar delta per step)

    At each unrolling step t:
        alpha_t = (alpha[t] + alpha_mod(e)[t]).clamp(...)
        lam_t   = (lam[t]   + lam_mod(e)[t]).clamp(...)

    Both modulation layers are zero-initialized so the model starts as plain LISTA.
    b_freq is the same for every signal in a batch (operator-level, not sample-level).

    Extra parameters beyond LISTA: n*d_embed + d_embed (encoder) + 2*d_embed*T (mods)
    For n=256, d_embed=32, T=30: 256*32+32 + 2*32*30 = 8192+32+1920 = 10,144 extra params.
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
        """
        A      : (m, n)
        y      : (B, m)
        b_freq : (n,)  binary frequency-mask indicator — same for whole batch
        """
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


# ──────────────────────────────────────────────────────────────
# 6. METRICS
# ──────────────────────────────────────────────────────────────

def raw_nrmse(x_hat, x_true):
    num   = (x_hat - x_true).norm(dim=1)
    denom = x_true.norm(dim=1).clamp(min=1e-8)
    return (num / denom).mean().item()


def topk_ls_nrmse(scores, x_true, A, k, device):
    """Top-k of |scores| -> LS amplitude correction -> NRMSE."""
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
    y = x_true @ A.T
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

def train_lista_single_mask(lista, op, X_train, X_test, alpha, args, dev, label=""):
    """Train one LISTA on a single operator. Returns test NRMSE history."""
    optimizer = optim.Adam(lista.parameters(), lr=args.lr)
    A = op["A"]
    n_train = X_train.shape[0]
    history = []

    for epoch in range(args.epochs):
        lista.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            idx  = perm[start: start + args.batch_size]
            X_b  = X_train[idx]
            y    = X_b @ A.T
            loss = nn.functional.mse_loss(lista(A, y), X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            lista.eval()
            with torch.no_grad():
                nrmse = raw_nrmse(lista(A, X_test @ A.T), X_test)
            history.append(nrmse)
            print(f"  [{label}] Epoch {epoch+1:>4}  NRMSE={nrmse:.4f}")

    return history


def train_shared_lista(lista, seen_ops, X_train, X_test, alpha, args, dev):
    """
    Train one LISTA jointly on all seen operators.
    Each batch randomly picks one seen mask.
    """
    optimizer = optim.Adam(lista.parameters(), lr=args.lr)
    n_seen  = len(seen_ops)
    n_train = X_train.shape[0]
    rng     = np.random.RandomState(args.seed + 1)
    history = []

    print("\n── Shared LISTA training ─────────────────────────────────")
    for epoch in range(args.epochs):
        lista.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            idx   = perm[start: start + args.batch_size]
            X_b   = X_train[idx]
            mask_i = int(rng.randint(n_seen))
            A      = seen_ops[mask_i]["A"]
            y      = X_b @ A.T
            loss   = nn.functional.mse_loss(lista(A, y), X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            lista.eval()
            avg_nrmse = np.mean([
                raw_nrmse(lista(op["A"], X_test @ op["A"].T).detach(), X_test)
                for op in seen_ops
            ])
            history.append(avg_nrmse)
            print(f"  Epoch {epoch+1:>4}  avg-seen-NRMSE={avg_nrmse:.4f}")

    return history


def train_cond_lista(cond_lista, seen_ops, X_train, X_test, alpha, args, dev):
    """
    Train CondLISTA jointly on all seen operators, passing b_freq each step.
    """
    optimizer = optim.Adam(cond_lista.parameters(), lr=args.lr)
    n_seen  = len(seen_ops)
    n_train = X_train.shape[0]
    rng     = np.random.RandomState(args.seed + 2)
    history = []

    print("\n── CondLISTA training ────────────────────────────────────")
    for epoch in range(args.epochs):
        cond_lista.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            idx    = perm[start: start + args.batch_size]
            X_b    = X_train[idx]
            mask_i = int(rng.randint(n_seen))
            op     = seen_ops[mask_i]
            A, b_freq = op["A"], op["b_freq"]
            y      = X_b @ A.T
            loss   = nn.functional.mse_loss(cond_lista(A, y, b_freq), X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            cond_lista.eval()
            avg_nrmse = np.mean([
                raw_nrmse(
                    cond_lista(op["A"], X_test @ op["A"].T, op["b_freq"]).detach(),
                    X_test
                )
                for op in seen_ops
            ])
            history.append(avg_nrmse)
            print(f"  Epoch {epoch+1:>4}  avg-seen-NRMSE={avg_nrmse:.4f}")

    return history


# ──────────────────────────────────────────────────────────────
# 8. EVALUATION — SEEN AND ZERO-SHOT
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_on_operator(op, x_true, S_true, alpha, args, dev,
                     lista=None, shared_lista=None, cond_lista=None,
                     per_mask_listas=None):
    """
    Compute NRMSE for all methods on a single operator.
    per_mask_listas: list of LISTA models (one per seen mask); for zero-shot on
    unseen masks, we average their predictions (no oracle picking).
    """
    A      = op["A"]
    b_freq = op["b_freq"]
    k      = args.k

    y    = x_true @ A.T
    xT   = ista_unroll(A, y, alpha, args.lam, args.T)
    res  = {}

    # ISTA baseline
    res["ista_nrmse"]    = raw_nrmse(xT, x_true)
    res["ista_topk_nrmse"] = topk_ls_nrmse(xT, x_true, A, k, dev)
    res["oracle_nrmse"]  = oracle_ls_nrmse(x_true, A, S_true, dev)

    # Per-mask LISTA (average over all seen models)
    if per_mask_listas is not None:
        preds = [m(A, y) for m in per_mask_listas]
        x_avg = torch.stack(preds, dim=0).mean(dim=0)
        res["per_mask_lista_nrmse"] = raw_nrmse(x_avg, x_true)
        # Also individual model NRMSEs (for logging)
        res["per_mask_lista_individual"] = [raw_nrmse(p, x_true) for p in preds]

    # Single LISTA (if provided — per-mask case for seen masks)
    if lista is not None:
        res["lista_nrmse"] = raw_nrmse(lista(A, y), x_true)

    # Shared LISTA
    if shared_lista is not None:
        res["shared_lista_nrmse"] = raw_nrmse(shared_lista(A, y), x_true)

    # CondLISTA
    if cond_lista is not None:
        res["cond_lista_nrmse"] = raw_nrmse(cond_lista(A, y, b_freq), x_true)

    return res


# ──────────────────────────────────────────────────────────────
# 9. ADAPTATION CURVES
# ──────────────────────────────────────────────────────────────

def adapt_curve_for_op(model_init, op, X_pool, X_test, args, dev,
                       model_type="lista", label=""):
    """
    Fine-tune model_init on N samples from op, evaluate on X_test.
    Returns dict: budget -> NRMSE.
    Returns budget=0 as zero-shot.
    model_type: "lista" | "cond_lista"
    """
    A      = op["A"]
    b_freq = op["b_freq"]

    # Zero-shot
    base = copy.deepcopy(model_init).eval()
    with torch.no_grad():
        if model_type == "cond_lista":
            zs_nrmse = raw_nrmse(base(A, X_test @ A.T, b_freq), X_test)
        else:
            zs_nrmse = raw_nrmse(base(A, X_test @ A.T), X_test)

    results = {0: zs_nrmse}

    for N in args.ft_budgets:
        if N == 0:
            continue
        N_actual = min(N, X_pool.shape[0])
        X_ft     = X_pool[:N_actual]

        model_ft  = copy.deepcopy(model_init)
        optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)

        for epoch in range(args.ft_epochs):
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


def build_adapt_curves(seen_ops, unseen_ops,
                       per_mask_listas, shared_lista, cond_lista,
                       X_train, X_test, alpha, args, dev):
    """
    For each unseen mask, build adaptation curves for shared_lista and cond_lista.
    For per_mask_lista: start from the seen-mask model with best seen NRMSE.
    Returns nested dict: method -> mask_id -> budget -> nrmse
    """
    print("\n── Adaptation curves (unseen masks) ─────────────────────")

    # Find best per-mask model (lowest avg seen-mask NRMSE)
    best_pm_idx = 0
    best_pm_nrmse = float("inf")
    for i, m in enumerate(per_mask_listas):
        avg = np.mean([
            raw_nrmse(m(op["A"], X_test @ op["A"].T).detach(), X_test)
            for op in seen_ops
        ])
        if avg < best_pm_nrmse:
            best_pm_nrmse = avg
            best_pm_idx   = i
    best_pm_model = per_mask_listas[best_pm_idx]
    print(f"  Per-mask LISTA: using seen-mask model {best_pm_idx} "
          f"as adaptation init (seen avg NRMSE={best_pm_nrmse:.4f})")

    curves = {
        "per_mask_lista": {},
        "shared_lista":   {},
        "cond_lista":     {},
    }

    for op in unseen_ops:
        mid = op["mask_id"]
        print(f"\n  Mask {mid} (unseen):")

        print("    per_mask_lista:")
        curves["per_mask_lista"][mid] = adapt_curve_for_op(
            best_pm_model, op, X_train, X_test, args, dev,
            model_type="lista", label="per_mask"
        )

        print("    shared_lista:")
        curves["shared_lista"][mid] = adapt_curve_for_op(
            shared_lista, op, X_train, X_test, args, dev,
            model_type="lista", label="shared"
        )

        print("    cond_lista:")
        curves["cond_lista"][mid] = adapt_curve_for_op(
            cond_lista, op, X_train, X_test, args, dev,
            model_type="cond_lista", label="cond"
        )

    return curves


# ──────────────────────────────────────────────────────────────
# 10. PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_summary(seen_results, unseen_results, oracle_nrmse, args, out_path):
    """4-panel summary: bar charts for seen + unseen, per-mask breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    methods = ["ista_topk_nrmse", "per_mask_lista_nrmse",
               "shared_lista_nrmse", "cond_lista_nrmse"]
    labels  = ["ISTA\ntop-k+LS", "PerMask\nLISTA", "Shared\nLISTA", "Cond\nLISTA"]
    colors  = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for ax, (results, title) in zip(axes, [
        (seen_results,   "Seen masks (avg NRMSE)"),
        (unseen_results, "Unseen masks — zero-shot (avg NRMSE)")
    ]):
        vals = [
            float(np.mean([r[m] for r in results.values() if m in r]))
            for m in methods
        ]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.axhline(oracle_nrmse, color="black", ls=":", lw=1.5, label=f"oracle LS={oracle_nrmse:.3f}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Exp 1: Intra-Family Transfer (Partial Fourier, n={args.n}, m={args.m}, k={args.k})\n"
        f"Seen masks: 0--{args.n_seen-1}  |  Unseen masks: {args.n_seen}--{args.n_masks-1}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSummary plot -> {out_path}")
    plt.close()


def plot_adapt_curves(curves, ista_baseline, oracle_nrmse, args, out_path):
    """Adaptation curves for unseen masks: one subplot per unseen mask."""
    unseen_ids = sorted(curves["shared_lista"].keys())
    n_unseen   = len(unseen_ids)
    ncols = min(n_unseen, 3)
    nrows = (n_unseen + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    style = {
        "per_mask_lista": ("darkorange", "s-", "PerMask LISTA"),
        "shared_lista":   ("forestgreen", "o-", "Shared LISTA"),
        "cond_lista":     ("crimson", "^-", "Cond LISTA"),
    }

    budgets = sorted({b for m_curves in curves["shared_lista"].values()
                      for b in m_curves.keys()})

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

        ax.axhline(ista_baseline[mid], color="steelblue",
                   ls="--", lw=1.5, label=f"ISTA top-k={ista_baseline[mid]:.3f}")
        ax.axhline(oracle_nrmse, color="black",
                   ls=":", lw=1.5, label=f"oracle={oracle_nrmse:.3f}")
        ax.set_xlabel("Adaptation samples")
        ax.set_ylabel("NRMSE")
        ax.set_title(f"Unseen mask {mid}", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_unseen, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle(
        "Exp 1: Adaptation curves on unseen partial Fourier masks",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Adaptation curves plot -> {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 11. VERDICT PRINTING
# ──────────────────────────────────────────────────────────────

def print_verdict(seen_res, unseen_res, adapt_curves, args):
    methods = ["ista_topk_nrmse", "per_mask_lista_nrmse",
               "shared_lista_nrmse", "cond_lista_nrmse"]
    names   = ["ISTA top-k+LS", "PerMask LISTA", "Shared LISTA", "Cond LISTA"]

    print("\n" + "=" * 66)
    print("  EXP 1 VERDICT — Intra-Family Partial Fourier Transfer")
    print("=" * 66)

    print("\n  [SEEN MASKS — avg NRMSE]")
    for m, name in zip(methods, names):
        vals = [r[m] for r in seen_res.values() if m in r]
        if vals:
            print(f"    {name:<22}: {np.mean(vals):.4f}")

    print("\n  [UNSEEN MASKS — zero-shot avg NRMSE]")
    for m, name in zip(methods, names):
        vals = [r[m] for r in unseen_res.values() if m in r]
        if vals:
            print(f"    {name:<22}: {np.mean(vals):.4f}")

    # Crossover: N samples for per-mask LISTA to match cond_lista zero-shot
    print("\n  [ADAPTATION — avg NRMSE by budget]")
    budgets = sorted(next(iter(adapt_curves["cond_lista"].values())).keys())
    shared_zs = np.mean([adapt_curves["shared_lista"][mid][0]
                         for mid in adapt_curves["shared_lista"]])
    cond_zs   = np.mean([adapt_curves["cond_lista"][mid][0]
                         for mid in adapt_curves["cond_lista"]])
    pm_zs     = np.mean([adapt_curves["per_mask_lista"][mid][0]
                         for mid in adapt_curves["per_mask_lista"]])

    print(f"    {'Budget':>8}  {'PerMask':>9}  {'Shared':>9}  {'Cond':>9}")
    for b in budgets:
        pm  = np.mean([adapt_curves["per_mask_lista"][mid].get(b, float("nan"))
                       for mid in adapt_curves["per_mask_lista"]])
        sh  = np.mean([adapt_curves["shared_lista"][mid].get(b, float("nan"))
                       for mid in adapt_curves["shared_lista"]])
        co  = np.mean([adapt_curves["cond_lista"][mid].get(b, float("nan"))
                       for mid in adapt_curves["cond_lista"]])
        print(f"    {b:>8}  {pm:>9.4f}  {sh:>9.4f}  {co:>9.4f}")

    print()
    if cond_zs < shared_zs - 0.005:
        print("  PASS: CondLISTA zero-shot beats Shared LISTA zero-shot.")
        print(f"        Gain = {shared_zs - cond_zs:.4f} NRMSE -> "
              "conditioning carries transferable operator info.")
    elif abs(cond_zs - shared_zs) <= 0.005:
        print("  INCONCLUSIVE: CondLISTA ties Shared LISTA zero-shot.")
        print("        Frequency-mask conditioning does not improve over shared.")
    else:
        print("  FAIL: Shared LISTA zero-shot beats CondLISTA.")
        print("        Conditioning hurts — check encoder or training diversity.")

    print()
    if pm_zs > shared_zs + 0.01:
        print("  Per-mask LISTA zero-shot is worse than Shared LISTA.")
        print("  -> Shared training already encodes intra-family structure.")
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
    print(f"n={args.n}  m={args.m}  k={args.k}")
    print(f"T={args.T}   lam={args.lam}")
    print(f"Masks total={args.n_masks}  seen={args.n_seen}  "
          f"unseen={args.n_masks - args.n_seen}")
    print(f"n_train={args.n_train}  n_test={args.n_test}  epochs={args.epochs}")

    # ── Build mask family ────────────────────────────────────
    print("\n── Building partial Fourier mask family ─────────────────")
    operators, sn_max = make_mask_family(
        args.n, args.m, args.n_masks, args.mask_seed_base, dev
    )
    alpha = 1.0 / sn_max
    seen_ops   = operators[:args.n_seen]
    unseen_ops = operators[args.n_seen:]

    for op in operators:
        tag = "seen  " if op["mask_id"] < args.n_seen else "unseen"
        print(f"  Mask {op['mask_id']}  [{tag}]  "
              f"sn={op['sn']:.4f}  "
              f"freq_idx[0:3]={op['freq_idx'][:3]}")
    print(f"  sn_max={sn_max:.4f}   alpha={alpha:.6f}")

    # ── Data ─────────────────────────────────────────────────
    X_train, S_train = make_signals(
        args.n, args.k, args.n_train,
        args.amp_lo, args.amp_hi, seed=args.seed, device=dev
    )
    X_test, S_test = make_signals(
        args.n, args.k, args.n_test,
        args.amp_lo, args.amp_hi, seed=args.seed + 999, device=dev
    )

    # ── Per-mask LISTA training ───────────────────────────────
    print("\n── Per-mask LISTA training ───────────────────────────────")
    per_mask_listas = []
    for op in seen_ops:
        mid   = op["mask_id"]
        lista = LISTA(args.T, alpha, args.lam, sn_max).to(dev)
        print(f"\n  Mask {mid}  ({sum(p.numel() for p in lista.parameters())} params)")
        train_lista_single_mask(
            lista, op, X_train, X_test, alpha, args, dev,
            label=f"mask{mid}"
        )
        lista.eval()
        per_mask_listas.append(lista)

    # ── Shared LISTA training ─────────────────────────────────
    shared_lista = LISTA(args.T, alpha, args.lam, sn_max).to(dev)
    print(f"\nShared LISTA ({sum(p.numel() for p in shared_lista.parameters())} params)")
    train_shared_lista(shared_lista, seen_ops, X_train, X_test, alpha, args, dev)
    shared_lista.eval()

    # ── CondLISTA training ────────────────────────────────────
    cond_lista = CondLISTA(
        args.T, alpha, args.lam, sn_max, args.n, d_embed=args.d_embed
    ).to(dev)
    n_cond = sum(p.numel() for p in cond_lista.parameters())
    print(f"\nCondLISTA ({n_cond} params,  "
          f"extra vs LISTA: {n_cond - 2*args.T})")
    train_cond_lista(cond_lista, seen_ops, X_train, X_test, alpha, args, dev)
    cond_lista.eval()

    # ── Oracle NRMSE (same for all operators, signals only) ──
    # Use the first seen operator as reference; oracle is operator-dependent.
    # We report the average across all operators.
    oracle_seen = float(np.mean([
        oracle_ls_nrmse(X_test, op["A"], S_test, dev) for op in seen_ops
    ]))
    oracle_unseen = float(np.mean([
        oracle_ls_nrmse(X_test, op["A"], S_test, dev) for op in unseen_ops
    ]))
    oracle_avg = (oracle_seen + oracle_unseen) / 2
    print(f"\n  Oracle LS NRMSE: seen={oracle_seen:.4f}  "
          f"unseen={oracle_unseen:.4f}")

    # ── Evaluate on all masks ─────────────────────────────────
    print("\n── Evaluating all methods ────────────────────────────────")
    seen_results   = {}
    unseen_results = {}

    print("\n  Seen masks:")
    for op in seen_ops:
        mid  = op["mask_id"]
        m_lista = per_mask_listas[mid]   # the model trained on this exact mask
        res = eval_on_operator(
            op, X_test, S_test, alpha, args, dev,
            lista=m_lista,
            shared_lista=shared_lista,
            cond_lista=cond_lista,
            per_mask_listas=per_mask_listas
        )
        seen_results[mid] = res
        print(f"  Mask {mid}: oracle={res['oracle_nrmse']:.4f}  "
              f"ista={res['ista_topk_nrmse']:.4f}  "
              f"per_mask={res.get('lista_nrmse', res.get('per_mask_lista_nrmse', float('nan'))):.4f}  "
              f"shared={res.get('shared_lista_nrmse', float('nan')):.4f}  "
              f"cond={res.get('cond_lista_nrmse', float('nan')):.4f}")

    print("\n  Unseen masks (zero-shot):")
    for op in unseen_ops:
        mid = op["mask_id"]
        res = eval_on_operator(
            op, X_test, S_test, alpha, args, dev,
            shared_lista=shared_lista,
            cond_lista=cond_lista,
            per_mask_listas=per_mask_listas
        )
        unseen_results[mid] = res
        print(f"  Mask {mid}: oracle={res['oracle_nrmse']:.4f}  "
              f"ista={res['ista_topk_nrmse']:.4f}  "
              f"per_mask(avg)={res.get('per_mask_lista_nrmse', float('nan')):.4f}  "
              f"shared={res.get('shared_lista_nrmse', float('nan')):.4f}  "
              f"cond={res.get('cond_lista_nrmse', float('nan')):.4f}")

    # ── Adaptation curves ─────────────────────────────────────
    adapt_curves = build_adapt_curves(
        seen_ops, unseen_ops,
        per_mask_listas, shared_lista, cond_lista,
        X_train, X_test, alpha, args, dev
    )

    # ── Verdict ───────────────────────────────────────────────
    print_verdict(seen_results, unseen_results, adapt_curves, args)

    # ── Save JSON ─────────────────────────────────────────────
    def detach_result(r):
        return {k: (v if not isinstance(v, (list, dict)) else v)
                for k, v in r.items()
                if not isinstance(v, torch.Tensor)}

    out_json = os.path.join(args.out_dir, "exp1_results.json")
    serialisable_curves = {}
    for method, mask_curves in adapt_curves.items():
        serialisable_curves[method] = {
            str(mid): {str(b): float(v) for b, v in curve.items()}
            for mid, curve in mask_curves.items()
        }
    with open(out_json, "w") as fh:
        json.dump({
            "args":           vars(args),
            "alpha":          float(alpha),
            "sn_max":         float(sn_max),
            "oracle_seen":    oracle_seen,
            "oracle_unseen":  oracle_unseen,
            "seen_results":   {str(k): detach_result(v) for k, v in seen_results.items()},
            "unseen_results": {str(k): detach_result(v) for k, v in unseen_results.items()},
            "adapt_curves":   serialisable_curves,
        }, fh, indent=2)
    print(f"\nResults JSON -> {out_json}")

    # ── Plots ─────────────────────────────────────────────────
    ista_baseline_per_mask = {
        mid: res["ista_topk_nrmse"] for mid, res in unseen_results.items()
    }
    plot_summary(
        seen_results, unseen_results, oracle_avg, args,
        os.path.join(args.out_dir, "exp1_summary.png")
    )
    plot_adapt_curves(
        adapt_curves, ista_baseline_per_mask, oracle_avg, args,
        os.path.join(args.out_dir, "exp1_adapt_curves.png")
    )


if __name__ == "__main__":
    main()
