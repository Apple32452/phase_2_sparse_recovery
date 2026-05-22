"""
exp2_cross_family.py

Cross-family transfer: train on partial Fourier (m=128), test zero-shot on Gaussian (m=128).

This is the proper architecture test. Phase 1 established:
  - naive top-k+LS (zero-shot):  NRMSE = 0.257   (Phase 1 baseline)
  - LISTA zero-shot Gaussian:     NRMSE = 0.327   (Phase 1 baseline)
  - LISTA catches up at N ≈ 50 Gaussian samples

This experiment asks two questions:

  Q1 (CondLISTA): Does conditioning on the eigenvalue spectrum of A^T A let
     CondLISTA — trained only on Fourier — exceed LISTA zero-shot (0.327)?
     The eigenvalue spectrum is fundamentally different between families:
       Fourier: concentrated near m/n (RIP near-isometry)
       Gaussian: Marchenko-Pastur spread, ratio m/n = 0.5
     The encoder sees this difference at inference time even though it was
     trained only on Fourier eigenvalue spectra.

  Q2 (AttnReconstructor): Does adding A^T A as a positional bias to attention
     over ISTA iterates improve cross-family zero-shot transfer beyond
     CondLISTA and beyond simple attention?
     At inference on Gaussian, the A^T A positional bias is structurally
     different (approximately scaled-identity) from the Fourier A^T A
     (structured, near-circulant). The test is whether the model has learned
     to use A^T A adaptively — not just memorized Fourier-specific routing.

Training: 5 partial Fourier operators (m=128), 4000 signals.
Test:     5 Gaussian operators (m=128), zero-shot + adaptation curves.

Methods:
  1. ista_baseline   -- ISTA(T=30,lam=0.05) + top-k+LS         [0 params]
  2. lista           -- LISTA(T=30) trained on Fourier family   [60 params]
  3. cond_lista      -- CondLISTA, eigenvalue(A^T A) conditioned [~18k params]
  4. attn_no_bias    -- AttnReconstructor on x_T, no A^T A bias [~14k params]
  5. attn_ata_bias   -- AttnReconstructor on x_T, A^T A bias    [~14k params]

Adaptation: fine-tune each on N in {0,10,25,50,100,200,500} Gaussian samples.
The crossover point (where method matches naive top-k NRMSE = 0.257) is the
key sample-efficiency metric.

Usage:
    python exp2_cross_family.py
    python exp2_cross_family.py --device cuda

Outputs:
    results_exp2_cf/exp2_cf_results.json
    results_exp2_cf/exp2_cf_summary.png
    results_exp2_cf/exp2_cf_adapt_curves.png
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
    p.add_argument("--n",              type=int,   default=256)
    p.add_argument("--m",              type=int,   default=128)
    p.add_argument("--k",              type=int,   default=25)
    p.add_argument("--amp_lo",         type=float, default=0.5)
    p.add_argument("--amp_hi",         type=float, default=2.0)
    p.add_argument("--T",              type=int,   default=30)
    p.add_argument("--lam",            type=float, default=0.05)
    p.add_argument("--n_fourier",      type=int,   default=5,
                   help="Number of Fourier training operators")
    p.add_argument("--n_gaussian",     type=int,   default=5,
                   help="Number of Gaussian test operators")
    p.add_argument("--n_train",        type=int,   default=4000)
    p.add_argument("--n_test",         type=int,   default=500)
    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--ft_epochs",      type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--d_embed",        type=int,   default=32,
                   help="CondLISTA eigenvalue-encoder output dim")
    p.add_argument("--d_model",        type=int,   default=32,
                   help="Attention model dimension (d_model % n_heads == 0)")
    p.add_argument("--n_heads",        type=int,   default=4)
    p.add_argument("--n_layers",       type=int,   default=2)
    p.add_argument("--d_ff",           type=int,   default=64)
    p.add_argument("--ft_budgets",     type=int,   nargs="+",
                   default=[0, 10, 25, 50, 100, 200, 500])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--fourier_seed_base", type=int, default=100,
                   help="Fourier op i: seed = fourier_seed_base + i*100")
    p.add_argument("--gaussian_seed_base", type=int, default=900,
                   help="Gaussian op i: seed = gaussian_seed_base + i*100")
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",        type=str,   default="results_exp2_cf")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2. OPERATORS
# ──────────────────────────────────────────────────────────────

def make_fourier_op(n, m, seed, device):
    """Partial Fourier operator — identical construction to Phase 1."""
    rng      = np.random.RandomState(seed)
    freq_idx = np.sort(rng.choice(n, m // 2, replace=False))
    F        = np.fft.fft(np.eye(n)) / np.sqrt(n)
    rows     = np.concatenate([F[freq_idx].real, F[freq_idx].imag], axis=0)
    A        = torch.tensor(rows[:m], dtype=torch.float32, device=device)
    return A


def make_gaussian_op(n, m, seed, device):
    """Gaussian N(0, 1/m) operator — identical to Phase 1."""
    rng = np.random.RandomState(seed)
    A   = torch.tensor(rng.randn(m, n) / np.sqrt(m),
                       dtype=torch.float32, device=device)
    return A


@torch.no_grad()
def eig_descriptor(A):
    """
    Sorted eigenvalues of A^T A — natural operator fingerprint.

    For partial Fourier (RIP): eigenvalues concentrated near m/n, small variance.
    For Gaussian (Marchenko-Pastur, ratio m/n=0.5):
        eigenvalues in [(1-sqrt(0.5))^2, (1+sqrt(0.5))^2] = [0.086, 2.914]
        with a broad continuous distribution — fundamentally different shape.

    This difference is the signal CondLISTA's encoder must detect at inference.
    Returns: (n,) float32 tensor, eigenvalues in descending order.
    """
    AtA     = A.T @ A                       # (n, n)
    eigvals = torch.linalg.eigvalsh(AtA)    # (n,) ascending, real
    return eigvals.flip(0)                  # descending


def build_operators(args, device):
    """
    Build Fourier (seen) and Gaussian (unseen) operator sets.
    Each operator dict has: A, AtA, eigvals, sn, label.
    """
    fourier_ops, gaussian_ops = [], []
    sn_max = 0.0

    print(f"  Fourier operators (m={args.m}):")
    for i in range(args.n_fourier):
        A    = make_fourier_op(args.n, args.m,
                               args.fourier_seed_base + i * 100, device)
        sn   = torch.linalg.norm(A, ord=2).item() ** 2
        sn_max = max(sn_max, sn)
        with torch.no_grad():
            AtA  = A.T @ A
            eigv = eig_descriptor(A)
        fourier_ops.append({"A": A, "AtA": AtA, "eigvals": eigv,
                             "sn": sn, "id": i, "family": "fourier"})
        print(f"    F{i}  sn={sn:.4f}  "
              f"eig[max={eigv[0]:.3f}, min={eigv[-1]:.3f}, "
              f"std={eigv.std().item():.4f}]")

    print(f"  Gaussian operators (m={args.m}):")
    for i in range(args.n_gaussian):
        A    = make_gaussian_op(args.n, args.m,
                                args.gaussian_seed_base + i * 100, device)
        sn   = torch.linalg.norm(A, ord=2).item() ** 2
        sn_max = max(sn_max, sn)
        with torch.no_grad():
            AtA  = A.T @ A
            eigv = eig_descriptor(A)
        gaussian_ops.append({"A": A, "AtA": AtA, "eigvals": eigv,
                              "sn": sn, "id": args.n_fourier + i,
                              "family": "gaussian"})
        print(f"    G{i}  sn={sn:.4f}  "
              f"eig[max={eigv[0]:.3f}, min={eigv[-1]:.3f}, "
              f"std={eigv.std().item():.4f}]")

    alpha = 1.0 / sn_max
    print(f"  sn_max={sn_max:.4f}   alpha={alpha:.6f}")
    return fourier_ops, gaussian_ops, sn_max, alpha


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
    Standard LISTA: per-step scalar alpha and lam. Parameters: 2*T.
    Trained on Fourier, applied zero-shot to Gaussian.
    Phase 1 established zero-shot Gaussian NRMSE ≈ 0.327 for this baseline.
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
            x = soft_threshold(x - alpha_t * (residual @ A),
                               self.lam[t].clamp(min=0.0))
        return x


class CondLISTA(nn.Module):
    """
    LISTA conditioned on the sorted eigenvalue spectrum of A^T A.

    The eigenvalue spectrum is fundamentally different across families:
      Fourier training: eigenvalues in a narrow band near m/n (RIP)
      Gaussian inference: Marchenko-Pastur spread, same mean but very different shape

    The encoder maps (n,) eigenvalue vector -> (d_embed,) features.
    Per-step modulation layers produce additive corrections to alpha_t and lam_t.
    Zero-initialized modulation: model starts identical to plain LISTA.

    Caveats:
      - Trained only on Fourier eigenvalue spectra
      - At Gaussian inference, gets a strongly OOD input to the encoder
      - This tests whether the eigenvalue geometry generalizes; expect a
        conservative result (likely ties or slightly beats LISTA zero-shot)
    """
    def __init__(self, T, alpha_init, lam_init, sn_max, n, d_embed=32):
        super().__init__()
        self.T         = T
        self.alpha_max = float(2.0 / sn_max)
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))

        # Eigenvalue encoder: sorted eig(A^T A) -> compact descriptor
        # Input normalized by n to keep values in [0, ~3] range
        self.op_encoder = nn.Sequential(
            nn.Linear(n, d_embed),
            nn.Tanh(),
            nn.Linear(d_embed, d_embed),
            nn.Tanh(),
        )
        self.alpha_mod = nn.Linear(d_embed, T, bias=False)
        self.lam_mod   = nn.Linear(d_embed, T, bias=False)
        nn.init.zeros_(self.alpha_mod.weight)
        nn.init.zeros_(self.lam_mod.weight)

    def forward(self, A, y, eigvals):
        """
        A       : (m, n)
        y       : (B, m)
        eigvals : (n,) sorted eigenvalues of A^T A
        """
        e           = self.op_encoder(eigvals)   # (d_embed,)
        alpha_delta = self.alpha_mod(e)           # (T,)
        lam_delta   = self.lam_mod(e)             # (T,)
        x = torch.zeros(y.shape[0], A.shape[1], device=y.device)
        for t in range(self.T):
            residual = x @ A.T - y
            alpha_t  = (self.alpha[t] + alpha_delta[t]).clamp(
                min=1e-6, max=self.alpha_max)
            lam_t    = (self.lam[t] + lam_delta[t]).clamp(min=0.0)
            x = soft_threshold(x - alpha_t * (residual @ A), lam_t)
        return x


# ── Attention components ──────────────────────────────────────

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, pos_bias=None):
        """X: (B, n, d); pos_bias: (n, n) additive bias on logits."""
        B, n, _ = X.shape
        H, dk   = self.n_heads, self.d_k
        Q = self.W_q(X).view(B, n, H, dk).transpose(1, 2)
        K = self.W_k(X).view(B, n, H, dk).transpose(1, 2)
        V = self.W_v(X).view(B, n, H, dk).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)       # (B, H, n, n)
        if pos_bias is not None:
            scores = scores + pos_bias.unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, n, H * dk)
        return self.W_o(out)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn  = MultiHeadSelfAttn(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, X, pos_bias=None):
        X = X + self.attn(self.norm1(X), pos_bias)
        X = X + self.ff(self.norm2(X))
        return X


class AttnReconstructor(nn.Module):
    """
    Attention-based reconstructor operating on ISTA iterates.

    Input:  x_T (B, n) — ISTA iterate after T steps on the target operator
    Output: x_T + correction (B, n) — residual refinement

    If use_ata_bias=True, the attention logits in each layer receive an
    additive positional bias:
        bias_ij = beta * log(|A^T A_ij| + ata_eps)
    where A is the CURRENT operator at inference time.

    Key property: A^T A changes between Fourier (training) and Gaussian
    (test). The model must have learned during Fourier training to USE the
    bias adaptively — not just memorize Fourier-specific routing.

    If the model ignores the bias: attn_ata_bias ≈ attn_no_bias (ablation fails)
    If the model uses the bias: attn_ata_bias improves at Gaussian inference
      because it sees A^T A is ~scaled-identity (Gaussian) vs structured (Fourier)
      and adapts routing accordingly.
    """
    def __init__(self, n, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                 use_ata_bias=True, ata_eps=1e-4):
        super().__init__()
        self.use_ata_bias = use_ata_bias
        self.ata_eps      = ata_eps
        self.embed  = nn.Linear(1, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.decode = nn.Linear(d_model, 1)
        if use_ata_bias:
            self.log_beta = nn.Parameter(torch.zeros(1))

    def _pos_bias(self, AtA):
        beta = torch.exp(self.log_beta)
        return beta * torch.log(AtA.abs() + self.ata_eps)

    def forward(self, x_T, AtA=None):
        X        = self.embed(x_T.unsqueeze(-1))      # (B, n, d_model)
        pos_bias = self._pos_bias(AtA) if (self.use_ata_bias and AtA is not None) else None
        for layer in self.layers:
            X = layer(X, pos_bias)
        return x_T + self.decode(X).squeeze(-1)        # residual correction


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

def train_lista(lista, fourier_ops, X_train, X_test, alpha, args, dev):
    optimizer = optim.Adam(lista.parameters(), lr=args.lr)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + 1)
    history   = []

    print("\n── LISTA training (Fourier family) ──────────────────────")
    for epoch in range(args.epochs):
        lista.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            X_b  = X_train[perm[start: start + args.batch_size]]
            op   = fourier_ops[int(rng.randint(len(fourier_ops)))]
            loss = nn.functional.mse_loss(lista(op["A"], X_b @ op["A"].T), X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            lista.eval()
            avg = _avg_nrmse_lista(lista, fourier_ops, X_test)
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  avg-Fourier-NRMSE={avg:.4f}")
    return history


def train_cond_lista(cond_lista, fourier_ops, X_train, X_test, alpha, args, dev):
    optimizer = optim.Adam(cond_lista.parameters(), lr=args.lr)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + 2)
    history   = []

    print("\n── CondLISTA training (Fourier family) ──────────────────")
    for epoch in range(args.epochs):
        cond_lista.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            X_b  = X_train[perm[start: start + args.batch_size]]
            op   = fourier_ops[int(rng.randint(len(fourier_ops)))]
            pred = cond_lista(op["A"], X_b @ op["A"].T, op["eigvals"])
            loss = nn.functional.mse_loss(pred, X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            cond_lista.eval()
            avg = _avg_nrmse_cond(cond_lista, fourier_ops, X_test)
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  avg-Fourier-NRMSE={avg:.4f}")
    return history


def train_attn(model, fourier_ops, X_train, X_test, alpha, args, dev, label):
    """Train AttnReconstructor on ISTA iterates from Fourier family."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + 10)
    history   = []

    print(f"\n── {label} training (Fourier family) ─────────────────────")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            X_b  = X_train[perm[start: start + args.batch_size]]
            op   = fourier_ops[int(rng.randint(len(fourier_ops)))]
            with torch.no_grad():
                xT = ista_unroll(op["A"], X_b @ op["A"].T, alpha, args.lam, args.T)
            pred = model(xT, op["AtA"] if model.use_ata_bias else None)
            loss = nn.functional.mse_loss(pred, X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            model.eval()
            avg = _avg_nrmse_attn(model, fourier_ops, X_test, alpha, args)
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  avg-Fourier-NRMSE={avg:.4f}")
    return history


@torch.no_grad()
def _avg_nrmse_lista(model, ops, X_test):
    return float(np.mean([
        raw_nrmse(model(op["A"], X_test @ op["A"].T), X_test) for op in ops
    ]))


@torch.no_grad()
def _avg_nrmse_cond(model, ops, X_test):
    return float(np.mean([
        raw_nrmse(model(op["A"], X_test @ op["A"].T, op["eigvals"]), X_test)
        for op in ops
    ]))


@torch.no_grad()
def _avg_nrmse_attn(model, ops, X_test, alpha, args):
    vals = []
    for op in ops:
        xT   = ista_unroll(op["A"], X_test @ op["A"].T, alpha, args.lam, args.T)
        pred = model(xT, op["AtA"] if model.use_ata_bias else None)
        vals.append(raw_nrmse(pred, X_test))
    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────
# 8. EVALUATION
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_on_op(op, x_true, S_true, alpha, args, dev,
               lista=None, cond_lista=None,
               attn_no_bias=None, attn_ata=None):
    A, AtA, eigv = op["A"], op["AtA"], op["eigvals"]
    y   = x_true @ A.T
    xT  = ista_unroll(A, y, alpha, args.lam, args.T)
    res = {}

    res["oracle_nrmse"]    = oracle_ls_nrmse(x_true, A, S_true, dev)
    res["ista_topk_nrmse"] = topk_ls_nrmse(xT, x_true, A, args.k, dev)

    if lista is not None:
        res["lista_nrmse"] = raw_nrmse(lista(A, y), x_true)
    if cond_lista is not None:
        res["cond_lista_nrmse"] = raw_nrmse(cond_lista(A, y, eigv), x_true)
    if attn_no_bias is not None:
        res["attn_no_bias_nrmse"] = raw_nrmse(attn_no_bias(xT, None), x_true)
    if attn_ata is not None:
        res["attn_ata_nrmse"] = raw_nrmse(attn_ata(xT, AtA), x_true)
    return res


# ──────────────────────────────────────────────────────────────
# 9. ADAPTATION CURVES
# ──────────────────────────────────────────────────────────────

def adapt_curve(model_init, op, X_pool, X_test, alpha, args, dev,
                model_type="lista", label=""):
    """Fine-tune on N Gaussian samples, return {budget: nrmse}."""
    A, AtA, eigv = op["A"], op["AtA"], op["eigvals"]

    def _infer(m, xt=None):
        y = X_test @ A.T
        if model_type == "lista":
            return raw_nrmse(m(A, y), X_test)
        elif model_type == "cond_lista":
            return raw_nrmse(m(A, y, eigv), X_test)
        else:  # attn
            if xt is None:
                xt = ista_unroll(A, y, alpha, args.lam, args.T)
            return raw_nrmse(m(xt, AtA if m.use_ata_bias else None), X_test)

    base = copy.deepcopy(model_init).eval()
    with torch.no_grad():
        zs = _infer(base)
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
                X_b = X_ft[perm[start: start + args.batch_size]]
                y_b = X_b @ A.T
                if model_type == "lista":
                    pred = model_ft(A, y_b)
                elif model_type == "cond_lista":
                    pred = model_ft(A, y_b, eigv)
                else:  # attn
                    with torch.no_grad():
                        xT_b = ista_unroll(A, y_b, alpha, args.lam, args.T)
                    pred = model_ft(xT_b, AtA if model_ft.use_ata_bias else None)
                loss = nn.functional.mse_loss(pred, X_b)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model_ft.eval()
        with torch.no_grad():
            nrmse = _infer(model_ft)
        results[N] = nrmse
        print(f"    [{label}] N={N_actual:>4}  NRMSE={nrmse:.4f}")
    return results


def build_adapt_curves(gaussian_ops, lista, cond_lista, attn_no_bias, attn_ata,
                       X_train, X_test, alpha, args, dev):
    print("\n── Adaptation curves on Gaussian operators ──────────────")
    # Average across all Gaussian operators for a robust curve
    # Also keep per-operator for per-op plots
    methods = {
        "lista":        (lista,        "lista",     "LISTA"),
        "cond_lista":   (cond_lista,   "cond_lista","CondLISTA"),
        "attn_no_bias": (attn_no_bias, "attn",      "Attn-NoBias"),
        "attn_ata":     (attn_ata,     "attn",      "Attn-AtA"),
    }
    curves = {m: {} for m in methods}

    for op in gaussian_ops:
        oid = op["id"]
        print(f"\n  Gaussian op {oid}:")
        for mkey, (model, mtype, mlabel) in methods.items():
            print(f"    {mlabel}:")
            curves[mkey][oid] = adapt_curve(
                model, op, X_train, X_test, alpha, args, dev,
                model_type=mtype, label=mlabel
            )
    return curves


# ──────────────────────────────────────────────────────────────
# 10. PLOTTING
# ──────────────────────────────────────────────────────────────

# Phase 1 reference numbers for baseline annotation
_P1_NAIVE_TOPK_LS = 0.257   # naive top-k+LS zero-shot Gaussian (Phase 1)
_P1_LISTA_ZS      = 0.327   # LISTA zero-shot Gaussian (Phase 1)


def plot_summary(fourier_res, gaussian_res, oracle_g, args, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    methods_f = ["ista_topk_nrmse", "lista_nrmse",
                 "cond_lista_nrmse", "attn_no_bias_nrmse", "attn_ata_nrmse"]
    methods_g = methods_f
    labels    = ["ISTA\ntop-k+LS", "LISTA", "Cond\nLISTA",
                 "Attn\n(no bias)", "Attn\n(A^TA bias)"]
    colors    = ["steelblue", "forestgreen", "crimson", "darkorange", "purple"]

    for ax, (res, title, show_p1) in zip(axes, [
        (fourier_res, f"Fourier (seen, avg NRMSE)", False),
        (gaussian_res, f"Gaussian (unseen, zero-shot avg NRMSE)", True),
    ]):
        vals = []
        for m in methods_g:
            v = [r[m] for r in res.values() if m in r]
            vals.append(float(np.mean(v)) if v else float("nan"))

        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.axhline(oracle_g, color="black", ls=":", lw=1.5,
                   label=f"oracle LS={oracle_g:.3f}")
        if show_p1:
            ax.axhline(_P1_NAIVE_TOPK_LS, color="steelblue", ls="--", lw=1.5,
                       label=f"Phase 1 naive={_P1_NAIVE_TOPK_LS:.3f}")
            ax.axhline(_P1_LISTA_ZS, color="forestgreen", ls="--", lw=1.5,
                       label=f"Phase 1 LISTA={_P1_LISTA_ZS:.3f}")
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Exp 2: Cross-Family Transfer  |  "
        f"Fourier (train) → Gaussian (zero-shot)\n"
        f"n={args.n}, m={args.m}, k={args.k}, T={args.T}, lam={args.lam}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSummary plot -> {out_path}")
    plt.close()


def plot_adapt_curves(curves, ista_ref, oracle_ref, args, out_path):
    """
    Single averaged adaptation curve across all Gaussian operators.
    Shows crossover with Phase 1 naive top-k+LS baseline.
    """
    budgets = sorted(next(iter(curves["lista"].values())).keys())

    def mean_curve(method):
        return [
            float(np.mean([curves[method][oid].get(b, float("nan"))
                           for oid in curves[method]]))
            for b in budgets
        ]

    fig, ax = plt.subplots(figsize=(9, 6))
    style = {
        "lista":        ("forestgreen", "o-",  2.0, "LISTA"),
        "cond_lista":   ("crimson",     "s-",  2.0, "CondLISTA"),
        "attn_no_bias": ("darkorange",  "^--", 1.5, "Attn (no bias)"),
        "attn_ata":     ("purple",      "D-",  2.0, "Attn (A^TA bias)"),
    }
    for mkey, (color, marker, lw, label) in style.items():
        if mkey not in curves:
            continue
        ys = mean_curve(mkey)
        ax.plot(budgets, ys, marker, color=color, lw=lw, ms=7, label=label)

    ax.axhline(_P1_NAIVE_TOPK_LS, color="steelblue", ls="--", lw=1.8,
               label=f"Phase 1 naive top-k+LS={_P1_NAIVE_TOPK_LS:.3f}")
    ax.axhline(ista_ref, color="gray", ls=":", lw=1.4,
               label=f"ISTA baseline={ista_ref:.3f}")
    ax.axhline(oracle_ref, color="black", ls=":", lw=1.2,
               label=f"oracle LS={oracle_ref:.3f}")

    ax.set_xlabel("Gaussian fine-tuning samples (N)")
    ax.set_ylabel("Gaussian NRMSE")
    ax.set_title(
        "Exp 2: Cross-Family Adaptation Curves\n"
        "Train on Fourier → adapt to Gaussian",
        fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Adaptation curves plot -> {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 11. VERDICT
# ──────────────────────────────────────────────────────────────

def print_verdict(fourier_res, gaussian_res, adapt_curves, args):
    def avg(results, key):
        v = [r[key] for r in results.values() if key in r]
        return float(np.mean(v)) if v else float("nan")

    print("\n" + "=" * 68)
    print("  EXP 2 VERDICT — Cross-Family Transfer: Fourier → Gaussian")
    print("=" * 68)

    print(f"\n  [FOURIER — seen, avg NRMSE]")
    for key, name in [("ista_topk_nrmse", "ISTA top-k+LS"),
                      ("lista_nrmse", "LISTA"),
                      ("cond_lista_nrmse", "CondLISTA"),
                      ("attn_no_bias_nrmse", "Attn (no bias)"),
                      ("attn_ata_nrmse", "Attn (A^TA bias)")]:
        v = avg(fourier_res, key)
        if not math.isnan(v):
            print(f"    {name:<24}: {v:.4f}")

    print(f"\n  [GAUSSIAN — zero-shot avg NRMSE]   (Phase 1 refs: naive=0.257, LISTA=0.327)")
    vals_g = {}
    for key, name in [("ista_topk_nrmse", "ISTA top-k+LS"),
                      ("lista_nrmse", "LISTA"),
                      ("cond_lista_nrmse", "CondLISTA"),
                      ("attn_no_bias_nrmse", "Attn (no bias)"),
                      ("attn_ata_nrmse", "Attn (A^TA bias)")]:
        v = avg(gaussian_res, key)
        vals_g[key] = v
        if not math.isnan(v):
            vs_lista = f"  ({v - _P1_LISTA_ZS:+.4f} vs LISTA zs)" if key != "lista_nrmse" else ""
            print(f"    {name:<24}: {v:.4f}{vs_lista}")

    lista_zs   = vals_g.get("lista_nrmse",        float("nan"))
    cond_zs    = vals_g.get("cond_lista_nrmse",    float("nan"))
    attn_nb_zs = vals_g.get("attn_no_bias_nrmse",  float("nan"))
    attn_ata_zs= vals_g.get("attn_ata_nrmse",      float("nan"))

    print()
    # Q1: Does A^T A bias help attention?
    if attn_ata_zs < attn_nb_zs - 0.005:
        print(f"  [Q2 PASS] A^TA bias improves attention: "
              f"{attn_nb_zs:.4f} → {attn_ata_zs:.4f}  (Δ={attn_nb_zs-attn_ata_zs:.4f})")
        print("            Operator geometry at inference time carries transfer signal.")
    elif abs(attn_ata_zs - attn_nb_zs) <= 0.005:
        print(f"  [Q2 INCONCLUSIVE] A^TA bias ties no-bias: "
              f"{attn_ata_zs:.4f} vs {attn_nb_zs:.4f}")
        print("            Model may not have learned to use A^TA adaptively.")
    else:
        print(f"  [Q2 NEGATIVE] A^TA bias hurts: "
              f"{attn_nb_zs:.4f} → {attn_ata_zs:.4f}")
        print("            A^TA encodes Fourier-specific structure that misleads on Gaussian.")

    # Q2: Does best attention beat LISTA zero-shot?
    best_attn = min(attn_ata_zs, attn_nb_zs)
    print()
    if best_attn < lista_zs - 0.005:
        print(f"  [ARCHITECTURE PASS] Best attention ({best_attn:.4f}) < "
              f"LISTA zero-shot ({lista_zs:.4f})")
        print("  -> Attention adds value beyond operator-conditioned unrolling.")
    elif best_attn < _P1_LISTA_ZS - 0.005:
        print(f"  [PARTIAL PASS] Best attention ({best_attn:.4f}) < "
              f"Phase 1 LISTA ({_P1_LISTA_ZS:.4f}) but check vs current LISTA.")
    else:
        print(f"  [ARCHITECTURE NEGATIVE] Attention ({best_attn:.4f}) "
              f"≥ LISTA zero-shot ({lista_zs:.4f})")
        print("  -> Attention does not add value over unrolling for this transfer.")

    # Crossover analysis
    print(f"\n  [ADAPTATION — avg NRMSE by Gaussian fine-tuning budget]")
    budgets = sorted(next(iter(adapt_curves["lista"].values())).keys())
    print(f"  {'Budget':>7}  {'LISTA':>8}  {'CondLISTA':>10}  "
          f"{'Attn-NB':>8}  {'Attn-AtA':>9}")
    for b in budgets:
        def mb(mkey):
            v = [adapt_curves[mkey][oid].get(b, float("nan"))
                 for oid in adapt_curves[mkey]]
            return float(np.mean(v))
        print(f"  {b:>7}  {mb('lista'):>8.4f}  {mb('cond_lista'):>10.4f}  "
              f"{mb('attn_no_bias'):>8.4f}  {mb('attn_ata'):>9.4f}")

    # Find crossover with Phase 1 naive baseline (0.257)
    print()
    for mkey, name in [("lista", "LISTA"), ("cond_lista", "CondLISTA"),
                       ("attn_ata", "Attn-AtA")]:
        if mkey not in adapt_curves:
            continue
        for b in budgets:
            v = float(np.mean([adapt_curves[mkey][oid].get(b, float("nan"))
                                for oid in adapt_curves[mkey]]))
            if v <= _P1_NAIVE_TOPK_LS + 0.002:
                print(f"  {name} matches naive top-k+LS baseline at N={b}")
                break
    print("=" * 68)


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
    print(f"n={args.n}  m={args.m}  k={args.k}  T={args.T}  lam={args.lam}")
    print(f"Train: {args.n_fourier} Fourier ops  |  "
          f"Test: {args.n_gaussian} Gaussian ops")
    print(f"Attn: d_model={args.d_model}, n_heads={args.n_heads}, "
          f"n_layers={args.n_layers}, d_ff={args.d_ff}")
    print(f"Phase 1 references: naive top-k+LS=0.257, LISTA zero-shot=0.327")

    # ── Build operators ───────────────────────────────────────
    print("\n── Building operators ───────────────────────────────────")
    fourier_ops, gaussian_ops, sn_max, alpha = build_operators(args, dev)

    # ── Signals ───────────────────────────────────────────────
    X_train, _      = make_signals(args.n, args.k, args.n_train,
                                    args.amp_lo, args.amp_hi,
                                    seed=args.seed, device=dev)
    X_test, S_test  = make_signals(args.n, args.k, args.n_test,
                                    args.amp_lo, args.amp_hi,
                                    seed=args.seed + 999, device=dev)

    # ── Build models ──────────────────────────────────────────
    lista = LISTA(args.T, alpha, args.lam, sn_max).to(dev)

    cond_lista = CondLISTA(
        args.T, alpha, args.lam, sn_max, args.n, d_embed=args.d_embed
    ).to(dev)

    attn_no_bias = AttnReconstructor(
        args.n, args.d_model, args.n_heads, args.n_layers,
        args.d_ff, use_ata_bias=False
    ).to(dev)

    attn_ata = AttnReconstructor(
        args.n, args.d_model, args.n_heads, args.n_layers,
        args.d_ff, use_ata_bias=True
    ).to(dev)

    n_lista = sum(p.numel() for p in lista.parameters())
    n_cond  = sum(p.numel() for p in cond_lista.parameters())
    n_attn  = sum(p.numel() for p in attn_ata.parameters())
    print(f"\nParameters: LISTA={n_lista}  CondLISTA={n_cond}  Attn={n_attn}")

    # ── Train all models on Fourier family ────────────────────
    train_lista(lista, fourier_ops, X_train, X_test, alpha, args, dev)
    lista.eval()

    train_cond_lista(cond_lista, fourier_ops, X_train, X_test, alpha, args, dev)
    cond_lista.eval()

    train_attn(attn_no_bias, fourier_ops, X_train, X_test, alpha, args, dev,
               label="Attn-NoBias")
    attn_no_bias.eval()

    train_attn(attn_ata, fourier_ops, X_train, X_test, alpha, args, dev,
               label="Attn-AtA")
    attn_ata.eval()

    # ── Evaluate ──────────────────────────────────────────────
    print("\n── Evaluating all methods ───────────────────────────────")
    fourier_res  = {}
    gaussian_res = {}

    print(f"\n  Fourier (seen):")
    for op in fourier_ops:
        oid = op["id"]
        res = eval_on_op(op, X_test, S_test, alpha, args, dev,
                         lista=lista, cond_lista=cond_lista,
                         attn_no_bias=attn_no_bias, attn_ata=attn_ata)
        fourier_res[oid] = res
        print(f"  F{oid}: ista={res['ista_topk_nrmse']:.4f}  "
              f"lista={res.get('lista_nrmse', float('nan')):.4f}  "
              f"cond={res.get('cond_lista_nrmse', float('nan')):.4f}  "
              f"attn_nb={res.get('attn_no_bias_nrmse', float('nan')):.4f}  "
              f"attn_ata={res.get('attn_ata_nrmse', float('nan')):.4f}")

    print(f"\n  Gaussian (unseen, zero-shot):")
    for op in gaussian_ops:
        oid = op["id"]
        res = eval_on_op(op, X_test, S_test, alpha, args, dev,
                         lista=lista, cond_lista=cond_lista,
                         attn_no_bias=attn_no_bias, attn_ata=attn_ata)
        gaussian_res[oid] = res
        print(f"  G{oid}: ista={res['ista_topk_nrmse']:.4f}  "
              f"lista={res.get('lista_nrmse', float('nan')):.4f}  "
              f"cond={res.get('cond_lista_nrmse', float('nan')):.4f}  "
              f"attn_nb={res.get('attn_no_bias_nrmse', float('nan')):.4f}  "
              f"attn_ata={res.get('attn_ata_nrmse', float('nan')):.4f}")

    # ── Adaptation curves ─────────────────────────────────────
    adapt_curves = build_adapt_curves(
        gaussian_ops, lista, cond_lista, attn_no_bias, attn_ata,
        X_train, X_test, alpha, args, dev
    )

    # ── Verdict ───────────────────────────────────────────────
    print_verdict(fourier_res, gaussian_res, adapt_curves, args)

    # ── Save JSON ─────────────────────────────────────────────
    def ser(r):
        return {k: float(v) for k, v in r.items() if isinstance(v, float)}

    serialisable_curves = {
        method: {
            str(oid): {str(b): float(v) for b, v in curve.items()}
            for oid, curve in op_curves.items()
        }
        for method, op_curves in adapt_curves.items()
    }

    oracle_g = float(np.mean([r["oracle_nrmse"] for r in gaussian_res.values()]))
    out_json = os.path.join(args.out_dir, "exp2_cf_results.json")
    with open(out_json, "w") as fh:
        json.dump({
            "args":           vars(args),
            "alpha":          float(alpha),
            "sn_max":         float(sn_max),
            "n_params":       {"lista": n_lista, "cond_lista": n_cond,
                               "attn": n_attn},
            "phase1_refs":    {"naive_topk_ls": _P1_NAIVE_TOPK_LS,
                               "lista_zeroshot": _P1_LISTA_ZS},
            "oracle_fourier": float(np.mean([r["oracle_nrmse"]
                                             for r in fourier_res.values()])),
            "oracle_gaussian": oracle_g,
            "fourier_results": {str(k): ser(v) for k, v in fourier_res.items()},
            "gaussian_results":{str(k): ser(v) for k, v in gaussian_res.items()},
            "adapt_curves":   serialisable_curves,
        }, fh, indent=2)
    print(f"\nResults JSON -> {out_json}")

    # ── Plots ─────────────────────────────────────────────────
    ista_ref = float(np.mean([r["ista_topk_nrmse"] for r in gaussian_res.values()]))
    plot_summary(fourier_res, gaussian_res, oracle_g, args,
                 os.path.join(args.out_dir, "exp2_cf_summary.png"))
    plot_adapt_curves(adapt_curves, ista_ref, oracle_g, args,
                      os.path.join(args.out_dir, "exp2_cf_adapt_curves.png"))


if __name__ == "__main__":
    main()
