"""
exp2_attention_block.py

Architecture test: does operator-aware attention add anything beyond
operator-conditioned LISTA unrolling?

Run ONLY if Exp 1 shows that intra-family transfer is possible
(i.e., CondLISTA zero-shot beats per-mask LISTA zero-shot by a meaningful margin).

Same setup as Exp 1: partial Fourier family, n=256, m=128, k=25.
Seen masks 0--4, unseen masks 5--9.

Methods:
  1. ista_baseline   -- ISTA top-k + LS  [0 learned params]
  2. cond_lista      -- CondLISTA from Exp 1  [best conditioned unrolling baseline]
  3. attn_no_bias    -- Attention reconstructor on ISTA iterates, no A^T A bias
  4. attn_ata_bias   -- Attention reconstructor on ISTA iterates, A^T A positional bias

Architecture (attn_ata_bias):
  embed   : x_T (B,n) -> (B, n, d_model)   per-coordinate linear
  layers  : L x TransformerLayer with optional A^T A additive positional bias
            L_ij = QK^T/sqrt(d_k) + beta * log(|A^T A|_ij + eps)
  decode  : (B, n, d_model) -> (B, n)       per-coordinate linear + residual

Evaluation: same as Exp 1 — seen, unseen zero-shot, adaptation curves.

Key question:
  Does attn_ata_bias < cond_lista on unseen zero-shot?
  -> If yes: attention over ISTA iterates with A^T A bias provides additional
     transfer gains beyond simple operator-conditioned unrolling.
  -> If no: conditioned unrolling is sufficient; attention is not adding value
     for this family/regime.

Usage:
    python exp2_attention_block.py
    python exp2_attention_block.py --device cuda

Outputs:
    results_exp2/exp2_results.json
    results_exp2/exp2_summary.png
    results_exp2/exp2_adapt_curves.png
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
    p.add_argument("--n_masks",        type=int,   default=10)
    p.add_argument("--n_seen",         type=int,   default=5)
    p.add_argument("--n_train",        type=int,   default=4000)
    p.add_argument("--n_test",         type=int,   default=500)
    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--ft_epochs",      type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=64,
                   help="Smaller than Exp 1 due to O(n^2) attention")
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--d_embed",        type=int,   default=32,
                   help="CondLISTA embedding dim")
    p.add_argument("--d_model",        type=int,   default=32,
                   help="Attention model dimension")
    p.add_argument("--n_heads",        type=int,   default=4)
    p.add_argument("--n_layers",       type=int,   default=2)
    p.add_argument("--d_ff",           type=int,   default=64,
                   help="Feed-forward hidden dim in attention layers")
    p.add_argument("--ft_budgets",     type=int,   nargs="+",
                   default=[0, 10, 25, 50, 100, 200])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--mask_seed_base", type=int,   default=100)
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",        type=str,   default="results_exp2")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2. OPERATOR / MASK GENERATION  (identical to exp1)
# ──────────────────────────────────────────────────────────────

def make_partial_fourier(n, m, mask_seed, device):
    rng      = np.random.RandomState(mask_seed)
    n_freqs  = m // 2
    freq_idx = np.sort(rng.choice(n, n_freqs, replace=False))
    F_complex = np.fft.fft(np.eye(n)) / np.sqrt(n)
    rows_real = F_complex[freq_idx].real
    rows_imag = F_complex[freq_idx].imag
    rows      = np.concatenate([rows_real, rows_imag], axis=0)
    A = torch.tensor(rows[:m], dtype=torch.float32, device=device)
    b_freq = torch.zeros(n, dtype=torch.float32, device=device)
    b_freq[freq_idx] = 1.0
    return A, b_freq, freq_idx


def make_mask_family(n, m, n_masks, mask_seed_base, device):
    operators = []
    sn_max    = 0.0
    for i in range(n_masks):
        seed = mask_seed_base + i * 100
        A, b_freq, freq_idx = make_partial_fourier(n, m, seed, device)
        sn_i   = torch.linalg.norm(A, ord=2).item() ** 2
        sn_max = max(sn_max, sn_i)
        # Precompute A^T A once — used as positional bias in attention
        with torch.no_grad():
            AtA = A.T @ A   # (n, n)
        operators.append({
            "A": A, "b_freq": b_freq, "freq_idx": freq_idx,
            "mask_id": i, "sn": sn_i, "AtA": AtA
        })
    return operators, sn_max


# ──────────────────────────────────────────────────────────────
# 3. DATA GENERATION  (identical to exp1)
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
# 4. ISTA HELPERS  (identical to exp1)
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
    """CondLISTA from Exp 1 — reproduced here for self-containedness."""
    def __init__(self, T, alpha_init, lam_init, sn_max, n, d_embed=32):
        super().__init__()
        self.T         = T
        self.alpha_max = float(2.0 / sn_max)
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))
        self.op_encoder = nn.Sequential(nn.Linear(n, d_embed), nn.Tanh())
        self.alpha_mod  = nn.Linear(d_embed, T, bias=False)
        self.lam_mod    = nn.Linear(d_embed, T, bias=False)
        nn.init.zeros_(self.alpha_mod.weight)
        nn.init.zeros_(self.lam_mod.weight)

    def forward(self, A, y, b_freq):
        e           = self.op_encoder(b_freq)
        alpha_delta = self.alpha_mod(e)
        lam_delta   = self.lam_mod(e)
        x = torch.zeros(y.shape[0], A.shape[1], device=y.device)
        for t in range(self.T):
            residual = x @ A.T - y
            alpha_t  = (self.alpha[t] + alpha_delta[t]).clamp(min=1e-6, max=self.alpha_max)
            lam_t    = (self.lam[t]   + lam_delta[t]).clamp(min=0.0)
            x = soft_threshold(x - alpha_t * (residual @ A), lam_t)
        return x


# ── Attention components ──────────────────────────────────────

class MultiHeadSelfAttn(nn.Module):
    """
    Multi-head self-attention with optional additive positional bias.
    pos_bias shape: (n, n) — added to attention logits before softmax.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, pos_bias=None):
        """
        X        : (B, n, d_model)
        pos_bias : (n, n) or None
        """
        B, n, _ = X.shape
        H, dk   = self.n_heads, self.d_k

        Q = self.W_q(X).view(B, n, H, dk).transpose(1, 2)  # (B, H, n, dk)
        K = self.W_k(X).view(B, n, H, dk).transpose(1, 2)
        V = self.W_v(X).view(B, n, H, dk).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)   # (B, H, n, n)
        if pos_bias is not None:
            # pos_bias is (n, n); broadcast over batch and heads
            scores = scores + pos_bias.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)                # (B, H, n, n)
        out  = (attn @ V).transpose(1, 2).contiguous()     # (B, n, H, dk)
        out  = out.view(B, n, H * dk)
        return self.W_o(out)                                # (B, n, d_model)


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
    Attention-based reconstructor on top of ISTA iterates.

    Input:  x_T  (B, n)  — ISTA iterate after T steps
    Output: x_hat (B, n) — residual-corrected estimate

    If use_ata_bias=True, adds a log-scaled A^T A positional bias to each
    attention layer's logits:
        L_ij = QK^T/sqrt(d_k)  +  beta * log(|AtA_ij| + eps)

    This encodes the sensing geometry (which signal coordinates are jointly
    measured) directly into attention routing, allowing the model to adapt
    its routing pattern to the current operator at inference time.

    The learnable log_beta (initialized to 0, so beta=1 at start) lets
    the model set the strength of the operator prior during training.

    Parameters: embed(1->d_model) + L*(attn+ff+2*norm) + decode(d_model->1) + log_beta
    For d_model=32, n_heads=4, d_ff=64, n_layers=2:
      ~32 + 2*(32*32*4 + 32*96) + 32 + 1 ≈ 14k params (much smaller than the sequence)
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
        # Learnable log_beta: exp(log_beta) = beta, init 0 -> beta=1
        if use_ata_bias:
            self.log_beta = nn.Parameter(torch.zeros(1))

    def _pos_bias(self, AtA):
        """Compute (n, n) additive log-bias from A^T A."""
        beta = torch.exp(self.log_beta)
        return beta * torch.log(AtA.abs() + self.ata_eps)  # (n, n)

    def forward(self, x_T, AtA=None):
        """
        x_T : (B, n)   ISTA iterate
        AtA : (n, n)   precomputed A^T A for current operator (or None)
        """
        X = self.embed(x_T.unsqueeze(-1))   # (B, n, d_model)

        pos_bias = None
        if self.use_ata_bias and AtA is not None:
            pos_bias = self._pos_bias(AtA)  # (n, n)

        for layer in self.layers:
            X = layer(X, pos_bias)

        delta = self.decode(X).squeeze(-1)  # (B, n) residual correction
        return x_T + delta


# ──────────────────────────────────────────────────────────────
# 6. METRICS  (identical to exp1)
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

def train_cond_lista(cond_lista, seen_ops, X_train, X_test, alpha, args, dev):
    optimizer = optim.Adam(cond_lista.parameters(), lr=args.lr)
    n_seen    = len(seen_ops)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + 2)
    history   = []

    print("\n── CondLISTA training ────────────────────────────────────")
    for epoch in range(args.epochs):
        cond_lista.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            idx    = perm[start: start + args.batch_size]
            X_b    = X_train[idx]
            mask_i = int(rng.randint(n_seen))
            op     = seen_ops[mask_i]
            y      = X_b @ op["A"].T
            loss   = nn.functional.mse_loss(
                cond_lista(op["A"], y, op["b_freq"]), X_b
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            cond_lista.eval()
            avg = np.mean([
                raw_nrmse(
                    cond_lista(op["A"], X_test @ op["A"].T, op["b_freq"]).detach(),
                    X_test
                )
                for op in seen_ops
            ])
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  avg-seen-NRMSE={avg:.4f}")
    return history


def train_attn_reconstructor(model, seen_ops, X_train, X_test, alpha, args, dev,
                              label="AttnRecon"):
    """
    Train AttnReconstructor on seen operators.
    For each batch: pick a random seen mask, compute ISTA iterate, then train.
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_seen    = len(seen_ops)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + 10)
    history   = []

    print(f"\n── {label} training ──────────────────────────────────────")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            idx    = perm[start: start + args.batch_size]
            X_b    = X_train[idx]
            mask_i = int(rng.randint(n_seen))
            op     = seen_ops[mask_i]
            A, AtA = op["A"], op["AtA"]
            y      = X_b @ A.T
            with torch.no_grad():
                xT = ista_unroll(A, y, alpha, args.lam, args.T)
            x_hat = model(xT, AtA if model.use_ata_bias else None)
            loss  = nn.functional.mse_loss(x_hat, X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            model.eval()
            avg = np.mean([
                _attn_nrmse(model, op, X_test, alpha, args, dev)
                for op in seen_ops
            ])
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  avg-seen-NRMSE={avg:.4f}")
    return history


@torch.no_grad()
def _attn_nrmse(model, op, X_test, alpha, args, dev):
    A, AtA = op["A"], op["AtA"]
    y  = X_test @ A.T
    xT = ista_unroll(A, y, alpha, args.lam, args.T)
    return raw_nrmse(model(xT, AtA if model.use_ata_bias else None), X_test)


# ──────────────────────────────────────────────────────────────
# 8. EVALUATION
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_on_operator(op, x_true, S_true, alpha, args, dev,
                     cond_lista=None, attn_no_bias=None, attn_ata=None):
    A, AtA, b_freq = op["A"], op["AtA"], op["b_freq"]
    k = args.k
    y = x_true @ A.T
    xT = ista_unroll(A, y, alpha, args.lam, args.T)

    res = {}
    res["oracle_nrmse"]      = oracle_ls_nrmse(x_true, A, S_true, dev)
    res["ista_topk_nrmse"]   = topk_ls_nrmse(xT, x_true, A, k, dev)

    if cond_lista is not None:
        res["cond_lista_nrmse"] = raw_nrmse(
            cond_lista(A, y, b_freq), x_true
        )
    if attn_no_bias is not None:
        res["attn_no_bias_nrmse"] = raw_nrmse(
            attn_no_bias(xT, None), x_true
        )
    if attn_ata is not None:
        res["attn_ata_nrmse"] = raw_nrmse(
            attn_ata(xT, AtA), x_true
        )
    return res


# ──────────────────────────────────────────────────────────────
# 9. ADAPTATION CURVES
# ──────────────────────────────────────────────────────────────

def adapt_curve_for_op(model_init, op, X_pool, X_test, alpha, args, dev,
                       model_type="cond_lista", label=""):
    """
    Fine-tune model_init on N samples from op, return budget -> NRMSE dict.
    model_type: "cond_lista" | "attn"
    """
    A, AtA, b_freq = op["A"], op["AtA"], op["b_freq"]

    base = copy.deepcopy(model_init).eval()
    with torch.no_grad():
        if model_type == "cond_lista":
            zs = raw_nrmse(base(A, X_test @ A.T, b_freq), X_test)
        else:
            xT = ista_unroll(A, X_test @ A.T, alpha, args.lam, args.T)
            zs = raw_nrmse(base(xT, AtA if base.use_ata_bias else None), X_test)
    results = {0: zs}

    for N in args.ft_budgets:
        if N == 0:
            continue
        N_actual = min(N, X_pool.shape[0])
        X_ft     = X_pool[:N_actual]
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
                    loss = nn.functional.mse_loss(pred, X_b)
                else:
                    with torch.no_grad():
                        xT_b = ista_unroll(A, y, alpha, args.lam, args.T)
                    pred = model_ft(xT_b, AtA if model_ft.use_ata_bias else None)
                    loss = nn.functional.mse_loss(pred, X_b)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model_ft.eval()
        with torch.no_grad():
            if model_type == "cond_lista":
                nrmse = raw_nrmse(model_ft(A, X_test @ A.T, b_freq), X_test)
            else:
                xT = ista_unroll(A, X_test @ A.T, alpha, args.lam, args.T)
                nrmse = raw_nrmse(
                    model_ft(xT, AtA if model_ft.use_ata_bias else None), X_test
                )
        results[N] = nrmse
        print(f"    [{label}] N={N_actual:>4}  NRMSE={nrmse:.4f}")
    return results


def build_adapt_curves(unseen_ops, cond_lista, attn_no_bias, attn_ata,
                       X_train, X_test, alpha, args, dev):
    print("\n── Adaptation curves (unseen masks) ─────────────────────")
    curves = {"cond_lista": {}, "attn_no_bias": {}, "attn_ata": {}}

    for op in unseen_ops:
        mid = op["mask_id"]
        print(f"\n  Mask {mid} (unseen):")

        print("    cond_lista:")
        curves["cond_lista"][mid] = adapt_curve_for_op(
            cond_lista, op, X_train, X_test, alpha, args, dev,
            model_type="cond_lista", label="cond"
        )
        print("    attn_no_bias:")
        curves["attn_no_bias"][mid] = adapt_curve_for_op(
            attn_no_bias, op, X_train, X_test, alpha, args, dev,
            model_type="attn", label="attn_no_bias"
        )
        print("    attn_ata:")
        curves["attn_ata"][mid] = adapt_curve_for_op(
            attn_ata, op, X_train, X_test, alpha, args, dev,
            model_type="attn", label="attn_ata"
        )
    return curves


# ──────────────────────────────────────────────────────────────
# 10. PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_summary(seen_results, unseen_results, oracle_nrmse, args, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    methods = ["ista_topk_nrmse", "cond_lista_nrmse",
               "attn_no_bias_nrmse", "attn_ata_nrmse"]
    labels  = ["ISTA\ntop-k+LS", "CondLISTA", "Attn\n(no bias)", "Attn\n(A^TA bias)"]
    colors  = ["steelblue", "crimson", "darkorange", "forestgreen"]

    for ax, (results, title) in zip(axes, [
        (seen_results,   "Seen masks (avg NRMSE)"),
        (unseen_results, "Unseen masks — zero-shot (avg NRMSE)")
    ]):
        vals = [
            float(np.mean([r[m] for r in results.values() if m in r]))
            for m in methods
        ]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.axhline(oracle_nrmse, color="black", ls=":", lw=1.5,
                   label=f"oracle LS={oracle_nrmse:.3f}")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Exp 2: Attention vs CondLISTA — Partial Fourier Family\n"
        f"n={args.n}, m={args.m}, k={args.k}  |  "
        f"d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSummary plot -> {out_path}")
    plt.close()


def plot_adapt_curves(curves, ista_baseline, oracle_nrmse, args, out_path):
    unseen_ids = sorted(curves["cond_lista"].keys())
    n_unseen   = len(unseen_ids)
    ncols = min(n_unseen, 3)
    nrows = (n_unseen + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    style = {
        "cond_lista":    ("crimson",      "s-", "CondLISTA"),
        "attn_no_bias":  ("darkorange",   "o-", "Attn (no bias)"),
        "attn_ata":      ("forestgreen",  "^-", "Attn (A^TA bias)"),
    }
    budgets = sorted({b for m_curves in curves["cond_lista"].values()
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
                   ls="--", lw=1.5, label=f"ISTA={ista_baseline[mid]:.3f}")
        ax.axhline(oracle_nrmse, color="black",
                   ls=":", lw=1.5, label=f"oracle={oracle_nrmse:.3f}")
        ax.set_xlabel("Adaptation samples")
        ax.set_ylabel("NRMSE")
        ax.set_title(f"Unseen mask {mid}", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_unseen, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.suptitle("Exp 2: Adaptation curves — Attn vs CondLISTA",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Adaptation curves plot -> {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 11. VERDICT
# ──────────────────────────────────────────────────────────────

def print_verdict(seen_res, unseen_res, adapt_curves, args):
    methods = ["ista_topk_nrmse", "cond_lista_nrmse",
               "attn_no_bias_nrmse", "attn_ata_nrmse"]
    names   = ["ISTA top-k+LS", "CondLISTA", "Attn (no bias)", "Attn (A^TA bias)"]

    print("\n" + "=" * 66)
    print("  EXP 2 VERDICT — Attention vs CondLISTA on Unseen Masks")
    print("=" * 66)

    print("\n  [SEEN MASKS — avg NRMSE]")
    for m, name in zip(methods, names):
        vals = [r[m] for r in seen_res.values() if m in r]
        if vals:
            print(f"    {name:<24}: {np.mean(vals):.4f}")

    print("\n  [UNSEEN MASKS — zero-shot avg NRMSE]")
    unseen_vals = {}
    for m, name in zip(methods, names):
        vals = [r[m] for r in unseen_res.values() if m in r]
        if vals:
            unseen_vals[m] = np.mean(vals)
            print(f"    {name:<24}: {np.mean(vals):.4f}")

    cond_zs  = unseen_vals.get("cond_lista_nrmse", float("inf"))
    attn_zs  = unseen_vals.get("attn_ata_nrmse",   float("inf"))
    atnb_zs  = unseen_vals.get("attn_no_bias_nrmse", float("inf"))

    print()
    print("  [ANALYSIS]")
    # Does A^T A bias help attention?
    if attn_zs < atnb_zs - 0.005:
        print(f"  A^TA bias improves attention zero-shot: "
              f"{atnb_zs:.4f} -> {attn_zs:.4f}  (gain={atnb_zs-attn_zs:.4f})")
        print("  -> Operator geometry in A^T A carries transferable signal.")
    elif abs(attn_zs - atnb_zs) <= 0.005:
        print(f"  A^TA bias ties no-bias attention ({attn_zs:.4f} vs {atnb_zs:.4f}).")
        print("  -> Positional bias from A^T A is not providing extra transfer signal.")
    else:
        print(f"  A^TA bias HURTS attention: {atnb_zs:.4f} -> {attn_zs:.4f}.")
        print("  -> A^T A bias may be overfitting source-operator structure.")

    print()
    # Does attention beat CondLISTA?
    best_attn = min(attn_zs, atnb_zs)
    if best_attn < cond_zs - 0.005:
        print(f"  PASS: Best attention ({best_attn:.4f}) beats CondLISTA ({cond_zs:.4f}).")
        print("  -> Attention adds value beyond operator-conditioned unrolling.")
    elif abs(best_attn - cond_zs) <= 0.005:
        print(f"  INCONCLUSIVE: Attention ({best_attn:.4f}) ties CondLISTA ({cond_zs:.4f}).")
        print("  -> Conditioned unrolling and attention are equivalent here.")
    else:
        print(f"  NEGATIVE: CondLISTA ({cond_zs:.4f}) beats attention ({best_attn:.4f}).")
        print("  -> For this regime, operator-conditioned unrolling is sufficient.")
        print("     Consider: larger model, more masks, or real MRI data.")

    print()
    print("  [ADAPTATION — avg NRMSE by budget]")
    budgets = sorted(next(iter(adapt_curves["cond_lista"].values())).keys())
    print(f"    {'Budget':>8}  {'CondLISTA':>10}  {'Attn-NoB':>10}  {'Attn-AtA':>10}")
    for b in budgets:
        co = np.mean([adapt_curves["cond_lista"][mid].get(b, float("nan"))
                      for mid in adapt_curves["cond_lista"]])
        nb = np.mean([adapt_curves["attn_no_bias"][mid].get(b, float("nan"))
                      for mid in adapt_curves["attn_no_bias"]])
        at = np.mean([adapt_curves["attn_ata"][mid].get(b, float("nan"))
                      for mid in adapt_curves["attn_ata"]])
        print(f"    {b:>8}  {co:>10.4f}  {nb:>10.4f}  {at:>10.4f}")
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
    print(f"Attn: d_model={args.d_model}  n_heads={args.n_heads}  "
          f"n_layers={args.n_layers}  d_ff={args.d_ff}")
    print(f"Masks total={args.n_masks}  seen={args.n_seen}  "
          f"unseen={args.n_masks - args.n_seen}")

    # ── Build mask family ────────────────────────────────────
    print("\n── Building partial Fourier mask family ─────────────────")
    operators, sn_max = make_mask_family(
        args.n, args.m, args.n_masks, args.mask_seed_base, dev
    )
    alpha      = 1.0 / sn_max
    seen_ops   = operators[:args.n_seen]
    unseen_ops = operators[args.n_seen:]

    for op in operators:
        tag = "seen  " if op["mask_id"] < args.n_seen else "unseen"
        print(f"  Mask {op['mask_id']}  [{tag}]  sn={op['sn']:.4f}")
    print(f"  sn_max={sn_max:.4f}   alpha={alpha:.6f}")

    # ── Data ─────────────────────────────────────────────────
    X_train, _ = make_signals(
        args.n, args.k, args.n_train,
        args.amp_lo, args.amp_hi, seed=args.seed, device=dev
    )
    X_test, S_test = make_signals(
        args.n, args.k, args.n_test,
        args.amp_lo, args.amp_hi, seed=args.seed + 999, device=dev
    )

    # ── CondLISTA (baseline from Exp 1) ───────────────────────
    cond_lista = CondLISTA(
        args.T, alpha, args.lam, sn_max, args.n, d_embed=args.d_embed
    ).to(dev)
    n_cond = sum(p.numel() for p in cond_lista.parameters())
    print(f"\nCondLISTA parameters: {n_cond}")
    train_cond_lista(cond_lista, seen_ops, X_train, X_test, alpha, args, dev)
    cond_lista.eval()

    # ── Attention reconstructors ───────────────────────────────
    attn_no_bias = AttnReconstructor(
        args.n, args.d_model, args.n_heads, args.n_layers,
        args.d_ff, use_ata_bias=False
    ).to(dev)
    attn_ata = AttnReconstructor(
        args.n, args.d_model, args.n_heads, args.n_layers,
        args.d_ff, use_ata_bias=True
    ).to(dev)
    n_attn = sum(p.numel() for p in attn_ata.parameters())
    print(f"AttnReconstructor parameters: {n_attn} "
          f"(+1 log_beta for ata variant)")

    train_attn_reconstructor(
        attn_no_bias, seen_ops, X_train, X_test, alpha, args, dev,
        label="Attn-NoBias"
    )
    attn_no_bias.eval()

    train_attn_reconstructor(
        attn_ata, seen_ops, X_train, X_test, alpha, args, dev,
        label="Attn-AtA"
    )
    attn_ata.eval()

    # ── Oracle ───────────────────────────────────────────────
    oracle_seen   = float(np.mean([
        oracle_ls_nrmse(X_test, op["A"], S_test, dev) for op in seen_ops
    ]))
    oracle_unseen = float(np.mean([
        oracle_ls_nrmse(X_test, op["A"], S_test, dev) for op in unseen_ops
    ]))
    oracle_avg = (oracle_seen + oracle_unseen) / 2
    print(f"\n  Oracle LS: seen={oracle_seen:.4f}  unseen={oracle_unseen:.4f}")

    # ── Evaluation ───────────────────────────────────────────
    print("\n── Evaluating all methods ────────────────────────────────")
    seen_results   = {}
    unseen_results = {}

    print("\n  Seen masks:")
    for op in seen_ops:
        mid = op["mask_id"]
        res = eval_on_operator(
            op, X_test, S_test, alpha, args, dev,
            cond_lista=cond_lista,
            attn_no_bias=attn_no_bias,
            attn_ata=attn_ata
        )
        seen_results[mid] = res
        print(f"  Mask {mid}: oracle={res['oracle_nrmse']:.4f}  "
              f"ista={res['ista_topk_nrmse']:.4f}  "
              f"cond={res.get('cond_lista_nrmse', float('nan')):.4f}  "
              f"attn_nb={res.get('attn_no_bias_nrmse', float('nan')):.4f}  "
              f"attn_ata={res.get('attn_ata_nrmse', float('nan')):.4f}")

    print("\n  Unseen masks (zero-shot):")
    for op in unseen_ops:
        mid = op["mask_id"]
        res = eval_on_operator(
            op, X_test, S_test, alpha, args, dev,
            cond_lista=cond_lista,
            attn_no_bias=attn_no_bias,
            attn_ata=attn_ata
        )
        unseen_results[mid] = res
        print(f"  Mask {mid}: oracle={res['oracle_nrmse']:.4f}  "
              f"ista={res['ista_topk_nrmse']:.4f}  "
              f"cond={res.get('cond_lista_nrmse', float('nan')):.4f}  "
              f"attn_nb={res.get('attn_no_bias_nrmse', float('nan')):.4f}  "
              f"attn_ata={res.get('attn_ata_nrmse', float('nan')):.4f}")

    # ── Adaptation curves ─────────────────────────────────────
    adapt_curves = build_adapt_curves(
        unseen_ops, cond_lista, attn_no_bias, attn_ata,
        X_train, X_test, alpha, args, dev
    )

    # ── Verdict ───────────────────────────────────────────────
    print_verdict(seen_results, unseen_results, adapt_curves, args)

    # ── Save JSON ─────────────────────────────────────────────
    def ser(r):
        return {k: float(v) for k, v in r.items()
                if isinstance(v, (int, float))}

    out_json = os.path.join(args.out_dir, "exp2_results.json")
    serialisable_curves = {}
    for method, mask_curves in adapt_curves.items():
        serialisable_curves[method] = {
            str(mid): {str(b): float(v) for b, v in curve.items()}
            for mid, curve in mask_curves.items()
        }
    with open(out_json, "w") as fh:
        json.dump({
            "args":               vars(args),
            "alpha":              float(alpha),
            "sn_max":             float(sn_max),
            "oracle_seen":        oracle_seen,
            "oracle_unseen":      oracle_unseen,
            "n_params_cond_lista": sum(p.numel() for p in cond_lista.parameters()),
            "n_params_attn":      n_attn,
            "seen_results":       {str(k): ser(v) for k, v in seen_results.items()},
            "unseen_results":     {str(k): ser(v) for k, v in unseen_results.items()},
            "adapt_curves":       serialisable_curves,
        }, fh, indent=2)
    print(f"\nResults JSON -> {out_json}")

    # ── Plots ─────────────────────────────────────────────────
    ista_baseline_per_mask = {
        mid: res["ista_topk_nrmse"] for mid, res in unseen_results.items()
    }
    plot_summary(
        seen_results, unseen_results, oracle_avg, args,
        os.path.join(args.out_dir, "exp2_summary.png")
    )
    plot_adapt_curves(
        adapt_curves, ista_baseline_per_mask, oracle_avg, args,
        os.path.join(args.out_dir, "exp2_adapt_curves.png")
    )


if __name__ == "__main__":
    main()
