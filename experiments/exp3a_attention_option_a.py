"""
exp3a_attention_option_a.py
===========================
Option A: Attention + A^T A Bias on the Gaussian Variable-m Family

Mirrors the Exp 2 attention architecture (AttnReconstructor on ISTA iterates
with log|A^T A| positional bias) but applied to the exp3 Gaussian family so
that the encoder is IN-DISTRIBUTION at the new-m test points — removing the
cross-family extrapolation that made Exp 2 fail.

Three-number tracking (Option A protocol):
  CHECK 1  Seen-m in-distribution
           Is AttnBias competitive with SharedLISTA on TRAINING operators?
           If not, the transfer comparison is contaminated — flag immediately.

  CHECK 2  New-m zero-shot: with vs without A^T A bias
           Direct test of whether operator geometry enables transfer.
           AttnBias vs AttnNoBias on m={96,160}

  CHECK 3  New-m zero-shot headline
           Does attention with operator geometry beat the best unrolling baseline?
           AttnBias vs SharedLISTA on m={96,160}

Outcome decision tree:
  (1) PASS + (2) PASS + (3) PASS  =>  paper result
  (1) FAIL                        =>  fix architecture (hybrid LISTA+attn) before concluding
  (1) PASS + (2/3) not passing    =>  operator geometry not helping; consider CondLISTA instead

Setup (identical to exp3):
  Operator family : Gaussian N(0, 1/m)
  Training m      : {64, 128, 192}  2 instances each -> 6 training ops
  Test (seen m)   : {64, 128, 192}  1 held-out instance each -> 3 ops
  Test (new m)    : {96, 160}       2 instances each -> 4 ops
  n=256, k=25, T=30, lam=0.05

Attention backbone: per-operator ISTA (T steps, per-op alpha=1/sn_max^2)
                    then AttnReconstructor refines the ISTA iterate.

Usage:
    python exp3a_attention_option_a.py
    python exp3a_attention_option_a.py --device cuda

Outputs:
    results_exp3a/exp3a_results.json
    results_exp3a/exp3a_summary.png     (3-panel: training ops / seen-m / new-m)
    results_exp3a/exp3a_adapt_curves.png
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
    p.add_argument("--n",           type=int,   default=256)
    p.add_argument("--k",           type=int,   default=25)
    p.add_argument("--amp_lo",      type=float, default=0.5)
    p.add_argument("--amp_hi",      type=float, default=2.0)
    p.add_argument("--T",           type=int,   default=30)
    p.add_argument("--lam",         type=float, default=0.05)
    p.add_argument("--n_train",     type=int,   default=4000)
    p.add_argument("--n_test",      type=int,   default=500)
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--ft_epochs",   type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    # Attention architecture (matches exp2 defaults)
    p.add_argument("--d_model",     type=int,   default=32)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--n_layers",    type=int,   default=2)
    p.add_argument("--d_ff",        type=int,   default=64)
    p.add_argument("--ata_eps",     type=float, default=1e-4,
                   help="Epsilon for log(|AtA| + eps) bias")
    # Option A check threshold
    p.add_argument("--check1_tol",  type=float, default=0.01,
                   help="Acceptable NRMSE gap (attn-shared) for Check 1 to pass")
    p.add_argument("--ft_budgets",  type=int,   nargs="+",
                   default=[0, 10, 25, 50, 100, 200])
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir",     type=str,   default="results_exp3a")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2. OPERATORS  (identical to exp3)
# ──────────────────────────────────────────────────────────────

def make_gaussian_op(n, m, seed, device, label=""):
    rng = np.random.RandomState(seed)
    A   = torch.tensor(
        rng.randn(m, n).astype(np.float32) / np.sqrt(m),
        device=device
    )
    with torch.no_grad():
        AtA     = A.T @ A
        eigvals = torch.linalg.eigvalsh(AtA)
        sn_max  = math.sqrt(max(eigvals[-1].item(), 1e-8))
        alpha   = 1.0 / (sn_max ** 2)
        eigv    = eigvals.flip(0)
    return {
        "A": A, "AtA": AtA, "eigvals": eigv,
        "sn_max": sn_max, "alpha": alpha,
        "m": m, "label": label,
    }


def build_operators(args, device):
    train_m    = [64, 128, 192]
    test_new_m = [96, 160]

    train_ops     = []
    test_seen_ops = []
    test_new_ops  = []
    global_sn_max = 0.0

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
                  f"eig_min={op['eigvals'][-1]:.5f}")

    print("  Test operators — seen m (held-out instance):")
    for m in train_m:
        seed = m * 1000 + 2
        lbl  = f"G_test_seenm{m}"
        op   = make_gaussian_op(args.n, m, seed, device, lbl)
        test_seen_ops.append(op)
        global_sn_max = max(global_sn_max, op["sn_max"])
        print(f"    {lbl}: m={m}  sn_max={op['sn_max']:.4f}")

    print("  Test operators — new m value (interpolation test):")
    for m in test_new_m:
        for inst in range(2):
            seed = m * 1000 + inst
            lbl  = f"G_test_newm{m}_i{inst}"
            op   = make_gaussian_op(args.n, m, seed, device, lbl)
            test_new_ops.append(op)
            global_sn_max = max(global_sn_max, op["sn_max"])
            print(f"    {lbl}: m={m}  sn_max={op['sn_max']:.4f}")

    global_alpha = 1.0 / (global_sn_max ** 2)
    print(f"\n  global sn_max={global_sn_max:.4f}   global_alpha={global_alpha:.6f}")
    return train_ops, test_seen_ops, test_new_ops, global_sn_max, global_alpha


# ──────────────────────────────────────────────────────────────
# 3. DATA  (identical to exp3)
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
# 4. ISTA HELPERS  (identical to exp3)
# ──────────────────────────────────────────────────────────────

def soft_threshold(x, lam):
    return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)


@torch.no_grad()
def ista_unroll(A, y, alpha, lam, T):
    """Fixed ISTA using per-operator step size alpha."""
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
    Shared per-step (alpha_t, lam_t), no operator info beyond sn_max.
    Trained on mixed-m Gaussian data.  [2*T = 60 params]
    """
    def __init__(self, T, alpha_init, lam_init):
        super().__init__()
        self.T     = T
        self.alpha = nn.Parameter(torch.full((T,), float(alpha_init)))
        self.lam   = nn.Parameter(torch.full((T,), float(lam_init)))

    def forward(self, A, y, op_sn_max):
        alpha_max = 2.0 / (op_sn_max ** 2)
        x = torch.zeros(y.shape[0], A.shape[1], device=y.device)
        for t in range(self.T):
            residual = x @ A.T - y
            alpha_t  = self.alpha[t].clamp(min=1e-6, max=alpha_max)
            x = soft_threshold(x - alpha_t * (residual @ A),
                               self.lam[t].clamp(min=0.0))
        return x


class MultiHeadSelfAttn(nn.Module):
    """
    Multi-head self-attention with optional additive positional bias.
    Matches the exp2 architecture exactly.
    pos_bias shape: (n, n) — added to attention logits before softmax.
    """
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
        """X: (B, n, d_model)   pos_bias: (n, n) or None"""
        B, n, _ = X.shape
        H, dk   = self.n_heads, self.d_k

        Q = self.W_q(X).view(B, n, H, dk).transpose(1, 2)  # (B, H, n, dk)
        K = self.W_k(X).view(B, n, H, dk).transpose(1, 2)
        V = self.W_v(X).view(B, n, H, dk).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)   # (B, H, n, n)
        if pos_bias is not None:
            scores = scores + pos_bias.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)                # (B, H, n, n)
        out  = (attn @ V).transpose(1, 2).contiguous()     # (B, n, H*dk)
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
    Attention-based reconstructor applied on top of ISTA iterates.
    Identical architecture to exp2 — ported to the Gaussian variable-m family.

    Pipeline:
      x_T  (B, n)          fixed ISTA iterate (per-operator alpha, T steps)
        -> embed (B, n, d_model)   per-coordinate linear
        -> n_layers TransformerLayers with optional A^T A positional bias
             L_ij = QK^T/sqrt(d_k)  +  beta * log(|AtA_ij| + eps)
        -> decode (B, n)   per-coordinate residual delta
        -> x_hat = x_T + delta

    use_ata_bias=True:  biases attention routing by sensing geometry (A^T A)
    use_ata_bias=False: purely data-driven attention (ablation)

    log_beta initialized to 0 (beta=1 at start); trained to adjust bias strength.
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
        return beta * torch.log(AtA.abs() + self.ata_eps)  # (n, n)

    def forward(self, x_T, AtA=None):
        """
        x_T : (B, n)   ISTA iterate
        AtA : (n, n)   precomputed A^T A (ignored if use_ata_bias=False)
        """
        X = self.embed(x_T.unsqueeze(-1))           # (B, n, d_model)

        pos_bias = None
        if self.use_ata_bias and AtA is not None:
            pos_bias = self._pos_bias(AtA)          # (n, n)

        for layer in self.layers:
            X = layer(X, pos_bias)

        delta = self.decode(X).squeeze(-1)           # (B, n) residual
        return x_T + delta


# ──────────────────────────────────────────────────────────────
# 6. METRICS  (identical to exp3)
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
            ((x_hat - x_true[i]).norm() /
             x_true[i].norm().clamp(min=1e-8)).item()
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
            ((x_hat - x_true[i]).norm() /
             x_true[i].norm().clamp(min=1e-8)).item()
        )
    return float(np.mean(errs))


# ──────────────────────────────────────────────────────────────
# 7. TRAINING
# ──────────────────────────────────────────────────────────────

def train_shared_lista(model, train_ops, X_train, X_test, args, dev):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + 1)
    history   = []

    print("\n── SharedLISTA training (mixed-m Gaussian: m={64,128,192}) ───")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            X_b = X_train[perm[start: start + args.batch_size]]
            op  = train_ops[int(rng.randint(len(train_ops)))]
            y_b = X_b @ op["A"].T
            pred = model(op["A"], y_b, op["sn_max"])
            loss = nn.functional.mse_loss(pred, X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                avg = _shared_avg_nrmse(model, train_ops, X_test)
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  train-op avg NRMSE={avg:.4f}")
    return history


@torch.no_grad()
def _shared_avg_nrmse(model, ops, X_test):
    vals = []
    for op in ops:
        y     = X_test @ op["A"].T
        x_hat = model(op["A"], y, op["sn_max"])
        vals.append(raw_nrmse(x_hat, X_test))
    return float(np.mean(vals))


def train_attn_reconstructor(model, train_ops, X_train, X_test, args, dev,
                              label="AttnReconstructor", seed_offset=10):
    """
    Train AttnReconstructor on mixed-m training operators.
    For each batch: pick random training operator, compute ISTA iterate
    (using per-operator alpha, frozen), then update attention parameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_train   = X_train.shape[0]
    rng       = np.random.RandomState(args.seed + seed_offset)
    history   = []

    print(f"\n── {label} training (mixed-m Gaussian: m={{64,128,192}}) ───")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, args.batch_size):
            X_b    = X_train[perm[start: start + args.batch_size]]
            op     = train_ops[int(rng.randint(len(train_ops)))]
            A, AtA = op["A"], op["AtA"]
            y_b    = X_b @ A.T
            with torch.no_grad():
                xT = ista_unroll(A, y_b, op["alpha"], args.lam, args.T)
            x_hat = model(xT, AtA if model.use_ata_bias else None)
            loss  = nn.functional.mse_loss(x_hat, X_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                avg = _attn_avg_nrmse(model, train_ops, X_test, args)
            history.append(avg)
            print(f"  Epoch {epoch+1:>4}  train-op avg NRMSE={avg:.4f}")
    return history


@torch.no_grad()
def _attn_avg_nrmse(model, ops, X_test, args):
    vals = []
    for op in ops:
        y  = X_test @ op["A"].T
        xT = ista_unroll(op["A"], y, op["alpha"], args.lam, args.T)
        x_hat = model(xT, op["AtA"] if model.use_ata_bias else None)
        vals.append(raw_nrmse(x_hat, X_test))
    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────
# 8. EVALUATION
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_op(op, x_true, S_true, args, dev,
            shared_lista=None, attn_no_bias=None, attn_bias=None):
    A   = op["A"]
    AtA = op["AtA"]
    y   = x_true @ A.T
    xT  = ista_unroll(A, y, op["alpha"], args.lam, args.T)

    res = {}
    res["oracle_nrmse"]    = oracle_ls_nrmse(x_true, A, S_true, dev)
    res["ista_topk_nrmse"] = topk_ls_nrmse(xT, x_true, A, args.k, dev)

    if shared_lista is not None:
        res["shared_lista_nrmse"] = raw_nrmse(
            shared_lista(A, y, op["sn_max"]), x_true)

    if attn_no_bias is not None:
        res["attn_no_bias_nrmse"] = raw_nrmse(
            attn_no_bias(xT, None), x_true)

    if attn_bias is not None:
        res["attn_bias_nrmse"] = raw_nrmse(
            attn_bias(xT, AtA), x_true)

    return res


# ──────────────────────────────────────────────────────────────
# 9. ADAPTATION CURVES
# ──────────────────────────────────────────────────────────────

def adapt_curve(model_init, op, X_pool, X_test, args, dev,
                model_type, label):
    """
    Fine-tune model_init on N samples from op, report NRMSE curve.
    model_type: "shared" | "attn_no_bias" | "attn_bias"
    """
    A, AtA = op["A"], op["AtA"]

    def _infer(m):
        with torch.no_grad():
            y = X_test @ A.T
            if model_type == "shared":
                return raw_nrmse(m(A, y, op["sn_max"]), X_test)
            else:
                xT = ista_unroll(A, y, op["alpha"], args.lam, args.T)
                return raw_nrmse(
                    m(xT, AtA if m.use_ata_bias else None), X_test)

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
                    pred = model_ft(A, y_b, op["sn_max"])
                else:
                    with torch.no_grad():
                        xT_b = ista_unroll(A, y_b, op["alpha"], args.lam, args.T)
                    pred = model_ft(xT_b, AtA if model_ft.use_ata_bias else None)
                loss = nn.functional.mse_loss(pred, X_b)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model_ft.eval()
        nrmse = _infer(model_ft)
        results[N] = nrmse
        print(f"    [{label}] N={N_actual:>4}  NRMSE={nrmse:.4f}")
    return results


# ──────────────────────────────────────────────────────────────
# 10. OPTION A THREE-NUMBER VERDICT
# ──────────────────────────────────────────────────────────────

def print_verdict(train_op_res_list, seen_res_list, new_m_res_list,
                  adapt_new, adapt_seen, args):
    def avg(result_list, key):
        v = [r[key] for r in result_list if key in r]
        return float(np.mean(v)) if v else float("nan")

    print("\n" + "=" * 72)
    print("  OPTION A — THREE-NUMBER VERDICT")
    print("=" * 72)

    # ── CHECK 1: In-distribution on training operators ────────────────────
    shared_tr  = avg(train_op_res_list, "shared_lista_nrmse")
    attn_b_tr  = avg(train_op_res_list, "attn_bias_nrmse")
    attn_nb_tr = avg(train_op_res_list, "attn_no_bias_nrmse")
    gap1 = attn_b_tr - shared_tr   # positive = attention is WORSE

    print(f"\n  CHECK 1 — Seen-m in-distribution (training operators, test signals)")
    print(f"    SharedLISTA        : {shared_tr:.4f}")
    print(f"    AttnNoBias         : {attn_nb_tr:.4f}")
    print(f"    AttnBias (+A^TA)   : {attn_b_tr:.4f}  (gap vs shared = {gap1:+.4f})")

    if gap1 <= args.check1_tol:
        status1 = "PASS"
        print(f"  [CHECK 1 PASS] AttnBias competitive on training operators (gap={gap1:+.4f} ≤ {args.check1_tol})")
    else:
        status1 = "FAIL"
        print(f"  [CHECK 1 FAIL] AttnBias WORSE by {gap1:.4f} on training operators.")
        print(f"                 Transfer comparison is contaminated — starting from a weaker model.")
        print(f"                 Next step: hybrid (SharedLISTA T//2 steps + attention refinement T//2 steps)")

    # ── CHECK 2: New-m, with vs without A^T A bias ────────────────────────
    attn_nb_new = avg(new_m_res_list, "attn_no_bias_nrmse")
    attn_b_new  = avg(new_m_res_list, "attn_bias_nrmse")
    gap2 = attn_nb_new - attn_b_new   # positive = bias HELPS

    print(f"\n  CHECK 2 — New-m zero-shot: with vs without A^T A bias")
    print(f"    AttnNoBias         : {attn_nb_new:.4f}")
    print(f"    AttnBias (+A^TA)   : {attn_b_new:.4f}  (Δ = {gap2:+.4f})")

    if gap2 > 0.005:
        status2 = "PASS"
        print(f"  [CHECK 2 PASS] A^T A bias enables transfer (Δ={gap2:.4f} > 0.005)")
    elif abs(gap2) <= 0.005:
        status2 = "INCONCLUSIVE"
        print(f"  [CHECK 2 INCONCLUSIVE] A^T A bias makes negligible difference (|Δ|={abs(gap2):.4f})")
    else:
        status2 = "NEGATIVE"
        print(f"  [CHECK 2 NEGATIVE] A^T A bias HURTS transfer (Δ={gap2:.4f})")

    # ── CHECK 3: New-m headline, AttnBias vs SharedLISTA ─────────────────
    shared_new = avg(new_m_res_list, "shared_lista_nrmse")
    gap3 = shared_new - attn_b_new    # positive = attention WINS

    print(f"\n  CHECK 3 — New-m zero-shot headline: AttnBias vs SharedLISTA")
    print(f"    SharedLISTA        : {shared_new:.4f}")
    print(f"    AttnBias (+A^TA)   : {attn_b_new:.4f}  (Δ = {gap3:+.4f})")

    if gap3 > 0.005:
        status3 = "PASS"
        print(f"  [CHECK 3 PASS] Attention with operator geometry beats SharedLISTA (Δ={gap3:.4f})")
    elif abs(gap3) <= 0.005:
        status3 = "INCONCLUSIVE"
        print(f"  [CHECK 3 INCONCLUSIVE] Attention ≈ SharedLISTA at new-m")
    else:
        status3 = "NEGATIVE"
        print(f"  [CHECK 3 NEGATIVE] SharedLISTA beats attention at new-m (Δ={gap3:.4f})")

    # ── Outcome ───────────────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  OUTCOME  Check1={status1}  Check2={status2}  Check3={status3}")
    if status1 == "PASS" and status2 == "PASS" and status3 == "PASS":
        print("  => PAPER RESULT: attention + A^T A geometry dominates on Gaussian family.")
    elif status1 == "FAIL":
        print("  => ARCHITECTURE PROBLEM: fix attention model before interpreting transfer.")
        print("     Suggested fix: LISTA front-end (T//2 steps) + attention refinement.")
    elif status2 != "PASS" and status3 != "PASS":
        print("  => GEOMETRY NOT HELPING: A^T A bias provides no transfer benefit.")
        print("     Consider CondLISTA (eigenvalue encoding) instead of attention.")
    else:
        print("  => PARTIAL result — see individual checks above for next steps.")

    # ── Full zero-shot breakdown ───────────────────────────────────────────
    print(f"\n  Full zero-shot breakdown:")
    for split_name, res_list in [
        ("Training operators (seen m, in-dist)", train_op_res_list),
        ("Seen-m test {64,128,192} (held-out)",  seen_res_list),
        ("New-m test {96,160} (unseen m)",        new_m_res_list),
    ]:
        print(f"\n    {split_name}:")
        for key, name in [
            ("ista_topk_nrmse",    "ISTA top-k+LS    "),
            ("shared_lista_nrmse", "SharedLISTA      "),
            ("attn_no_bias_nrmse", "AttnNoBias       "),
            ("attn_bias_nrmse",    "AttnBias (+A^TA) "),
            ("oracle_nrmse",       "Oracle LS        "),
        ]:
            v = avg(res_list, key)
            if not math.isnan(v):
                print(f"      {name}: {v:.4f}")

    # ── Adaptation table ─────────────────────────────────────────────────
    budgets = sorted(next(iter(adapt_new["shared"].values())).keys())
    print(f"\n  Adaptation curves — new-m avg NRMSE vs fine-tuning samples:")
    print(f"  {'N':>7}  {'SharedLISTA':>12}  {'AttnNoBias':>11}  {'AttnBias':>9}")
    for b in budgets:
        def mb(mkey):
            vals = [adapt_new[mkey][op["label"]].get(b, float("nan"))
                    for op in adapt_new["_ops"]]
            return float(np.mean(vals))
        print(f"  {b:>7}  {mb('shared'):>12.4f}  {mb('attn_no_bias'):>11.4f}"
              f"  {mb('attn_bias'):>9.4f}")

    print("=" * 72)
    return {"check1": status1, "check2": status2, "check3": status3,
            "shared_train": shared_tr, "attn_bias_train": attn_b_tr,
            "shared_new_m": shared_new, "attn_bias_new_m": attn_b_new,
            "attn_no_bias_new_m": attn_nb_new}


# ──────────────────────────────────────────────────────────────
# 11. PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_summary(train_op_res, seen_res_list, new_m_res_list, args, out_path):
    """
    3-panel bar chart mapping directly to the three Option A checks.
      Panel 1: Training operators   (CHECK 1 — in-distribution)
      Panel 2: Seen-m test          (background, same m as training)
      Panel 3: New-m test           (CHECKS 2+3 — transfer)
    """
    methods = ["shared_lista_nrmse", "attn_no_bias_nrmse", "attn_bias_nrmse"]
    labels  = ["Shared\nLISTA", "Attn\n(no bias)", "Attn\n(+A^TA)"]
    colors  = ["forestgreen", "darkorange", "crimson"]

    panels = [
        (train_op_res,   "CHECK 1: Training Operators\n(seen m, in-distribution)"),
        (seen_res_list,  "Seen-m Test {64,128,192}\n(held-out instance, zero-shot)"),
        (new_m_res_list, "CHECK 2+3: New-m Test {96,160}\n(unseen m, zero-shot)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (res_list, title) in zip(axes, panels):
        ista_vals = [r["ista_topk_nrmse"] for r in res_list if "ista_topk_nrmse" in r]
        ista_avg  = float(np.mean(ista_vals)) if ista_vals else float("nan")

        vals = []
        for m in methods:
            v = [r[m] for r in res_list if m in r]
            vals.append(float(np.mean(v)) if v else float("nan"))

        all_labels = ["ISTA\ntop-k+LS"] + labels
        all_vals   = [ista_avg] + vals
        all_colors = ["steelblue"] + colors

        bars = ax.bar(all_labels, all_vals, color=all_colors, alpha=0.85, width=0.55)
        for bar, v in zip(bars, all_vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Exp 3A: Attention + A^TA Bias — Gaussian Family  "
        f"(n={args.n}, k={args.k}, T={args.T}, λ={args.lam}, "
        f"d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers})",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Summary plot -> {out_path}")
    plt.close()


def plot_adapt_curves(adapt_new, adapt_seen, args, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (adapt, title) in zip(axes, [
        (adapt_new,  "New m {96, 160} — unseen m value"),
        (adapt_seen, "Seen m {64,128,192} — held-out instance"),
    ]):
        ops     = adapt["_ops"]
        budgets = sorted(next(iter(adapt["shared"].values())).keys())

        def mean_curve(mkey):
            return [
                float(np.mean([adapt[mkey][op["label"]].get(b, float("nan"))
                                for op in ops]))
                for b in budgets
            ]

        for mkey, color, marker, lbl in [
            ("shared",       "forestgreen", "o-",  "SharedLISTA"),
            ("attn_no_bias", "darkorange",  "D--", "AttnNoBias"),
            ("attn_bias",    "crimson",     "s-",  "AttnBias (+A^TA)"),
        ]:
            ys = mean_curve(mkey)
            ax.plot(budgets, ys, marker, color=color, lw=2.0, ms=7, label=lbl)

        ax.set_xlabel("Fine-tuning samples (N)")
        ax.set_ylabel("NRMSE")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Exp 3A: Adaptation Curves — SharedLISTA vs AttnReconstructor",
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
    print(f"Attention: d_model={args.d_model}  n_heads={args.n_heads}  "
          f"n_layers={args.n_layers}  d_ff={args.d_ff}")
    print(f"Check-1 tolerance: {args.check1_tol}")
    print(f"Training m: {{64, 128, 192}}  (2 instances each -> 6 train ops)")
    print(f"Test seen m: {{64, 128, 192}}  (1 held-out instance -> 3 ops)")
    print(f"Test new m:  {{96, 160}}       (2 instances each -> 4 ops)")

    # ── Build operators ───────────────────────────────────────────────────
    print("\n── Building operators ───────────────────────────────────")
    (train_ops, test_seen_ops, test_new_ops,
     global_sn_max, global_alpha) = build_operators(args, dev)

    # ── Signals ───────────────────────────────────────────────────────────
    X_train, _ = make_signals(
        args.n, args.k, args.n_train,
        args.amp_lo, args.amp_hi, seed=args.seed, device=dev)
    X_test, S_test = make_signals(
        args.n, args.k, args.n_test,
        args.amp_lo, args.amp_hi, seed=args.seed + 999, device=dev)

    # ── Build models ───────────────────────────────────────────────────────
    alpha_init = global_alpha
    lam_init   = args.lam * alpha_init

    shared_lista = SharedLISTA(args.T, alpha_init, lam_init).to(dev)
    attn_no_bias = AttnReconstructor(
        args.n, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff,
        use_ata_bias=False, ata_eps=args.ata_eps
    ).to(dev)
    attn_bias = AttnReconstructor(
        args.n, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff,
        use_ata_bias=True, ata_eps=args.ata_eps
    ).to(dev)

    n_shared    = sum(p.numel() for p in shared_lista.parameters())
    n_attn_nb   = sum(p.numel() for p in attn_no_bias.parameters())
    n_attn_bias = sum(p.numel() for p in attn_bias.parameters())
    print(f"\nParameters:")
    print(f"  SharedLISTA         : {n_shared}")
    print(f"  AttnNoBias          : {n_attn_nb}")
    print(f"  AttnBias (+A^TA)    : {n_attn_bias}")
    print(f"  (attention backbone : per-operator ISTA, T={args.T} steps, frozen)")

    # ── Train ─────────────────────────────────────────────────────────────
    train_shared_lista(shared_lista, train_ops, X_train, X_test, args, dev)
    shared_lista.eval()

    train_attn_reconstructor(attn_no_bias, train_ops, X_train, X_test, args, dev,
                              label="AttnNoBias", seed_offset=10)
    attn_no_bias.eval()

    train_attn_reconstructor(attn_bias, train_ops, X_train, X_test, args, dev,
                              label="AttnBias (+A^TA)", seed_offset=20)
    attn_bias.eval()

    # ── CHECK 1: Diagnostic on training operators (in-distribution) ────────
    print("\n── CHECK 1 diagnostic: training operators ───────────────")
    print("   (Must pass before transfer results are interpretable)")
    train_op_res = []
    for op in train_ops:
        with torch.no_grad():
            r = eval_op(op, X_test, S_test, args, dev,
                        shared_lista, attn_no_bias, attn_bias)
        train_op_res.append(r)
        print(f"  {op['label']}: "
              f"shared={r['shared_lista_nrmse']:.4f}  "
              f"attn_nb={r['attn_no_bias_nrmse']:.4f}  "
              f"attn_b={r['attn_bias_nrmse']:.4f}")

    # ── Zero-shot evaluation ───────────────────────────────────────────────
    print("\n── Zero-shot evaluation ─────────────────────────────────")
    seen_res  = {}
    new_m_res = {}

    print("\n  Test: seen-m (held-out instance):")
    for op in test_seen_ops:
        with torch.no_grad():
            r = eval_op(op, X_test, S_test, args, dev,
                        shared_lista, attn_no_bias, attn_bias)
        seen_res[op["label"]] = r
        print(f"  {op['label']}: "
              f"ista={r['ista_topk_nrmse']:.4f}  "
              f"shared={r['shared_lista_nrmse']:.4f}  "
              f"attn_nb={r['attn_no_bias_nrmse']:.4f}  "
              f"attn_b={r['attn_bias_nrmse']:.4f}")

    print("\n  Test: new-m (unseen m value):")
    for op in test_new_ops:
        with torch.no_grad():
            r = eval_op(op, X_test, S_test, args, dev,
                        shared_lista, attn_no_bias, attn_bias)
        new_m_res[op["label"]] = r
        print(f"  {op['label']}: "
              f"ista={r['ista_topk_nrmse']:.4f}  "
              f"shared={r['shared_lista_nrmse']:.4f}  "
              f"attn_nb={r['attn_no_bias_nrmse']:.4f}  "
              f"attn_b={r['attn_bias_nrmse']:.4f}")

    # ── Adaptation curves ──────────────────────────────────────────────────
    print("\n── Adaptation curves (new-m operators) ─────────────────")
    adapt_new = {
        "shared": {}, "attn_no_bias": {}, "attn_bias": {},
        "_ops": test_new_ops,
    }
    for op in test_new_ops:
        lbl = op["label"]
        print(f"\n  {lbl} (m={op['m']}):")
        adapt_new["shared"][lbl] = adapt_curve(
            shared_lista, op, X_train, X_test, args, dev, "shared", "SharedLISTA")
        adapt_new["attn_no_bias"][lbl] = adapt_curve(
            attn_no_bias, op, X_train, X_test, args, dev, "attn_no_bias", "AttnNoBias")
        adapt_new["attn_bias"][lbl] = adapt_curve(
            attn_bias, op, X_train, X_test, args, dev, "attn_bias", "AttnBias")

    print("\n── Adaptation curves (seen-m operators) ─────────────────")
    adapt_seen = {
        "shared": {}, "attn_no_bias": {}, "attn_bias": {},
        "_ops": test_seen_ops,
    }
    for op in test_seen_ops:
        lbl = op["label"]
        print(f"\n  {lbl} (m={op['m']}):")
        adapt_seen["shared"][lbl] = adapt_curve(
            shared_lista, op, X_train, X_test, args, dev, "shared", "SharedLISTA")
        adapt_seen["attn_no_bias"][lbl] = adapt_curve(
            attn_no_bias, op, X_train, X_test, args, dev, "attn_no_bias", "AttnNoBias")
        adapt_seen["attn_bias"][lbl] = adapt_curve(
            attn_bias, op, X_train, X_test, args, dev, "attn_bias", "AttnBias")

    # ── Three-number verdict ───────────────────────────────────────────────
    check_results = print_verdict(
        train_op_res,
        list(seen_res.values()),
        list(new_m_res.values()),
        adapt_new, adapt_seen, args
    )

    # ── Save JSON ──────────────────────────────────────────────────────────
    def ser_list(lst):
        return [{k: float(v) for k, v in r.items()} for r in lst]

    def ser_res(d):
        return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in d.items()}

    def ser_curves(c):
        return {
            k: {lbl: {str(b): float(v) for b, v in curve.items()}
                for lbl, curve in op_curves.items()}
            for k, op_curves in c.items() if k != "_ops"
        }

    out_json = os.path.join(args.out_dir, "exp3a_results.json")
    with open(out_json, "w") as fh:
        json.dump({
            "args":             vars(args),
            "global_sn_max":    float(global_sn_max),
            "global_alpha":     float(global_alpha),
            "n_params": {
                "shared_lista": n_shared,
                "attn_no_bias": n_attn_nb,
                "attn_bias":    n_attn_bias,
            },
            "check_results":    check_results,
            "train_op_results": ser_list(train_op_res),
            "seen_results":     ser_res(seen_res),
            "new_m_results":    ser_res(new_m_res),
            "adapt_new":        ser_curves(adapt_new),
            "adapt_seen":       ser_curves(adapt_seen),
        }, fh, indent=2)
    print(f"\nResults JSON -> {out_json}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_summary(
        train_op_res,
        list(seen_res.values()),
        list(new_m_res.values()),
        args,
        os.path.join(args.out_dir, "exp3a_summary.png")
    )
    plot_adapt_curves(
        adapt_new, adapt_seen, args,
        os.path.join(args.out_dir, "exp3a_adapt_curves.png")
    )


if __name__ == "__main__":
    main()
