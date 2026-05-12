"""
GAST: Gram-Aware Support Transformer.
Quick proof-of-concept comparing two architectures on the same data:

  - CoordMLP   the per-coordinate MLP from learned_above_pt.py /
               learned_compressible.py (the current paper's detector)
  - GAST       same I/O contract, but the per-coord MLP is replaced
               with a small transformer encoder over the n coordinate
               tokens, with an additive attention bias derived from
               |A^T A| (off-diagonal). This is the "Gram-biased
               coordinate attention" architecture.

Two regimes are supported via --regime:

  strict55          n=256, m=128, k=55, strict k-sparse signals
                    (IoU plateau test: does attention crack 0.51?)
  compressible25    n=256, m=128, k=25, training tail ~ Unif[0.1, 0.4]
                    (extension test: does attention widen the existing
                    compressibility win?)

Default: --regime strict55 --arch both, single seed. Reports CoordMLP vs
GAST IoU and NRMSE side by side, plus the classical baselines for
context. ~3-8 min on CPU per architecture per regime.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--regime", choices=["strict55", "compressible25"],
                    default="strict55")
parser.add_argument("--arch", choices=["mlp", "gast", "both"], default="both")
parser.add_argument("--op-seed", type=int, default=0)
parser.add_argument("--init-seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--d", type=int, default=24, help="GAST embedding dim")
parser.add_argument("--n-heads", type=int, default=4)
parser.add_argument("--n-layers", type=int, default=2)
parser.add_argument("--mlp-hidden", type=int, default=64)
parser.add_argument("--n-train", type=int, default=4000)
parser.add_argument("--n-test", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--tag", type=str, default="quick")
args = parser.parse_args()

n, m = 256, 128
T_ista, lam_ista = 30, 0.05
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.init_seed)
print(f"device={device}  regime={args.regime}  arch={args.arch}")

if args.regime == "strict55":
    k = 55
    eval_tails = [0.0]              # strict-sparse only
    train_tail_fn = lambda rng: 0.0
elif args.regime == "compressible25":
    k = 25
    eval_tails = [0.1, 0.2, 0.3, 0.4]
    train_tail_fn = lambda rng: float(rng.uniform(0.1, 0.4))
print(f"n={n}, m={m}, k={k}")

# ----------------------------------------------------------------------
# Operator + Gram bias
# ----------------------------------------------------------------------

np_rng = np.random.RandomState(args.op_seed)
A_np = np_rng.randn(m, n).astype(np.float32)
A_np = A_np / np.linalg.norm(A_np, axis=0, keepdims=True)
A = torch.tensor(A_np, device=device)

G_np = A_np.T @ A_np                       # full Gram (n, n)
gram_diag_np = np.diag(G_np).astype(np.float32)
G_off = G_np.copy()
np.fill_diagonal(G_off, 0.0)
max_coh_np = np.max(np.abs(G_off), axis=0).astype(np.float32)
mu = float(np.max(max_coh_np))
print(f"mu(A) = {mu:.4f}")

# Off-diagonal |G| as the attention bias matrix (n, n).
# Diagonal zeroed so a token doesn't get an artificial self-bias on top
# of the Q.K dot product.
gram_bias = torch.tensor(np.abs(G_off), device=device, dtype=torch.float32)
gram_diag = torch.tensor(gram_diag_np, device=device)
max_coh = torch.tensor(max_coh_np, device=device)

spec_norm = float(np.linalg.norm(A_np, 2))
alpha_ista = 0.95 / spec_norm**2

# ----------------------------------------------------------------------
# Signals  (matches learned_compressible.py)
# ----------------------------------------------------------------------

def gen_compressible_batch(n_sig, k, tail_amp_fn, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_sig, n), dtype=np.float32)
    Tail = np.zeros((n_sig, n), dtype=np.float32)
    for i in range(n_sig):
        supp = rng.choice(n, size=k, replace=False)
        amps = rng.uniform(0.5, 2.0, size=k) * rng.choice([-1, 1], size=k)
        X[i, supp] = amps
        ta = tail_amp_fn(rng)
        if ta > 0:
            t = ta * rng.standard_normal(n).astype(np.float32)
            t[supp] = 0.0
            Tail[i] = t
    return X + Tail

def topk_labels(X, k):
    Xt = torch.as_tensor(X)
    idx = Xt.abs().topk(k, dim=1).indices
    S = torch.zeros_like(Xt)
    S.scatter_(1, idx, 1.0)
    return S

print(f"Generating {args.n_train} train signals ...")
X_train_np = gen_compressible_batch(args.n_train, k, train_tail_fn, seed=1)
S_train_np = topk_labels(X_train_np, k).numpy()
X_train = torch.tensor(X_train_np, device=device)
S_train = torch.tensor(S_train_np, device=device)

test_sets = {}
for ta in eval_tails:
    seed_te = 2000 + int(ta * 1000)
    X_te = gen_compressible_batch(args.n_test, k,
                                   (lambda rng, t=ta: t), seed=seed_te)
    test_sets[ta] = X_te

# ----------------------------------------------------------------------
# ISTA + features  (matches learned_compressible.py)
# ----------------------------------------------------------------------

def soft_threshold(x, lam):
    return torch.sign(x) * torch.clamp(x.abs() - lam, min=0.0)

def ista(A, y, alpha, lam, T):
    x = torch.zeros(y.shape[0], A.shape[1], device=A.device)
    for _ in range(T):
        residual = x @ A.T - y
        x = soft_threshold(x - alpha * (residual @ A), lam)
    return x

def make_features(A, y, x_T, gram_diag, max_coh):
    B = x_T.shape[0]
    abs_x = x_T.abs()
    residual = x_T @ A.T - y
    res_corr = -residual @ A
    init_proxy = y @ A
    gram_b = gram_diag.unsqueeze(0).expand(B, -1)
    coh_b = max_coh.unsqueeze(0).expand(B, -1)
    return torch.stack([abs_x, res_corr, init_proxy, gram_b, coh_b], dim=-1)

print("Precomputing ISTA + features (train) ...")
with torch.no_grad():
    y_train = X_train @ A.T
    x_T_train = ista(A, y_train, alpha_ista, lam_ista, T_ista)
    feats_train = make_features(A, y_train, x_T_train, gram_diag, max_coh)

feat_mean = feats_train.reshape(-1, feats_train.shape[-1]).mean(dim=0)
feat_std = feats_train.reshape(-1, feats_train.shape[-1]).std(dim=0).clamp(min=1e-6)
feats_train_n = (feats_train - feat_mean) / feat_std

# ----------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------

class CoordMLP(nn.Module):
    """Baseline: per-coordinate MLP (matches the paper's current detector)."""
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, feats, gram_bias=None):    # gram_bias unused for MLP
        return self.net(feats).squeeze(-1)


class GramBiasedAttention(nn.Module):
    """Multi-head self-attention with additive Gram-matrix bias.

    For each head h:
        scores = (Q K^T) / sqrt(d_h) + alpha_h * gram_bias
        attn   = softmax(scores)
        out    = attn @ V

    alpha_h is a per-head learnable scalar, initialized at 0 so the model
    starts as vanilla attention and learns the Gram-bias scale.
    """
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.alpha = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x, gram_bias):
        B, n_tok, d = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, n_tok, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k_, v = qkv[0], qkv[1], qkv[2]                  # each (B, H, n, hd)
        scores = (q @ k_.transpose(-2, -1)) / (hd ** 0.5)  # (B, H, n, n)
        bias = (self.alpha.view(1, H, 1, 1)
                * gram_bias.unsqueeze(0).unsqueeze(0))     # (1, H, n, n)
        scores = scores + bias
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, n_tok, d)
        return self.out(out)


class GASTBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = GramBiasedAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(),
            nn.Linear(d_ff, d),
        )
    def forward(self, x, gram_bias):
        x = x + self.attn(self.ln1(x), gram_bias)
        x = x + self.ffn(self.ln2(x))
        return x


class GAST(nn.Module):
    """Gram-Aware Support Transformer."""
    def __init__(self, in_dim=5, d=24, n_heads=4, n_layers=2, d_ff=None):
        super().__init__()
        d_ff = d_ff if d_ff is not None else 2 * d
        self.in_proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(
            [GASTBlock(d, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, 1)
    def forward(self, feats, gram_bias):
        x = self.in_proj(feats)
        for blk in self.blocks:
            x = blk(x, gram_bias)
        return self.out_proj(self.out_ln(x)).squeeze(-1)


def make_model(arch):
    if arch == "mlp":
        return CoordMLP(in_dim=5, hidden=args.mlp_hidden).to(device)
    return GAST(in_dim=5, d=args.d, n_heads=args.n_heads,
                n_layers=args.n_layers).to(device)

# ----------------------------------------------------------------------
# Train + eval
# ----------------------------------------------------------------------

def n_params(model):
    return sum(p.numel() for p in model.parameters())

pos_weight = torch.tensor([(n - k) / k], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def gast_alpha_summary(model):
    """For GAST, return per-layer per-head alpha values as a flat list of floats."""
    if not isinstance(model, GAST):
        return None
    out = []
    for li, blk in enumerate(model.blocks):
        out.append((li, blk.attn.alpha.detach().cpu().numpy().tolist()))
    return out

def fmt_alpha(alpha_summary):
    if alpha_summary is None:
        return ""
    parts = []
    for li, vals in alpha_summary:
        parts.append("L{}=[{}]".format(
            li, ",".join(f"{v:+.3f}" for v in vals)))
    return " | alpha " + " ".join(parts)

def train_model(model, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"  training (n_params={n_params(model)}) ...")
    alpha_history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(args.n_train, device=device)
        total = 0.0
        nb = 0
        for start in range(0, args.n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            logits = model(feats_train_n[idx], gram_bias)
            loss = criterion(logits, S_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            nb += 1
        a_summary = gast_alpha_summary(model)
        if a_summary is not None:
            alpha_history.append({"epoch": epoch + 1,
                                  "alpha": [vs for _, vs in a_summary]})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>3}  loss={total/nb:.4f}"
                  f"{fmt_alpha(a_summary)}")
    print(f"  done in {time.time()-t0:.1f}s")
    return alpha_history

# Classical baselines (used at eval time)
def naive_topk(A_np, y, k):
    return np.argpartition(-np.abs(A_np.T @ y), k - 1)[:k]

def omp(A_np, y, k):
    r = y.copy()
    selected = []
    for _ in range(k):
        scores = np.abs(A_np.T @ r)
        for s in selected:
            scores[s] = -np.inf
        j = int(np.argmax(scores))
        selected.append(j)
        A_S = A_np[:, selected]
        x_S, *_ = np.linalg.lstsq(A_S, y, rcond=None)
        r = y - A_S @ x_S
    return np.array(selected)

def cosamp(A_np, y, k, max_iters=30):
    nn_ = A_np.shape[1]
    x = np.zeros(nn_)
    S_prev = set()
    for _ in range(max_iters):
        r = y - A_np @ x
        u = A_np.T @ r
        omega = set(int(i) for i in np.argpartition(-np.abs(u), 2 * k - 1)[:2 * k])
        T = sorted(omega | set(int(i) for i in np.nonzero(x)[0]))
        A_T = A_np[:, T]
        b_T, *_ = np.linalg.lstsq(A_T, y, rcond=None)
        b = np.zeros(nn_)
        b[T] = b_T
        S_new = set(int(i) for i in np.argpartition(-np.abs(b), k - 1)[:k])
        S_list = sorted(S_new)
        x = np.zeros(nn_)
        x[S_list], *_ = np.linalg.lstsq(A_np[:, S_list], y, rcond=None)
        if S_new == S_prev:
            break
        S_prev = S_new
    return np.array(sorted(S_new))

def support_iou(S_pred_idx, S_true_set):
    p = set(int(i) for i in S_pred_idx)
    inter = len(p & S_true_set)
    union = len(p | S_true_set)
    return inter / union if union > 0 else 0.0

def support_ls_nrmse(A_np, y, S_pred_idx, x_true):
    S_list = sorted(int(i) for i in S_pred_idx)
    if not S_list:
        return 1.0
    A_S = A_np[:, S_list]
    x_S, *_ = np.linalg.lstsq(A_S, y, rcond=None)
    x_hat = np.zeros(A_np.shape[1])
    x_hat[S_list] = x_S
    denom = max(np.linalg.norm(x_true), 1e-12)
    return float(np.linalg.norm(x_hat - x_true) / denom)

# Cache classical baseline supports per test signal so we don't recompute
# across architectures
_baselines_cache = {}

def get_baseline_supports(ta):
    if ta in _baselines_cache:
        return _baselines_cache[ta]
    X_te = test_sets[ta]
    out = {"naive": [], "omp": [], "cosamp": [], "oracle": []}
    for i in range(args.n_test):
        x_t = X_te[i]
        y_i = A_np @ x_t
        out["naive"].append(naive_topk(A_np, y_i, k))
        out["omp"].append(omp(A_np, y_i, k))
        out["cosamp"].append(cosamp(A_np, y_i, k))
        out["oracle"].append(np.argpartition(-np.abs(x_t), k - 1)[:k])
    _baselines_cache[ta] = out
    return out

def eval_model(model, ta):
    """Returns per-method dict with mean NRMSE and mean IoU on test set."""
    X_te_np = test_sets[ta]
    X_te = torch.tensor(X_te_np, device=device)
    model.eval()
    with torch.no_grad():
        y_te = X_te @ A.T
        x_T_te = ista(A, y_te, alpha_ista, lam_ista, T_ista)
        feats_te = make_features(A, y_te, x_T_te, gram_diag, max_coh)
        feats_te_n = (feats_te - feat_mean) / feat_std
        logits = model(feats_te_n, gram_bias)
        learned_topk = logits.topk(k, dim=1).indices.cpu().numpy()

    base = get_baseline_supports(ta)

    res = {name: {"nrmse": [], "iou": []}
           for name in ["naive", "omp", "cosamp", "learned", "oracle"]}
    for i in range(args.n_test):
        x_t = X_te_np[i]
        y_i = A_np @ x_t
        S_true = set(int(j) for j in np.argpartition(-np.abs(x_t), k - 1)[:k])
        for name, supp in [("naive",   base["naive"][i]),
                           ("omp",     base["omp"][i]),
                           ("cosamp",  base["cosamp"][i]),
                           ("learned", learned_topk[i]),
                           ("oracle",  base["oracle"][i])]:
            res[name]["nrmse"].append(support_ls_nrmse(A_np, y_i, supp, x_t))
            res[name]["iou"].append(support_iou(supp, S_true))
    return {n_: {"nrmse_mean": float(np.mean(v["nrmse"])),
                 "nrmse_std":  float(np.std(v["nrmse"])),
                 "iou_mean":   float(np.mean(v["iou"])),
                 "iou_std":    float(np.std(v["iou"]))}
            for n_, v in res.items()}

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

archs_to_run = ["mlp", "gast"] if args.arch == "both" else [args.arch]
all_results = {}
alpha_histories = {}

for arch in archs_to_run:
    print(f"\n=== arch = {arch} ===")
    torch.manual_seed(args.init_seed)              # same init seed per arch
    model = make_model(arch)
    hist = train_model(model, args.epochs)
    if hist:
        alpha_histories[arch] = hist
    arch_results = {}
    for ta in eval_tails:
        r = eval_model(model, ta)
        arch_results[ta] = r
        print(f"  tail={ta}:  learned NRMSE={r['learned']['nrmse_mean']:.4f}  "
              f"IoU={r['learned']['iou_mean']:.4f}  "
              f"(CoSaMP NRMSE={r['cosamp']['nrmse_mean']:.4f}  "
              f"IoU={r['cosamp']['iou_mean']:.4f})")
    all_results[arch] = arch_results

# ----------------------------------------------------------------------
# Side-by-side comparison
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print(f"  SIDE-BY-SIDE  (regime={args.regime}, n={n}, m={m}, k={k})")
print("=" * 78)

for ta in eval_tails:
    print(f"\n  tail={ta}")
    print(f"    {'method':<14} {'NRMSE':>10} {'IoU':>10}")
    # show classical baselines from any arch's eval (they're cached and identical)
    ref_arch = archs_to_run[0]
    for name in ["naive", "omp", "cosamp"]:
        r = all_results[ref_arch][ta][name]
        print(f"    {name:<14} {r['nrmse_mean']:>10.4f} {r['iou_mean']:>10.4f}")
    for arch in archs_to_run:
        r = all_results[arch][ta]["learned"]
        tag = "MLP (baseline)" if arch == "mlp" else "GAST (ours)"
        print(f"    {tag:<14} {r['nrmse_mean']:>10.4f} {r['iou_mean']:>10.4f}")
    r_orcl = all_results[ref_arch][ta]["oracle"]
    print(f"    {'oracle':<14} {r_orcl['nrmse_mean']:>10.4f} {r_orcl['iou_mean']:>10.4f}")

# Verdict: GAST vs MLP, GAST vs CoSaMP
print("\n" + "=" * 78)
print("  VERDICT")
print("=" * 78)
if "mlp" in all_results and "gast" in all_results:
    for ta in eval_tails:
        mlp_n  = all_results["mlp"][ta]["learned"]["nrmse_mean"]
        gast_n = all_results["gast"][ta]["learned"]["nrmse_mean"]
        mlp_i  = all_results["mlp"][ta]["learned"]["iou_mean"]
        gast_i = all_results["gast"][ta]["learned"]["iou_mean"]
        cs_n   = all_results["mlp"][ta]["cosamp"]["nrmse_mean"]
        d_mlp  = mlp_n  - gast_n
        d_cos  = cs_n   - gast_n
        d_iou  = gast_i - mlp_i
        print(f"  tail={ta}")
        print(f"    GAST vs MLP:    NRMSE delta = {d_mlp:+.4f}  "
              f"IoU delta = {d_iou:+.4f}  "
              f"({'GAST better' if d_mlp > 0 else 'MLP better'})")
        print(f"    GAST vs CoSaMP: NRMSE delta = {d_cos:+.4f}  "
              f"({'GAST beats CoSaMP' if d_cos > 0.01 else 'GAST ties CoSaMP' if abs(d_cos) <= 0.01 else 'GAST loses to CoSaMP'})")

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

out = Path(__file__).resolve().parent / f"gast_quick_{args.regime}_{args.tag}.json"
with out.open("w") as f:
    json.dump({
        "args": vars(args),
        "n": n, "m": m, "k": k, "mu": mu,
        "results": {arch: {str(t): v for t, v in r.items()}
                    for arch, r in all_results.items()},
        "alpha_history": alpha_histories,
    }, f, indent=2)
print(f"\nWrote {out}")
