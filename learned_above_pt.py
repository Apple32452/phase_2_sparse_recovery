"""
Prototype: learned operator-aware support detector above the sparsity
phase transition.

Setting: Gaussian A in R^{128x256}, k=55  (above Gaussian phase transition,
where CoSaMP's NRMSE is ~0.56 vs oracle ~0).  Train a coord-wise MLP on
operator-aware features and compare against naive top-k, OMP, CoSaMP,
and oracle on a held-out test set.

Per-coordinate features fed to the detector:
  - |x^(T)_j|                              iterate magnitude after T ISTA steps
  - A_{:,j}^T (y - A x^(T))                residual correlation
  - A_{:,j}^T y                            initial proxy
  - (A^T A)_{jj}                           Gram diagonal
  - max_{i != j} |A_{:,i}^T A_{:,j}|       max off-diagonal coherence

The first three are signal-dependent and computed per signal. The last
two are operator-only and constant across signals (precomputed once).
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

n, m, k = 256, 128, 55
n_train, n_test = 4000, 500
T_ista = 30
lam_ista = 0.05
batch_size = 128
n_epochs = 80
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

print(f"device = {device}")
print(f"n={n}, m={m}, k={k} (above Gaussian phase transition)")

# Gaussian operator with unit-norm columns
np_rng = np.random.RandomState(0)
A_np = np_rng.randn(m, n).astype(np.float32)
A_np = A_np / np.linalg.norm(A_np, axis=0, keepdims=True)
A = torch.tensor(A_np, device=device)

# Operator-only features (precomputed)
G_np = A_np.T @ A_np
gram_diag_np = np.diag(G_np).astype(np.float32)
G_off = G_np.copy()
np.fill_diagonal(G_off, 0.0)
max_coh_np = np.max(np.abs(G_off), axis=0).astype(np.float32)
mu = float(np.max(max_coh_np))
print(f"mu(A) = {mu:.4f}, mean max-coherence per col = {max_coh_np.mean():.4f}")

gram_diag = torch.tensor(gram_diag_np, device=device)
max_coh = torch.tensor(max_coh_np, device=device)

spec_norm = float(np.linalg.norm(A_np, 2))
alpha_ista = 0.95 / spec_norm**2

# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------

def gen_signals(n_signals, k, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_signals, n), dtype=np.float32)
    S = np.zeros((n_signals, n), dtype=np.float32)
    for i in range(n_signals):
        supp = rng.choice(n, size=k, replace=False)
        amps = rng.uniform(0.5, 2.0, size=k) * rng.choice([-1, 1], size=k)
        X[i, supp] = amps
        S[i, supp] = 1.0
    return X, S

X_train_np, S_train_np = gen_signals(n_train, k, seed=1)
X_test_np, S_test_np = gen_signals(n_test, k, seed=2)

X_train = torch.tensor(X_train_np, device=device)
S_train = torch.tensor(S_train_np, device=device)
X_test = torch.tensor(X_test_np, device=device)
S_test = torch.tensor(S_test_np, device=device)

# ----------------------------------------------------------------------
# ISTA + features
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
    """Returns (B, n, F) feature tensor."""
    B = x_T.shape[0]
    abs_x = x_T.abs()
    residual = x_T @ A.T - y                  # (B, m)
    res_corr = -residual @ A                   # (B, n) = A^T (y - Ax)
    init_proxy = y @ A                         # (B, n) = A^T y
    gram_b = gram_diag.unsqueeze(0).expand(B, -1)
    coh_b = max_coh.unsqueeze(0).expand(B, -1)
    return torch.stack([abs_x, res_corr, init_proxy, gram_b, coh_b], dim=-1)

# ----------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------

class CoordMLP(nn.Module):
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, feats):
        return self.net(feats).squeeze(-1)

model = CoordMLP(in_dim=5, hidden=64).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"detector params = {n_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
pos_weight = torch.tensor([(n - k) / k], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Precompute features (operator and ISTA steps are fixed during training)
print("Precomputing ISTA iterates and features ...")
with torch.no_grad():
    y_train = X_train @ A.T
    x_T_train = ista(A, y_train, alpha_ista, lam_ista, T_ista)
    feats_train = make_features(A, y_train, x_T_train, gram_diag, max_coh)
    y_test = X_test @ A.T
    x_T_test = ista(A, y_test, alpha_ista, lam_ista, T_ista)
    feats_test = make_features(A, y_test, x_T_test, gram_diag, max_coh)

# Per-feature normalization (across signals and coords) for stable training
feat_mean = feats_train.reshape(-1, feats_train.shape[-1]).mean(dim=0)
feat_std = feats_train.reshape(-1, feats_train.shape[-1]).std(dim=0).clamp(min=1e-6)
feats_train_n = (feats_train - feat_mean) / feat_std
feats_test_n = (feats_test - feat_mean) / feat_std

print(f"feature mean = {feat_mean.cpu().numpy()}")
print(f"feature std  = {feat_std.cpu().numpy()}")

# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

print("\nTraining ...")
t0 = time.time()
for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(n_train, device=device)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n_train, batch_size):
        idx = perm[start:start + batch_size]
        logits = model(feats_train_n[idx])
        loss = criterion(logits, S_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            logits_te = model(feats_test_n)
            top_k_idx = logits_te.topk(k, dim=1).indices
            S_pred = torch.zeros_like(S_test)
            S_pred.scatter_(1, top_k_idx, 1.0)
            inter = (S_pred * S_test).sum(dim=1)
            union = ((S_pred + S_test) > 0).float().sum(dim=1)
            iou = (inter / union).mean().item()
        print(f"  epoch {epoch+1:>3}  loss={total_loss/n_batches:.4f}  test IoU={iou:.4f}")
print(f"Training done in {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Eval: collect supports from all methods, compute NRMSE via support-LS
# ----------------------------------------------------------------------

def support_iou(S_pred_idx, S_true_set):
    p = set(int(i) for i in S_pred_idx)
    inter = len(p & S_true_set)
    union = len(p | S_true_set)
    return inter / union if union > 0 else 0.0

def support_ls_nrmse(A_np, y_np, S_pred_idx, x_true):
    S_list = sorted(int(i) for i in S_pred_idx)
    if len(S_list) == 0:
        return 1.0
    A_S = A_np[:, S_list]
    x_S, *_ = np.linalg.lstsq(A_S, y_np, rcond=None)
    x_hat = np.zeros(A_np.shape[1])
    x_hat[S_list] = x_S
    denom = max(np.linalg.norm(x_true), 1e-12)
    return float(np.linalg.norm(x_hat - x_true) / denom)

# Classical baselines
def naive_topk(A_np, y, k):
    scores = np.abs(A_np.T @ y)
    return np.argpartition(-scores, k - 1)[:k]

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

# Precompute learned detector supports on test set
model.eval()
with torch.no_grad():
    logits_te = model(feats_test_n)
    learned_topk = logits_te.topk(k, dim=1).indices.cpu().numpy()

print("\nEvaluating all methods on 500 test signals ...")
methods = {"naive_topk": [], "omp": [], "cosamp": [], "learned": [], "oracle": []}
ious = {"naive_topk": [], "omp": [], "cosamp": [], "learned": [], "oracle": []}

for i in range(n_test):
    x_true = X_test_np[i]
    S_true = set(int(j) for j in np.nonzero(S_test_np[i])[0])
    y_i = A_np @ x_true

    s_naive = naive_topk(A_np, y_i, k)
    s_omp   = omp(A_np, y_i, k)
    s_cos   = cosamp(A_np, y_i, k)
    s_learn = learned_topk[i]
    s_orcl  = np.array(sorted(S_true))

    for name, supp in [("naive_topk", s_naive), ("omp", s_omp),
                       ("cosamp", s_cos), ("learned", s_learn),
                       ("oracle", s_orcl)]:
        methods[name].append(support_ls_nrmse(A_np, y_i, supp, x_true))
        ious[name].append(support_iou(supp, S_true))

# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print(f" Results: Gaussian m={m}, n={n}, k={k}, n_test={n_test}")
print("=" * 70)
print(f"\n  {'method':<14} {'NRMSE':>16}    {'support IoU':>16}")
summary = {}
for name in ["naive_topk", "omp", "cosamp", "learned", "oracle"]:
    nrm = np.array(methods[name])
    iu  = np.array(ious[name])
    summary[name] = {
        "nrmse_mean": float(nrm.mean()),
        "nrmse_std":  float(nrm.std()),
        "iou_mean":   float(iu.mean()),
        "iou_std":    float(iu.std()),
    }
    print(f"  {name:<14}  {nrm.mean():.4f} ± {nrm.std():.4f}   "
          f"{iu.mean():.4f} ± {iu.std():.4f}")

# Verdict
cs = summary["cosamp"]["nrmse_mean"]
ln = summary["learned"]["nrmse_mean"]
nv = summary["naive_topk"]["nrmse_mean"]
print("\n=== Verdict ===")
print(f"  Learned detector NRMSE : {ln:.4f}")
print(f"  CoSaMP NRMSE           : {cs:.4f}")
print(f"  Naive top-k NRMSE      : {nv:.4f}")
if ln < cs - 0.02:
    verdict = f"WINS: learned beats CoSaMP by {(cs - ln)/cs*100:.1f}% relative."
elif ln < cs + 0.02:
    verdict = f"TIES CoSaMP (within 2%); architecture is competitive but not dominant."
elif ln < nv:
    verdict = f"Beats naive top-k but not CoSaMP. Headroom exists for richer methods."
else:
    verdict = f"Fails to beat naive top-k. Simple coord-wise MLP cannot crack this regime."
print(f"  -> {verdict}")

# ----------------------------------------------------------------------
# Plot + save
# ----------------------------------------------------------------------

out_dir = Path(__file__).resolve().parent
out_png = out_dir / "learned_above_pt.png"
out_json = out_dir / "learned_above_pt.json"

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
order = ["naive_topk", "omp", "cosamp", "learned", "oracle"]
colors = {"naive_topk": "#bdbdbd", "omp": "#d95f02",
          "cosamp": "#e7298a", "learned": "#1f78b4", "oracle": "#7570b3"}
labels = ["naive\ntop-k", "OMP", "CoSaMP", "learned\n(ours)", "oracle"]

ious_mean = [summary[m]["iou_mean"] for m in order]
nrm_mean = [summary[m]["nrmse_mean"] for m in order]
bar_colors = [colors[m] for m in order]
xs = np.arange(len(order))

axes[0].bar(xs, ious_mean, color=bar_colors, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(xs)
axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel("Support IoU")
axes[0].set_ylim(0, 1.05)
axes[0].set_title(f"Support recovery (k={k})", fontsize=10)
axes[0].grid(True, axis="y", alpha=0.3, linewidth=0.5)

axes[1].bar(xs, nrm_mean, color=bar_colors, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(xs)
axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylabel("NRMSE (support-LS)")
axes[1].set_title(f"Reconstruction error (k={k})", fontsize=10)
axes[1].grid(True, axis="y", alpha=0.3, linewidth=0.5)

fig.tight_layout(pad=0.5)
fig.savefig(out_png, dpi=180)

with out_json.open("w") as f:
    json.dump({
        "config": {"n": n, "m": m, "k": k, "T_ista": T_ista, "lam_ista": lam_ista,
                   "n_train": n_train, "n_test": n_test, "epochs": n_epochs,
                   "n_params": n_params, "mu": mu},
        "summary": summary,
        "verdict": verdict,
    }, f, indent=2)

print(f"\nWrote {out_png}")
print(f"Wrote {out_json}")
