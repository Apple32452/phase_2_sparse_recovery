"""
Quick test: does a learned operator-aware support detector beat CoSaMP
in the compressible-signal regime, where CoSaMP's strict-k-sparse
assumption is most exposed?

Setup mirrors `cosamp_stress_test.py::sweep_compressible` so results
drop into the same axis:
  - Gaussian A in R^{128 x 256}, k=25
  - Compressible signals: k spikes + Gaussian tail of amplitude `tail`
  - Tail amplitudes evaluated: {0.0, 0.05, 0.1, 0.2}

Detector training distribution: random tail amplitude per signal
~ Uniform[0, 0.2], so it sees the whole regime, not just one slice.
Labels = top-k of |x_full| (= spike support when tail=0; otherwise
includes large tail entries).

Reports: NRMSE (lower better) at each tail amplitude for
naive top-k / OMP / CoSaMP / learned (ours) / oracle (= top-k of |x|).
Verdict line says whether learned beats CoSaMP at any tail.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["mixed", "fixed", "extended"],
                    default="mixed",
                    help="mixed: train Uniform[0, tail_max]; "
                         "fixed: train at single tail value; "
                         "extended: train Uniform[tail_lo, tail_hi]")
parser.add_argument("--tail-train-lo", type=float, default=0.0)
parser.add_argument("--tail-train-hi", type=float, default=0.2)
parser.add_argument("--tail-fixed", type=float, default=0.2)
parser.add_argument("--tail-eval", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2])
parser.add_argument("--tag", type=str, default="mixed",
                    help="suffix for output files")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--op-seed", type=int, default=0,
                    help="seed for the Gaussian operator draw")
parser.add_argument("--init-seed", type=int, default=42,
                    help="seed for torch model initialization")
parser.add_argument("--save-per-signal", action="store_true",
                    help="save per-signal NRMSE arrays in JSON for paired analysis")
args = parser.parse_args()

n, m, k = 256, 128, 25
n_train, n_test = 4000, 500
T_ista = 30
lam_ista = 0.05
batch_size = 128
n_epochs = args.epochs
lr = 1e-3
tail_eval = args.tail_eval
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.init_seed)

print(f"device = {device}")
print(f"n={n}, m={m}, k={k} (matches sweep_compressible axis)")

# Gaussian operator with unit-norm columns (matches learned_above_pt.py)
np_rng = np.random.RandomState(args.op_seed)
A_np = np_rng.randn(m, n).astype(np.float32)
A_np = A_np / np.linalg.norm(A_np, axis=0, keepdims=True)
A = torch.tensor(A_np, device=device)

G_np = A_np.T @ A_np
gram_diag_np = np.diag(G_np).astype(np.float32)
G_off = G_np.copy()
np.fill_diagonal(G_off, 0.0)
max_coh_np = np.max(np.abs(G_off), axis=0).astype(np.float32)
mu = float(np.max(max_coh_np))
print(f"mu(A) = {mu:.4f}")

gram_diag = torch.tensor(gram_diag_np, device=device)
max_coh = torch.tensor(max_coh_np, device=device)

spec_norm = float(np.linalg.norm(A_np, 2))
alpha_ista = 0.95 / spec_norm**2

# ----------------------------------------------------------------------
# Compressible signals
# ----------------------------------------------------------------------

def gen_compressible_batch(n_sig, k, tail_amp_fn, seed):
    """tail_amp_fn(rng) -> scalar tail amplitude (allows random per-signal)."""
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
    """Labels = top-k of |X| per row; returns (n_sig, n) one-hot."""
    Xt = torch.as_tensor(X)
    idx = Xt.abs().topk(k, dim=1).indices
    S = torch.zeros_like(Xt)
    S.scatter_(1, idx, 1.0)
    return S

# Training-set tail-amplitude policy
if args.mode == "fixed":
    tail_desc = f"fixed tail={args.tail_fixed}"
    def tail_policy(rng):
        return float(args.tail_fixed)
elif args.mode == "extended":
    tail_desc = f"Uniform[{args.tail_train_lo}, {args.tail_train_hi}]"
    def tail_policy(rng):
        return float(rng.uniform(args.tail_train_lo, args.tail_train_hi))
else:  # mixed
    tail_desc = f"Uniform[0, {args.tail_train_hi}]"
    def tail_policy(rng):
        return float(rng.uniform(0.0, args.tail_train_hi))

print(f"Mode: {args.mode}  ({tail_desc})")
print(f"Generating {n_train} training signals ...")
X_train_np = gen_compressible_batch(n_train, k, tail_policy, seed=1)
S_train_np = topk_labels(X_train_np, k).numpy()
X_train = torch.tensor(X_train_np, device=device)
S_train = torch.tensor(S_train_np, device=device)

# Test sets: one per tail_eval value (fixed seed -> reproducible)
test_sets = {}
for ta in tail_eval:
    print(f"Generating {n_test} test signals at tail={ta} ...")
    X_te = gen_compressible_batch(n_test, k, lambda rng, t=ta: t, seed=2000 + int(ta * 1000))
    test_sets[ta] = X_te

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
    B = x_T.shape[0]
    abs_x = x_T.abs()
    residual = x_T @ A.T - y
    res_corr = -residual @ A
    init_proxy = y @ A
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

print("Precomputing ISTA + features (train) ...")
with torch.no_grad():
    y_train = X_train @ A.T
    x_T_train = ista(A, y_train, alpha_ista, lam_ista, T_ista)
    feats_train = make_features(A, y_train, x_T_train, gram_diag, max_coh)
feat_mean = feats_train.reshape(-1, feats_train.shape[-1]).mean(dim=0)
feat_std = feats_train.reshape(-1, feats_train.shape[-1]).std(dim=0).clamp(min=1e-6)
feats_train_n = (feats_train - feat_mean) / feat_std

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
        print(f"  epoch {epoch+1:>3}  loss={total_loss/n_batches:.4f}")
print(f"Training done in {time.time() - t0:.1f}s")

# ----------------------------------------------------------------------
# Classical baselines
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Evaluate at each tail amplitude
# ----------------------------------------------------------------------

print(f"\nEvaluating at tail amplitudes {tail_eval} ...\n")
results = {}
for ta in tail_eval:
    X_te_np = test_sets[ta]
    X_te = torch.tensor(X_te_np, device=device)
    with torch.no_grad():
        y_te = X_te @ A.T
        x_T_te = ista(A, y_te, alpha_ista, lam_ista, T_ista)
        feats_te = make_features(A, y_te, x_T_te, gram_diag, max_coh)
        feats_te_n = (feats_te - feat_mean) / feat_std
        logits_te = model(feats_te_n)
        learned_topk = logits_te.topk(k, dim=1).indices.cpu().numpy()

    methods = {"naive_topk": [], "omp": [], "cosamp": [],
               "learned": [], "oracle_topk": []}
    for i in range(n_test):
        x_true = X_te_np[i]
        y_i = A_np @ x_true
        s_naive = naive_topk(A_np, y_i, k)
        s_omp   = omp(A_np, y_i, k)
        s_cos   = cosamp(A_np, y_i, k)
        s_learn = learned_topk[i]
        # oracle for compressible signal = top-k of |x| (best k-support reconstruction)
        s_orcl  = np.argpartition(-np.abs(x_true), k - 1)[:k]
        for name, supp in [("naive_topk", s_naive), ("omp", s_omp),
                           ("cosamp", s_cos), ("learned", s_learn),
                           ("oracle_topk", s_orcl)]:
            methods[name].append(support_ls_nrmse(A_np, y_i, supp, x_true))

    summary = {name: {"nrmse_mean": float(np.mean(v)),
                      "nrmse_std": float(np.std(v))}
               for name, v in methods.items()}
    if args.save_per_signal:
        for name, v in methods.items():
            summary[name]["nrmse_per_signal"] = [float(x) for x in v]
    results[ta] = summary
    print(f"  tail={ta}:")
    for name in ["naive_topk", "omp", "cosamp", "learned", "oracle_topk"]:
        s = summary[name]
        print(f"    {name:<12} NRMSE = {s['nrmse_mean']:.4f} +/- {s['nrmse_std']:.4f}")

# ----------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)
wins = []
ties = []
losses = []
for ta in tail_eval:
    cs = results[ta]["cosamp"]["nrmse_mean"]
    ln = results[ta]["learned"]["nrmse_mean"]
    rel = (cs - ln) / max(cs, 1e-9) * 100
    if ln < cs - 0.01:
        wins.append((ta, cs, ln, rel))
    elif ln < cs + 0.01:
        ties.append((ta, cs, ln, rel))
    else:
        losses.append((ta, cs, ln, rel))
    tag = "WIN" if ln < cs - 0.01 else "TIE" if ln < cs + 0.01 else "LOSS"
    print(f"  tail={ta:<5}  CoSaMP={cs:.4f}  learned={ln:.4f}  "
          f"delta={cs-ln:+.4f}  ({tag}, {rel:+.1f}% rel)")

if wins:
    print(f"\n  >>> Learned BEATS CoSaMP at: {[w[0] for w in wins]}")
elif ties:
    print(f"\n  >>> Learned TIES CoSaMP at: {[t[0] for t in ties]}")
else:
    print("\n  >>> Learned never beats or ties CoSaMP in this sweep.")

# ----------------------------------------------------------------------
# Plot + save
# ----------------------------------------------------------------------

out_dir = Path(__file__).resolve().parent
out_png = out_dir / f"learned_compressible_{args.tag}.png"
out_json = out_dir / f"learned_compressible_{args.tag}.json"

methods_order = ["naive_topk", "omp", "cosamp", "learned", "oracle_topk"]
colors = {"naive_topk": "#bdbdbd", "omp": "#d95f02",
          "cosamp": "#e7298a", "learned": "#1f78b4", "oracle_topk": "#7570b3"}
labels_pretty = {"naive_topk": "naive top-k", "omp": "OMP",
                 "cosamp": "CoSaMP", "learned": "learned (ours)",
                 "oracle_topk": "oracle (top-k of |x|)"}

fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.5))
xs = list(tail_eval)
for mname in methods_order:
    ys = [results[ta][mname]["nrmse_mean"] for ta in tail_eval]
    ax.plot(xs, ys, marker="o", color=colors[mname],
            label=labels_pretty[mname], lw=1.6, ms=5)
ax.set_xlabel("off-support tail amplitude")
ax.set_ylabel("NRMSE")
ax.set_title(f"Compressible-signal regime (Gaussian, k={k}, m={m}, n={n})")
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(out_png, dpi=180)
print(f"\nWrote {out_png}")

with out_json.open("w") as f:
    json.dump({
        "config": {"n": n, "m": m, "k": k, "T_ista": T_ista,
                   "lam_ista": lam_ista, "n_train": n_train,
                   "n_test": n_test, "epochs": n_epochs,
                   "n_params": n_params, "mu": mu,
                   "mode": args.mode, "tail_desc": tail_desc,
                   "tail_eval": tail_eval, "tag": args.tag,
                   "op_seed": args.op_seed, "init_seed": args.init_seed},
        "results": {str(ta): summ for ta, summ in results.items()},
        "wins": [{"tail": ta, "cosamp": cs, "learned": ln, "rel_pct": rel}
                 for ta, cs, ln, rel in wins],
        "ties": [{"tail": ta, "cosamp": cs, "learned": ln, "rel_pct": rel}
                 for ta, cs, ln, rel in ties],
    }, f, indent=2)
print(f"Wrote {out_json}")
