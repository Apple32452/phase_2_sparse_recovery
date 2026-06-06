import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


RESULTS_DIR = Path("results/lista_baseline")
FIGURES_DIR = Path("figures/lista_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_matrix(m, n, rng):
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    return A


def make_blocks(n, block_size):
    return [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]


def sample_block_sparse_signal(n, k, block_size, rng):
    blocks = make_blocks(n, block_size)
    q = math.ceil(k / block_size)
    active_blocks = rng.choice(len(blocks), size=q, replace=False)

    support = []
    for b in active_blocks:
        support.extend(blocks[b].tolist())

    support = np.array(support[:k])
    x = np.zeros(n)
    x[support] = rng.normal(size=len(support))
    return x, support


def make_dataset(A, n_samples, k, block_size, rng):
    n = A.shape[1]
    X = np.zeros((n_samples, n), dtype=np.float32)
    Y = np.zeros((n_samples, A.shape[0]), dtype=np.float32)
    supports = []

    for i in range(n_samples):
        x, support = sample_block_sparse_signal(n, k, block_size, rng)
        y = A @ x
        X[i] = x.astype(np.float32)
        Y[i] = y.astype(np.float32)
        supports.append(support)

    return Y, X, supports


def fit_on_support(A, y, support, n):
    support = np.array(sorted(set(support)), dtype=int)
    xhat = np.zeros(n)

    if len(support) == 0:
        return xhat

    As = A[:, support]
    coef, *_ = np.linalg.lstsq(As, y, rcond=None)
    xhat[support] = coef
    return xhat


def topk_refit(A, y, scores, k):
    n = A.shape[1]
    support = np.argsort(np.abs(scores))[-k:]
    return fit_on_support(A, y, support, n)


def nrmse(xhat, x):
    return np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-12)


def support_iou(xhat, support_true, k):
    pred = set(np.argsort(np.abs(xhat))[-k:].tolist())
    true = set(support_true.tolist())
    inter = len(pred & true)
    union = len(pred | true)
    return inter / max(union, 1)


def mean_se(vals):
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


def soft_threshold(z, theta):
    return torch.sign(z) * torch.relu(torch.abs(z) - theta)


class StableLISTA(nn.Module):
    """
    Stable unrolled ISTA baseline.

    x_{t+1} = S_theta_t(x_t + alpha_t A^T(y - A x_t))

    alpha_t is constrained to be in (0, 1/L), where L = ||A||_2^2.
    This prevents the raw LISTA iterates from exploding.
    """
    def __init__(self, A_np, n_layers=20):
        super().__init__()

        A = torch.tensor(A_np, dtype=torch.float32)
        self.register_buffer("A", A)

        L = torch.linalg.norm(A, ord=2) ** 2
        self.register_buffer("L", L)

        self.n_layers = n_layers
        self.n = A.shape[1]

        self.raw_alpha = nn.Parameter(torch.zeros(n_layers))
        self.raw_theta = nn.Parameter(torch.full((n_layers,), -3.0))

    def forward(self, y):
        x = torch.zeros(y.shape[0], self.n, device=y.device, dtype=y.dtype)

        for t in range(self.n_layers):
            # alpha in (0, 1/L)
            alpha = torch.sigmoid(self.raw_alpha[t]) / (self.L + 1e-12)

            # positive threshold
            theta = torch.nn.functional.softplus(self.raw_theta[t])

            residual = y - x @ self.A.T
            z = x + alpha * (residual @ self.A)
            x = soft_threshold(z, theta)

        return x


def train_model(A, Y_train, X_train, n_layers=20, epochs=40, batch_size=128, lr=2e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StableLISTA(A, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    Y = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X = torch.tensor(X_train, dtype=torch.float32).to(device)

    n_train = Y.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n_train, device=device)
        total = 0.0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            yb = Y[idx]
            xb = X[idx]

            pred = model(yb)

            # Reconstruction loss plus small support-sharpening penalty.
            mse = torch.mean((pred - xb) ** 2)
            l1 = 1e-5 * torch.mean(torch.abs(pred))
            loss = mse + l1

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total += float(loss.item()) * len(idx)

        if epoch in {0, epochs - 1}:
            print(f"    epoch {epoch+1:02d}/{epochs}, loss={total / n_train:.6f}")

    return model


def evaluate_model(model, A, Y_test, X_test, supports, k):
    device = next(model.parameters()).device

    with torch.no_grad():
        Y = torch.tensor(Y_test, dtype=torch.float32).to(device)
        pred = model(Y).cpu().numpy()

    raw_nrmse, raw_iou = [], []
    refit_nrmse, refit_iou = [], []

    for i in range(len(X_test)):
        x = X_test[i]
        y = Y_test[i]
        s_true = supports[i]

        x_raw = pred[i]
        raw_nrmse.append(nrmse(x_raw, x))
        raw_iou.append(support_iou(x_raw, s_true, k))

        x_refit = topk_refit(A, y, x_raw, k)
        refit_nrmse.append(nrmse(x_refit, x))
        refit_iou.append(support_iou(x_refit, s_true, k))

    return {
        "stable_lista_raw": {"nrmse": raw_nrmse, "iou": raw_iou},
        "stable_lista_topk_refit": {"nrmse": refit_nrmse, "iou": refit_iou},
    }


def run_setting(m, k, seeds, n=256, block_size=5, n_train=3000, n_test=300):
    all_vals = {
        "stable_lista_raw": {"nrmse": [], "iou": []},
        "stable_lista_topk_refit": {"nrmse": [], "iou": []},
    }

    for seed in seeds:
        print(f"  seed={seed}")
        rng = np.random.default_rng(seed)
        A = make_matrix(m, n, rng)

        Y_train, X_train, _ = make_dataset(A, n_train, k, block_size, rng)
        Y_test, X_test, supports = make_dataset(A, n_test, k, block_size, rng)

        model = train_model(A, Y_train, X_train)

        vals = evaluate_model(model, A, Y_test, X_test, supports, k)

        for method in all_vals:
            all_vals[method]["nrmse"].extend(vals[method]["nrmse"])
            all_vals[method]["iou"].extend(vals[method]["iou"])

    summary = {}

    for method in all_vals:
        nrmse_mean, nrmse_se = mean_se(all_vals[method]["nrmse"])
        iou_mean, iou_se = mean_se(all_vals[method]["iou"])

        summary[method] = {
            "nrmse_mean": nrmse_mean,
            "nrmse_se": nrmse_se,
            "iou_mean": iou_mean,
            "iou_se": iou_se,
        }

    return summary


def main():
    settings = [(96, 40), (96, 55)]
    seeds = list(range(10))

    results = {}

    for m, k in settings:
        print("=" * 80)
        print(f"Stable LISTA baseline: m={m}, k={k}")
        key = f"m={m},k={k}"
        results[key] = run_setting(m, k, seeds)

    out_json = RESULTS_DIR / "stable_lista_baseline_10seed.json"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_json}")

    print("\nStable LISTA baseline results")
    print("-" * 90)
    print(f"{'setting':12s} {'method':28s} {'NRMSE':>10s} {'SE':>10s} {'IoU':>10s}")

    for setting, table in results.items():
        for method, row in table.items():
            print(
                f"{setting:12s} {method:28s} "
                f"{row['nrmse_mean']:10.4f} "
                f"{row['nrmse_se']:10.4f} "
                f"{row['iou_mean']:10.4f}"
            )

    adaptive_ref = {
        "m=96,k=40": 0.0037,
        "m=96,k=55": 0.1087,
    }

    methods = ["stable_lista_raw", "stable_lista_topk_refit", "adaptive_refinement"]
    x = np.arange(len(settings))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(methods):
        means, ses = [], []

        for m, k in settings:
            key = f"m={m},k={k}"

            if method == "adaptive_refinement":
                means.append(adaptive_ref[key])
                ses.append(0.0)
            else:
                means.append(results[key][method]["nrmse_mean"])
                ses.append(results[key][method]["nrmse_se"])

        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=ses,
            capsize=4,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}, k={k}" for m, k in settings])
    ax.set_ylabel("NRMSE")
    ax.set_title("Stable LISTA baseline vs adaptive refinement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_fig = FIGURES_DIR / "stable_lista_baseline_10seed_nrmse.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")

    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
