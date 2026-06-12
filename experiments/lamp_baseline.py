import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("Please install PyTorch first: pip install torch")


RESULTS_DIR = Path("results/lamp_baseline")
FIGURES_DIR = Path("figures/lamp_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_matrix(m, n, rng):
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    return A.astype(np.float32)


def make_blocks(n, block_size):
    n_full = (n // block_size) * block_size
    return [np.arange(i, i + block_size) for i in range(0, n_full, block_size)]


def sample_block_sparse_signal(n, k, block_size, rng):
    blocks = make_blocks(n, block_size)
    q = k // block_size
    active_blocks = rng.choice(len(blocks), size=q, replace=False)

    support = []
    for b in active_blocks:
        support.extend(blocks[b].tolist())

    support = np.array(sorted(support), dtype=int)

    x = np.zeros(n, dtype=np.float32)
    x[support] = rng.normal(size=len(support)).astype(np.float32)

    return x, support


def make_dataset(A, n_samples, k, block_size, rng):
    n = A.shape[1]
    m = A.shape[0]

    X = np.zeros((n_samples, n), dtype=np.float32)
    Y = np.zeros((n_samples, m), dtype=np.float32)
    supports = []

    for i in range(n_samples):
        x, support = sample_block_sparse_signal(n, k, block_size, rng)
        y = A @ x

        X[i] = x
        Y[i] = y
        supports.append(support)

    return Y, X, supports


def fit_on_support(A, y, support, n, ridge=1e-8):
    support = np.array(sorted(set(support)), dtype=int)
    xhat = np.zeros(n, dtype=np.float32)

    if len(support) == 0:
        return xhat

    As = A[:, support]

    try:
        coef, *_ = np.linalg.lstsq(As, y, rcond=1e-10)
    except np.linalg.LinAlgError:
        G = As.T @ As
        b = As.T @ y
        reg = ridge * np.trace(G).real / max(G.shape[0], 1)
        if reg <= 0 or not np.isfinite(reg):
            reg = ridge
        coef = np.linalg.solve(G + reg * np.eye(G.shape[0]), b)

    xhat[support] = coef.astype(np.float32)
    return xhat


def nrmse(xhat, x):
    return float(np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-12))


def support_iou_from_xhat(xhat, support_true, k):
    pred = set(np.argsort(np.abs(xhat))[-k:].tolist())
    true = set(support_true.tolist())
    inter = len(pred & true)
    union = len(pred | true)
    return inter / max(union, 1)


def topk_refit(A, y, scores, k):
    n = A.shape[1]
    support = np.argsort(np.abs(scores))[-k:]
    xhat = fit_on_support(A, y, support, n)
    return xhat, np.array(sorted(support), dtype=int)


def soft_threshold(z, theta):
    return torch.sign(z) * torch.relu(torch.abs(z) - theta)


def cosamp(A, y, k, n_iters=20):
    m, n = A.shape
    residual = y.copy()
    support = set()
    xhat = np.zeros(n, dtype=np.float32)

    for _ in range(n_iters):
        proxy = A.T @ residual
        omega = set(np.argsort(np.abs(proxy))[-2 * k:].tolist())
        merged = np.array(sorted(support | omega), dtype=int)

        b = fit_on_support(A, y, merged, n)
        support = set(np.argsort(np.abs(b))[-k:].tolist())

        xhat = fit_on_support(A, y, support, n)
        residual = y - A @ xhat

        if np.linalg.norm(residual) < 1e-10:
            break

    return xhat


class LAMPNet(nn.Module):
    """
    Practical LAMP-style learned AMP baseline.

    Layer update:
        pseudo_t = x_t + alpha_t A^T v_t
        x_{t+1} = soft_threshold(pseudo_t, lambda_t * ||v_t|| / sqrt(m))
        v_{t+1} = y - A x_{t+1} + beta_t * div_t / delta * v_t

    This is a learned message-passing baseline for Gaussian sensing.
    """

    def __init__(self, A_np, n_layers=20):
        super().__init__()

        A = torch.tensor(A_np, dtype=torch.float32)
        self.register_buffer("A", A)

        L = torch.linalg.norm(A, ord=2) ** 2
        self.register_buffer("L", L)

        self.m = A.shape[0]
        self.n = A.shape[1]
        self.delta = self.m / self.n
        self.n_layers = n_layers

        self.raw_alpha = nn.Parameter(torch.zeros(n_layers))
        self.raw_lambda = nn.Parameter(torch.full((n_layers,), 0.0))
        self.raw_beta = nn.Parameter(torch.full((n_layers,), 2.5))

    def forward(self, y):
        batch = y.shape[0]
        x = torch.zeros(batch, self.n, dtype=y.dtype, device=y.device)
        v = y.clone()

        for t in range(self.n_layers):
            alpha = torch.sigmoid(self.raw_alpha[t]) * 1.8 / (self.L + 1e-12)
            lam = torch.nn.functional.softplus(self.raw_lambda[t])
            beta = torch.sigmoid(self.raw_beta[t])

            sigma = torch.linalg.norm(v, dim=1, keepdim=True) / math.sqrt(self.m)
            theta = lam * sigma

            pseudo = x + alpha * (v @ self.A)
            x_new = soft_threshold(pseudo, theta)

            div = torch.mean((torch.abs(pseudo) > theta).float(), dim=1, keepdim=True)
            v_new = y - x_new @ self.A.T + beta * (div / max(self.delta, 1e-12)) * v

            # Mild damping for finite-dimensional stability.
            x = 0.8 * x_new + 0.2 * x
            v = 0.8 * v_new + 0.2 * v

        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_lamp(
    A,
    Y_train,
    X_train,
    n_layers=20,
    epochs=40,
    batch_size=128,
    lr=2e-3,
):
    device = get_device()

    model = LAMPNet(A, n_layers=n_layers).to(device)
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

            mse = torch.mean((pred - xb) ** 2)
            sparse_penalty = torch.mean(torch.abs(pred))
            loss = mse + 1e-6 * sparse_penalty

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total += float(loss.item()) * len(idx)

        if epoch in {0, epochs - 1}:
            print(f"    epoch {epoch + 1:02d}/{epochs}, loss={total / n_train:.6f}")

    return model


def evaluate_lamp_model(model, A, Y_test, X_test, supports, k):
    device = next(model.parameters()).device

    with torch.no_grad():
        Y = torch.tensor(Y_test, dtype=torch.float32).to(device)
        pred = model(Y).cpu().numpy()

    values = {
        "lamp_raw": {"nrmse": [], "iou": []},
        "lamp_topk_refit": {"nrmse": [], "iou": []},
    }

    for i in range(len(X_test)):
        x = X_test[i]
        y = Y_test[i]
        support_true = supports[i]

        x_raw = pred[i]
        values["lamp_raw"]["nrmse"].append(nrmse(x_raw, x))
        values["lamp_raw"]["iou"].append(support_iou_from_xhat(x_raw, support_true, k))

        x_refit, _ = topk_refit(A, y, x_raw, k)
        values["lamp_topk_refit"]["nrmse"].append(nrmse(x_refit, x))
        values["lamp_topk_refit"]["iou"].append(support_iou_from_xhat(x_refit, support_true, k))

    return values


def mean_se(vals):
    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, se


def run_setting(
    m,
    k,
    seeds,
    n=256,
    block_size=5,
    n_train=3000,
    n_test=300,
    n_layers=20,
    epochs=40,
):
    all_vals = {
        "cosamp": {"nrmse": [], "iou": []},
        "lamp_raw": {"nrmse": [], "iou": []},
        "lamp_topk_refit": {"nrmse": [], "iou": []},
        "oracle": {"nrmse": [], "iou": []},
    }

    for seed in seeds:
        print(f"  seed={seed}")
        rng = np.random.default_rng(seed)
        A = make_matrix(m, n, rng)

        Y_train, X_train, _ = make_dataset(A, n_train, k, block_size, rng)
        Y_test, X_test, supports = make_dataset(A, n_test, k, block_size, rng)

        model = train_lamp(
            A,
            Y_train,
            X_train,
            n_layers=n_layers,
            epochs=epochs,
            batch_size=128,
            lr=2e-3,
        )

        vals = evaluate_lamp_model(model, A, Y_test, X_test, supports, k)

        for method in ["lamp_raw", "lamp_topk_refit"]:
            all_vals[method]["nrmse"].extend(vals[method]["nrmse"])
            all_vals[method]["iou"].extend(vals[method]["iou"])

        for i in range(len(X_test)):
            x = X_test[i]
            y = Y_test[i]
            support_true = supports[i]

            xhat = cosamp(A, y, k)
            all_vals["cosamp"]["nrmse"].append(nrmse(xhat, x))
            all_vals["cosamp"]["iou"].append(support_iou_from_xhat(xhat, support_true, k))

            xhat = fit_on_support(A, y, support_true, n)
            all_vals["oracle"]["nrmse"].append(nrmse(xhat, x))
            all_vals["oracle"]["iou"].append(1.0)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    args = parser.parse_args()

    if args.quick:
        seeds = [0]
        n_train = 500
        n_test = 100
        epochs = 5
        n_layers = 10
        suffix = "quick"
    else:
        seeds = list(range(10))
        n_train = 3000
        n_test = 300
        epochs = 40
        n_layers = 20
        suffix = "10seed"

    settings = [(96, 40), (96, 55)]
    results = {}

    for m, k in settings:
        print("=" * 80)
        print(f"LAMP baseline: m={m}, k={k}")
        key = f"m={m},k={k}"
        results[key] = run_setting(
            m,
            k,
            seeds,
            n=256,
            block_size=5,
            n_train=n_train,
            n_test=n_test,
            n_layers=n_layers,
            epochs=epochs,
        )

    out_json = RESULTS_DIR / f"lamp_baseline_{suffix}.json"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_json}")

    print("\nLAMP baseline results")
    print("-" * 100)
    print(f"{'setting':12s} {'method':24s} {'NRMSE':>10s} {'SE':>10s} {'IoU':>10s}")

    for setting, table in results.items():
        for method, row in table.items():
            print(
                f"{setting:12s} {method:24s} "
                f"{row['nrmse_mean']:10.4f} "
                f"{row['nrmse_se']:10.4f} "
                f"{row['iou_mean']:10.4f}"
            )

    adaptive_ref = {
        "m=96,k=40": {"nrmse_mean": 0.0037, "nrmse_se": 0.0006},
        "m=96,k=55": {"nrmse_mean": 0.1087, "nrmse_se": 0.0038},
    }

    amp_ref = {
        "m=96,k=40": {"nrmse_mean": 0.2468, "nrmse_se": 0.0028},
        "m=96,k=55": {"nrmse_mean": 0.5368, "nrmse_se": 0.0024},
    }

    block_lista_ref = {
        "m=96,k=40": {"nrmse_mean": 0.2610, "nrmse_se": 0.0031},
        "m=96,k=55": {"nrmse_mean": 0.4452, "nrmse_se": 0.0030},
    }

    plot_methods = [
        "cosamp",
        "amp_topk_refit",
        "block_lista_refit",
        "lamp_raw",
        "lamp_topk_refit",
        "adaptive_refinement",
        "oracle",
    ]

    labels = {
        "cosamp": "CoSaMP",
        "amp_topk_refit": "AMP top-k",
        "block_lista_refit": "Block-LISTA refit",
        "lamp_raw": "LAMP raw",
        "lamp_topk_refit": "LAMP top-k refit",
        "adaptive_refinement": "adaptive",
        "oracle": "oracle",
    }

    xloc = np.arange(len(settings))
    width = 0.11

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, method in enumerate(plot_methods):
        means = []
        ses = []

        for m, k in settings:
            key = f"m={m},k={k}"

            if method == "adaptive_refinement":
                means.append(adaptive_ref[key]["nrmse_mean"])
                ses.append(adaptive_ref[key]["nrmse_se"])
            elif method == "amp_topk_refit":
                means.append(amp_ref[key]["nrmse_mean"])
                ses.append(amp_ref[key]["nrmse_se"])
            elif method == "block_lista_refit":
                means.append(block_lista_ref[key]["nrmse_mean"])
                ses.append(block_lista_ref[key]["nrmse_se"])
            else:
                means.append(results[key][method]["nrmse_mean"])
                ses.append(results[key][method]["nrmse_se"])

        ax.bar(
            xloc + (i - 3) * width,
            means,
            width,
            yerr=ses,
            capsize=4,
            label=labels[method],
        )

    ax.set_xticks(xloc)
    ax.set_xticklabels([f"m={m}, k={k}" for m, k in settings])
    ax.set_ylabel("NRMSE")
    ax.set_title("LAMP learned AMP baseline vs adaptive refinement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_fig = FIGURES_DIR / f"lamp_baseline_{suffix}_nrmse.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")

    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
