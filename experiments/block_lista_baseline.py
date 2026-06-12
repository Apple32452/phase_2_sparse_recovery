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


RESULTS_DIR = Path("results/block_lista_baseline")
FIGURES_DIR = Path("figures/block_lista_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def make_matrix(m, n, rng):
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    return A


def make_blocks(n, block_size):
    """
    Use only full blocks to avoid the final incomplete block issue for n=256, block_size=5.
    The final leftover coordinate is simply never activated.
    """
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


def fit_on_support(A, y, support, n, ridge=1e-8):
    support = np.array(sorted(set(support)), dtype=int)
    xhat = np.zeros(n)

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

    xhat[support] = coef
    return xhat


def nrmse(xhat, x):
    return float(np.linalg.norm(xhat - x) / (np.linalg.norm(x) + 1e-12))


def support_iou_from_xhat(xhat, support_true, k):
    pred = set(np.argsort(np.abs(xhat))[-k:].tolist())
    true = set(support_true.tolist())

    inter = len(pred & true)
    union = len(pred | true)

    return inter / max(union, 1)


def support_iou_from_support(pred_support, support_true):
    pred = set(pred_support.tolist())
    true = set(support_true.tolist())

    inter = len(pred & true)
    union = len(pred | true)

    return inter / max(union, 1)


def block_scores_from_vector(v, blocks):
    abs_v = np.abs(v)
    return np.array([float(np.linalg.norm(abs_v[block], ord=2)) for block in blocks])


def support_from_top_blocks(scores, blocks, q):
    chosen_blocks = np.argsort(scores)[-q:]
    support = []

    for b in chosen_blocks:
        support.extend(blocks[b].tolist())

    return np.array(sorted(support), dtype=int)


def block_topk_refit(A, y, scores, k, block_size):
    n = A.shape[1]
    blocks = make_blocks(n, block_size)
    q = k // block_size

    block_scores = block_scores_from_vector(scores, blocks)
    support = support_from_top_blocks(block_scores, blocks, q)

    return fit_on_support(A, y, support, n), support


def cosamp(A, y, k, n_iters=20):
    m, n = A.shape
    residual = y.copy()
    support = set()
    xhat = np.zeros(n)

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


def group_soft_threshold(x, theta, block_size):
    """
    Group soft thresholding over contiguous full blocks.

    For each block B:
        x_B <- max(1 - theta / ||x_B||_2, 0) x_B
    """
    batch, n = x.shape
    n_full = (n // block_size) * block_size
    n_blocks = n_full // block_size

    x_full = x[:, :n_full].reshape(batch, n_blocks, block_size)
    norms = torch.linalg.norm(x_full, dim=2, keepdim=True)

    scale = torch.relu(1.0 - theta / (norms + 1e-12))
    x_thresh = x_full * scale

    out = torch.zeros_like(x)
    out[:, :n_full] = x_thresh.reshape(batch, n_full)
    return out


class BlockLISTA(nn.Module):
    """
    Stable Block-LISTA / Group-LISTA.

    x_{t+1} = GroupSoftThreshold(
        x_t + alpha_t A^T(y - A x_t),
        theta_t
    )

    alpha_t is constrained by the Lipschitz constant ||A||_2^2.
    """
    def __init__(self, A_np, block_size=5, n_layers=20):
        super().__init__()

        A = torch.tensor(A_np, dtype=torch.float32)
        self.register_buffer("A", A)

        L = torch.linalg.norm(A, ord=2) ** 2
        self.register_buffer("L", L)

        self.n = A.shape[1]
        self.block_size = block_size
        self.n_layers = n_layers

        self.raw_alpha = nn.Parameter(torch.zeros(n_layers))
        self.raw_theta = nn.Parameter(torch.full((n_layers,), -3.0))

    def forward(self, y):
        x = torch.zeros(y.shape[0], self.n, device=y.device, dtype=y.dtype)

        for t in range(self.n_layers):
            alpha = torch.sigmoid(self.raw_alpha[t]) / (self.L + 1e-12)
            theta = torch.nn.functional.softplus(self.raw_theta[t])

            residual = y - x @ self.A.T
            z = x + alpha * (residual @ self.A)

            x = group_soft_threshold(z, theta, self.block_size)

        return x


def train_block_lista(
    A,
    Y_train,
    X_train,
    block_size=5,
    n_layers=20,
    epochs=40,
    batch_size=128,
    lr=2e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BlockLISTA(A, block_size=block_size, n_layers=n_layers).to(device)
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

            # Small group sparsity penalty for stability.
            n = pred.shape[1]
            n_full = (n // block_size) * block_size
            n_blocks = n_full // block_size
            pred_blocks = pred[:, :n_full].reshape(pred.shape[0], n_blocks, block_size)
            group_l1 = torch.mean(torch.linalg.norm(pred_blocks, dim=2))

            loss = mse + 1e-5 * group_l1

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total += float(loss.item()) * len(idx)

        if epoch in {0, epochs - 1}:
            print(f"    epoch {epoch + 1:02d}/{epochs}, loss={total / n_train:.6f}")

    return model


def evaluate_model(model, A, Y_test, X_test, supports, k, block_size):
    device = next(model.parameters()).device

    with torch.no_grad():
        Y = torch.tensor(Y_test, dtype=torch.float32).to(device)
        pred = model(Y).cpu().numpy()

    values = {
        "block_lista_raw": {"nrmse": [], "iou": []},
        "block_lista_block_refit": {"nrmse": [], "iou": []},
    }

    for i in range(len(X_test)):
        x = X_test[i]
        y = Y_test[i]
        support_true = supports[i]

        x_raw = pred[i]
        values["block_lista_raw"]["nrmse"].append(nrmse(x_raw, x))
        values["block_lista_raw"]["iou"].append(support_iou_from_xhat(x_raw, support_true, k))

        x_refit, pred_support = block_topk_refit(A, y, x_raw, k, block_size)
        values["block_lista_block_refit"]["nrmse"].append(nrmse(x_refit, x))
        values["block_lista_block_refit"]["iou"].append(
            support_iou_from_support(pred_support, support_true)
        )

    return values


def mean_se(vals):
    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, se


def run_setting(m, k, seeds, n=256, block_size=5, n_train=3000, n_test=300):
    all_vals = {
        "cosamp": {"nrmse": [], "iou": []},
        "block_lista_raw": {"nrmse": [], "iou": []},
        "block_lista_block_refit": {"nrmse": [], "iou": []},
        "oracle": {"nrmse": [], "iou": []},
    }

    for seed in seeds:
        print(f"  seed={seed}")
        rng = np.random.default_rng(seed)
        A = make_matrix(m, n, rng)

        Y_train, X_train, _ = make_dataset(A, n_train, k, block_size, rng)
        Y_test, X_test, supports = make_dataset(A, n_test, k, block_size, rng)

        model = train_block_lista(
            A,
            Y_train,
            X_train,
            block_size=block_size,
            n_layers=20,
            epochs=40,
            batch_size=128,
            lr=2e-3,
        )

        # Learned block baseline.
        vals = evaluate_model(model, A, Y_test, X_test, supports, k, block_size)

        for method in ["block_lista_raw", "block_lista_block_refit"]:
            all_vals[method]["nrmse"].extend(vals[method]["nrmse"])
            all_vals[method]["iou"].extend(vals[method]["iou"])

        # CoSaMP and oracle baselines on the same test samples.
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
    settings = [(96, 40), (96, 55)]
    seeds = list(range(10))

    results = {}

    for m, k in settings:
        print("=" * 80)
        print(f"Block-LISTA baseline: m={m}, k={k}")
        key = f"m={m},k={k}"
        results[key] = run_setting(m, k, seeds)

    out_json = RESULTS_DIR / "block_lista_baseline_10seed.json"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_json}")

    print("\nBlock-LISTA baseline results")
    print("-" * 100)
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
        "m=96,k=40": {"nrmse_mean": 0.0037, "nrmse_se": 0.0006},
        "m=96,k=55": {"nrmse_mean": 0.1087, "nrmse_se": 0.0038},
    }

    # AMP reference from your completed AMP baseline.
    amp_ref = {
        "m=96,k=40": {"nrmse_mean": 0.2468, "nrmse_se": 0.0028},
        "m=96,k=55": {"nrmse_mean": 0.5368, "nrmse_se": 0.0024},
    }

    plot_methods = [
        "cosamp",
        "amp_topk_refit",
        "block_lista_raw",
        "block_lista_block_refit",
        "adaptive_refinement",
        "oracle",
    ]

    labels = {
        "cosamp": "CoSaMP",
        "amp_topk_refit": "AMP top-k",
        "block_lista_raw": "Block-LISTA raw",
        "block_lista_block_refit": "Block-LISTA block refit",
        "adaptive_refinement": "adaptive",
        "oracle": "oracle",
    }

    xloc = np.arange(len(settings))
    width = 0.13

    fig, ax = plt.subplots(figsize=(13, 5))

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
            else:
                means.append(results[key][method]["nrmse_mean"])
                ses.append(results[key][method]["nrmse_se"])

        ax.bar(
            xloc + (i - 2.5) * width,
            means,
            width,
            yerr=ses,
            capsize=4,
            label=labels[method],
        )

    ax.set_xticks(xloc)
    ax.set_xticklabels([f"m={m}, k={k}" for m, k in settings])
    ax.set_ylabel("NRMSE")
    ax.set_title("Block-LISTA baseline vs adaptive refinement")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_fig = FIGURES_DIR / "block_lista_baseline_10seed_nrmse.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")

    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
