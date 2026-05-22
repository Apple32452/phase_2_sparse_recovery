"""
Plot the per-head, per-layer Gram-bias scaling alpha during GAST training,
side-by-side for the strict-sparse k=55 and compressible k=25 regimes.
Both regimes show alpha learning to non-trivial magnitudes; only the
compressible regime translates that into a test-time gain.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

runs = [
    ("strict55, k=55",       HERE / "gast_quick_strict55_F1_alpha.json"),
    ("compressible25, k=25", HERE / "gast_quick_compressible25_F2_compressible.json"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
cmap_l0 = plt.cm.Blues(np.linspace(0.45, 0.95, 4))
cmap_l1 = plt.cm.Oranges(np.linspace(0.45, 0.95, 4))

for ax, (title, path) in zip(axes, runs):
    with path.open() as f:
        data = json.load(f)
    hist = data["alpha_history"]["gast"]
    epochs = [h["epoch"] for h in hist]
    alpha = np.array([h["alpha"] for h in hist])     # (E, n_layers, n_heads)
    n_layers = alpha.shape[1]
    n_heads  = alpha.shape[2]
    for h in range(n_heads):
        ax.plot(epochs, alpha[:, 0, h], color=cmap_l0[h], lw=1.4,
                label=f"L0 h{h}" if ax is axes[0] else None)
        ax.plot(epochs, alpha[:, 1, h], color=cmap_l1[h], lw=1.4, ls="--",
                label=f"L1 h{h}" if ax is axes[0] else None)
    ax.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch", fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylim(-3.0, 3.0)

axes[0].set_ylabel(r"Gram-bias scale $\alpha_h$", fontsize=9)
axes[0].legend(fontsize=6.5, loc="lower left", ncol=2,
               handlelength=1.4, columnspacing=0.6, handletextpad=0.4)

fig.tight_layout(pad=0.5)
out = HERE / "alpha_trajectories.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Wrote {out}")
