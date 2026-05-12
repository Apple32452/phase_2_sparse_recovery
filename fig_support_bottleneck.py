"""
Figure: Support IoU vs Gaussian NRMSE (Fourier -> Gaussian zero-shot transfer).

Loads the same JSON used for Table 1 of the Asilomar abstract and produces a
single-column scatter that visualizes the central diagnosis: methods that
share a support-recovery score share a recovery NRMSE, and the only path to
the oracle floor is through better support identification.

Output: fig_support_bottleneck.pdf  (and .png for inspection)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "phase_1" / "results" / "ista_comparison_T30_lam0.05.json"

with SRC.open() as f:
    data = json.load(f)

methods = data["methods"]
naive = methods["naive_topk_ls"]
det = methods["det_ls"]
lista = methods["lista_zero_shot"]
oracle = methods["oracle_ls"]
raw_ista = methods["raw_ista"]

points = [
    ("Oracle support + LS", 1.0, oracle["gaussian_nrmse"], "#1b9e77", "o"),
    ("Naive top-k + LS", naive["gaussian_iou"], naive["gaussian_nrmse"], "#d95f02", "s"),
    ("Learned detector + LS", det["gaussian_iou"], det["gaussian_nrmse"], "#7570b3", "^"),
]

fig, ax = plt.subplots(figsize=(3.4, 2.6))

for label, iou, nrmse, color, marker in points:
    ax.scatter(iou, nrmse, s=70, c=color, marker=marker, edgecolor="black",
               linewidth=0.6, label=label, zorder=3)

# LISTA: no logged IoU. Plot as a horizontal reference and label.
# To replace with a full point, compute IoU from LISTA's top-k support on the
# Gaussian test set using the saved checkpoint sat_e2e_source_T30_lam0.05.pt.
lista_nrmse = lista["gaussian_nrmse"]
ax.axhline(lista_nrmse, color="#666666", linestyle="--", linewidth=1.0, zorder=1)
ax.text(0.02, lista_nrmse + 0.012, f"LISTA zero-shot ({lista_nrmse:.3f})",
        fontsize=7.5, color="#444444")

# Raw ISTA reference (no support selection at all)
raw_nrmse = raw_ista["gaussian_nrmse"]
ax.axhline(raw_nrmse, color="#bbbbbb", linestyle=":", linewidth=1.0, zorder=1)
ax.text(0.02, raw_nrmse + 0.012, f"Raw ISTA ({raw_nrmse:.3f})",
        fontsize=7.5, color="#666666")

# Annotate the overlapping naive / detector cluster
ax.annotate(
    "naive ≈ detector",
    xy=(naive["gaussian_iou"], naive["gaussian_nrmse"]),
    xytext=(0.50, 0.18),
    fontsize=7.5,
    arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
)

ax.set_xlim(0.0, 1.05)
ax.set_ylim(-0.02, 0.55)
ax.set_xlabel("Support IoU on Gaussian test")
ax.set_ylabel("Gaussian NRMSE")
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.legend(loc="upper right", fontsize=7, framealpha=0.95)

fig.tight_layout(pad=0.3)
out_pdf = Path(__file__).resolve().parent / "fig_support_bottleneck.pdf"
out_png = Path(__file__).resolve().parent / "fig_support_bottleneck.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=200)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
