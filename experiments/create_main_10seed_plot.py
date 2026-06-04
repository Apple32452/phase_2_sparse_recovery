import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/adaptive_learned_block_refinement")
FIGURES_DIR = Path("figures/adaptive_learned_block_refinement")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_JSON = RESULTS_DIR / "adaptive_noise_robustness.json"
OUTPUT_JSON = RESULTS_DIR / "main_10seed_sigma0.json"
OUTPUT_FIG = FIGURES_DIR / "main_10seed_nrmse.png"

SETTINGS = [
    (96, 40),
    (96, 55),
]

METHODS = [
    "naive",
    "cosamp",
    "block_score_topk",
    "learned_block_scorer",
    "one_step_refinement",
    "adaptive_refinement",
    "oracle",
]

DISPLAY_NAMES = {
    "naive": "naive",
    "cosamp": "CoSaMP",
    "block_score_topk": "block score",
    "learned_block_scorer": "learned block",
    "one_step_refinement": "one-step",
    "adaptive_refinement": "adaptive",
    "oracle": "oracle",
}


def load_rows(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "aggregate" in data:
        return data["aggregate"]

    if isinstance(data, list):
        return data

    raise ValueError("Expected JSON to contain either a list or an 'aggregate' list.")


def row_matches(row, m, k, method):
    noise = row.get("noise_std", row.get("noise", row.get("sigma", None)))
    return (
        int(row.get("m")) == m
        and int(row.get("k")) == k
        and row.get("method") == method
        and abs(float(noise) - 0.0) < 1e-12
    )


def get_value(row, keys):
    for key in keys:
        if key in row:
            return float(row[key])
    raise KeyError(f"Missing keys {keys} in row: {row}")


def main():
    rows = load_rows(INPUT_JSON)

    summary = {}
    available_methods = []

    for method in METHODS:
        found_any = False
        for m, k in SETTINGS:
            matches = [r for r in rows if row_matches(r, m, k, method)]
            if matches:
                found_any = True
                break
        if found_any:
            available_methods.append(method)

    for m, k in SETTINGS:
        setting_key = f"m={m},k={k}"
        summary[setting_key] = {}

        for method in available_methods:
            matches = [r for r in rows if row_matches(r, m, k, method)]
            if not matches:
                continue

            row = matches[0]
            summary[setting_key][method] = {
                "nrmse_mean": get_value(row, ["nrmse_mean", "nrmse", "NRMSE"]),
                "nrmse_se": get_value(row, ["nrmse_se", "se", "NRMSE_SE"]),
                "iou_mean": get_value(row, ["iou_mean", "iou", "IoU"]),
            }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {OUTPUT_JSON}")
    print("\nMain 10-seed sigma=0 result")
    print("-" * 90)
    print(f"{'setting':12s} {'method':24s} {'NRMSE':>10s} {'SE':>10s} {'IoU':>10s}")

    for setting_key, table in summary.items():
        for method, vals in table.items():
            print(
                f"{setting_key:12s} "
                f"{method:24s} "
                f"{vals['nrmse_mean']:10.4f} "
                f"{vals['nrmse_se']:10.4f} "
                f"{vals['iou_mean']:10.4f}"
            )

    x = np.arange(len(SETTINGS))
    width = 0.11

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, method in enumerate(available_methods):
        means = []
        errors = []

        for m, k in SETTINGS:
            setting_key = f"m={m},k={k}"
            vals = summary[setting_key][method]
            means.append(vals["nrmse_mean"])
            errors.append(vals["nrmse_se"])

        offset = (i - (len(available_methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=DISPLAY_NAMES.get(method, method),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"m={m}, k={k}" for m, k in SETTINGS], fontsize=13)
    ax.set_ylabel("NRMSE", fontsize=14)
    ax.set_title("Main 10-seed result: adaptive learned block refinement", fontsize=17)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=11, ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
