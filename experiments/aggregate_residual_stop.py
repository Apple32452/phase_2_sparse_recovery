import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RDIR = ROOT / "results" / "adaptive_learned_block_refinement"
FDIR = ROOT / "figures" / "adaptive_learned_block_refinement"
FDIR.mkdir(parents=True, exist_ok=True)

SETTINGS = [
    ("m96_k40", [
        "adaptive_learned_block_refinement_m96_k40_residual_stop.json",
        "adaptive_learned_block_refinement_m96_k40_residual_stop_seed1.json",
        "adaptive_learned_block_refinement_m96_k40_residual_stop_seed2.json",
    ]),
    ("m96_k55", [
        "adaptive_learned_block_refinement_m96_k55_residual_stop.json",
        "adaptive_learned_block_refinement_m96_k55_residual_stop_seed1.json",
        "adaptive_learned_block_refinement_m96_k55_residual_stop_seed2.json",
    ]),
]

METHODS = [
    "naive",
    "cosamp",
    "block_score_topk",
    "learned_block_scorer",
    "one_step_refinement",
    "fixed_iterative_refinement",
    "adaptive_refinement",
    "oracle",
]

ALIASES = {
    "CoSaMP": "cosamp",
    "COSAMP": "cosamp",
    "cosamp_true_k": "cosamp",
    "block_score": "block_score_topk",
    "block_score_topK": "block_score_topk",
    "learned_block": "learned_block_scorer",
    "learned": "learned_block_scorer",
    "one_step": "one_step_refinement",
    "one_step_refine": "one_step_refinement",
    "fixed_iterative": "fixed_iterative_refinement",
    "fixed_iterative_refine": "fixed_iterative_refinement",
    "adaptive": "adaptive_refinement",
    "adaptive_refine": "adaptive_refinement",
}


def norm_name(x):
    x = str(x).strip()
    return ALIASES.get(x, x)


def parse_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for key in ["mean", "value", "avg"]:
            if key in x:
                return parse_float(x[key])
    if isinstance(x, (list, tuple, np.ndarray)):
        vals = [parse_float(v) for v in x]
        vals = [v for v in vals if v is not None]
        if vals:
            return float(np.mean(vals))
    if isinstance(x, str):
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", x)
        if m:
            return float(m.group(0))
    return None


def get_metric(entry, metric):
    candidates = [
        metric,
        metric.lower(),
        metric.upper(),
        f"{metric}_mean",
        f"mean_{metric}",
        f"{metric.upper()}_mean",
        f"{metric} mean",
        f"{metric.upper()} mean",
    ]

    if isinstance(entry, dict):
        for key in candidates:
            if key in entry:
                val = parse_float(entry[key])
                if val is not None:
                    return val

        for nested_key in ["metrics", "summary", "stats"]:
            if nested_key in entry and isinstance(entry[nested_key], dict):
                val = get_metric(entry[nested_key], metric)
                if val is not None:
                    return val

    return None


def has_metric(entry):
    return get_metric(entry, "nrmse") is not None


def find_methods_anywhere(obj):
    """
    Recursively search the JSON object for method result entries.
    Handles:
      {"cosamp": {"nrmse": ...}}
      {"method": "cosamp", "nrmse": ...}
      [{"method": "cosamp", ...}, ...]
      [["cosamp", {"nrmse": ...}], ...]
    """
    found = {}

    def visit(x):
        if isinstance(x, dict):
            # Case 1: dictionary key is method name.
            for k, v in x.items():
                nk = norm_name(k)
                if nk in METHODS and isinstance(v, dict) and has_metric(v):
                    found[nk] = v

            # Case 2: this dictionary is one row with a method-name field.
            name_keys = [
                "method",
                "method_name",
                "name",
                "label",
                "Method",
                "algorithm",
                "algo",
            ]
            for key in name_keys:
                if key in x:
                    nk = norm_name(x[key])
                    if nk in METHODS and has_metric(x):
                        found[nk] = x

            # Recurse.
            for v in x.values():
                visit(v)

        elif isinstance(x, list):
            # Case 3: list row like ["cosamp", {"nrmse": ...}]
            if len(x) >= 2:
                nk = norm_name(x[0])
                if nk in METHODS and isinstance(x[1], dict) and has_metric(x[1]):
                    found[nk] = x[1]

            for v in x:
                visit(v)

    visit(obj)
    return found


def load_file(path):
    with path.open() as f:
        data = json.load(f)

    methods = find_methods_anywhere(data)

    if not methods:
        print("\nCould not find methods in:", path)
        print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else type(data))
        print("First 1200 chars of JSON:")
        print(json.dumps(data, indent=2)[:1200])
        raise RuntimeError("Parser found zero method entries. See debug output above.")

    return methods


aggregate = {}

for setting, files in SETTINGS:
    rows = []

    for fname in files:
        path = RDIR / fname
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        methods = load_file(path)
        rows.append(methods)

    if not rows:
        raise RuntimeError(f"No files loaded for {setting}")

    aggregate[setting] = {
        "n_seeds": len(rows),
        "methods": {},
    }

    for method in METHODS:
        nrmse_vals = []
        iou_vals = []

        for row in rows:
            if method not in row:
                print(f"Warning: missing {method} for {setting}. Available: {list(row.keys())}")
                continue

            nrmse = get_metric(row[method], "nrmse")
            iou = get_metric(row[method], "iou")

            if nrmse is not None:
                nrmse_vals.append(nrmse)
            if iou is not None:
                iou_vals.append(iou)

        if not nrmse_vals:
            print(f"Skipping {method} for {setting}: no NRMSE values found.")
            continue

        nrmse_vals = np.array(nrmse_vals, dtype=float)
        iou_vals = np.array(iou_vals, dtype=float) if iou_vals else np.array([np.nan])

        aggregate[setting]["methods"][method] = {
            "nrmse_mean": float(np.mean(nrmse_vals)),
            "nrmse_std": float(np.std(nrmse_vals)),
            "nrmse_se": float(np.std(nrmse_vals) / np.sqrt(len(nrmse_vals))),
            "iou_mean": float(np.nanmean(iou_vals)),
            "iou_std": float(np.nanstd(iou_vals)),
            "iou_se": float(np.nanstd(iou_vals) / np.sqrt(len(iou_vals))),
            "nrmse_values": nrmse_vals.tolist(),
            "iou_values": iou_vals.tolist(),
        }


out_json = RDIR / "aggregate_residual_stop.json"
with out_json.open("w") as f:
    json.dump(aggregate, f, indent=2)

print(f"\nWrote {out_json}")

print("\nAggregate residual-stop results")
print("-" * 95)
print(f"{'setting':<12} {'method':<28} {'NRMSE mean':>12} {'NRMSE SE':>10} {'IoU mean':>10}")

for setting, s in aggregate.items():
    for method in METHODS:
        if method not in s["methods"]:
            continue
        r = s["methods"][method]
        print(
            f"{setting:<12} {method:<28} "
            f"{r['nrmse_mean']:>12.4f} {r['nrmse_se']:>10.4f} {r['iou_mean']:>10.4f}"
        )


# Plot aggregate NRMSE.
labels = ["m=96,k=40", "m=96,k=55"]
setting_keys = ["m96_k40", "m96_k55"]

available_methods = [
    m for m in METHODS
    if all(m in aggregate[s]["methods"] for s in setting_keys)
]

if not available_methods:
    raise RuntimeError("No common methods found for plotting.")

x = np.arange(len(labels))
width = min(0.10, 0.80 / len(available_methods))

fig, ax = plt.subplots(figsize=(13, 5.5))

offset_center = (len(available_methods) - 1) / 2

for j, method in enumerate(available_methods):
    means = [
        aggregate[s]["methods"][method]["nrmse_mean"]
        for s in setting_keys
    ]
    ses = [
        aggregate[s]["methods"][method]["nrmse_se"]
        for s in setting_keys
    ]

    ax.bar(
        x + (j - offset_center) * width,
        means,
        width,
        yerr=ses,
        capsize=3,
        label=method,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("NRMSE")
ax.set_title("Residual-stop adaptive learned block refinement: aggregate across seeds")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()

out_png = FDIR / "aggregate_residual_stop_nrmse.png"
fig.savefig(out_png, dpi=180)
print(f"Wrote {out_png}")
