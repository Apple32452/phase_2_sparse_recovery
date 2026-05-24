# Sparse Recovery Above the Phase Transition

This repository contains Phase 2 experiments for studying **when learned sparse recovery has a real advantage over classical compressed-sensing algorithms**.

The project compares classical sparse-recovery methods such as **OMP**, **CoSaMP**, and **HTP** against learned operator-aware support detectors under strict-sparse, noisy, and compressible-signal regimes.

The central question is:

> When do learned sparse-recovery methods have a real advantage over classical algorithms?

Current conclusion:

- Below the phase transition, classical methods such as CoSaMP are very strong.
- Near or above the transition, support identification becomes difficult, creating headroom for learning.
- The strongest learned advantage appears in **compressible-signal regimes**, where exact-sparsity assumptions become mismatched.

---

## Project Summary

We study the sparse linear inverse problem:

```text
y = Ax + ε
```

where:

- `A` is a sensing matrix,
- `x` is a sparse or compressible signal,
- `y` is the measurement vector,
- `ε` is measurement noise.

The main experimental setting is:

```text
n = 256
m = 128
```

The sparsity level `k`, measurement noise, sensing operator, and compressibility level are varied to test when classical and learned recovery methods succeed or fail.

---

## Main Findings

### 1. Below the transition, classical methods win

At `k = 25`, with `n = 256` and `m = 128`, CoSaMP achieves near-exact recovery on both partial Fourier and Gaussian operators.

This means easy strict-sparse recovery is not a good regime for claiming a learned advantage, because classical algorithms already solve the problem nearly perfectly.

### 2. Near the transition, learning has headroom but not full dominance

At larger sparsity levels such as `k = 55`, support identification becomes harder.

Oracle-support least squares can still recover accurately, which means the inverse problem is still well-posed if the correct support is known. However, greedy methods such as OMP and CoSaMP can begin to fail.

A coordinate-wise learned support detector improves over simple correlation baselines and OMP, but it does not consistently dominate CoSaMP in the strict-sparse setting.

Therefore, the correct claim is:

> Learning has headroom when support identification becomes the bottleneck, but it does not automatically beat classical sparse recovery in every regime.

### 3. Compressible signals show the strongest learned advantage

For Gaussian compressible signals, the learned detector trained on mixed tail amplitudes beats CoSaMP at larger off-support tail amplitudes.

In the current project draft, the learned detector improves over CoSaMP by approximately **8–10% relative NRMSE** at tail amplitudes `0.3–0.4`, consistently across multiple operator seeds and model initializations.

This is the strongest current empirical evidence that learned operator-aware support detection can help when exact-sparsity assumptions break.

### 4. Attention is explored as a set-level mechanism

The project also explores a **Gram-Aware Support Transformer**, or **GAST**, which uses attention over coordinate tokens and injects Gram-matrix information into the attention logits.

The motivation is that CoSaMP performs set-level reasoning through its merge-refit-prune loop, while a coordinate-wise MLP scores each coordinate independently.

Preliminary results suggest that attention modestly improves the compressible regime, but a single attention block does not fully solve the strict-sparse `k = 55` regime.

---

## Repository Structure

```text
phase_2_sparse_recovery/
├── README.md
├── requirements.txt
├── .gitignore
│
├── experiments/
│   ├── cosamp_stress_test.py
│   ├── cosamp_vs_table1.py
│   ├── diagnostic_structured.py
│   ├── learned_above_pt.py
│   ├── learned_compressible.py
│   ├── secure_compressible.py
│   ├── gast_quick.py
│   ├── plot_gast_alphas.py
│   ├── exp1_family_transfer.py
│   ├── exp1b_cross_m_transfer.py
│   ├── exp2_attention_block.py
│   ├── exp2_cross_family.py
│   ├── exp3_gaussian_family.py
│   ├── exp3a_attention_option_a.py
│   └── fig_support_bottleneck.py
│
├── results/
│   ├── cosamp/
│   ├── diagnostics/
│   ├── learned_above_pt/
│   ├── learned_compressible/
│   ├── secure_compressible/
│   ├── gast/
│   ├── exp1/
│   ├── exp1b/
│   ├── exp2_cf/
│   ├── exp3/
│   └── exp3a/
│
├── figures/
│   ├── cosamp/
│   ├── diagnostics/
│   ├── learned_above_pt/
│   ├── learned_compressible/
│   ├── gast/
│   └── paper/
│
├── paper/
│   ├── asilomar_abstract.tex
│   ├── asilomar_abstract.pdf
│   ├── asilomar_abstract_v2.tex
│   ├── asilomar_abstract_v2.pdf
│   ├── research_strategy_summary.tex
│   └── research_strategy_summary.pdf
│
├── notebooks/
│   └── exploration.ipynb
│
├── scripts/
│   └── reproduce_cosamp.sh
│
├── src/
│   └── sparse_recovery/
│       └── __init__.py
│
└── tests/
```

---

## Main Experiments

### CoSaMP stress test

File:

```text
experiments/cosamp_stress_test.py
```

This script runs the main diagnostic stress test across three perturbation axes:

1. **Sparsity sweep**
   - Varies `k`
   - Tests where OMP and CoSaMP begin to fail.

2. **Noise sweep**
   - Varies SNR.
   - Tests robustness under measurement noise.

3. **Compressible-signal sweep**
   - Varies off-support tail amplitude.
   - Tests behavior when signals are compressible rather than exactly sparse.

The script compares:

- naive top-k correlation + least squares,
- OMP,
- CoSaMP,
- oracle support + least squares.

Outputs:

```text
results/cosamp/cosamp_stress_test.json
figures/cosamp/cosamp_stress_test.png
```

Important note: for compressible signals, the oracle/target support is defined as:

```text
TopK(|x|)
```

That is, the best `k`-term support of the full compressible signal, not necessarily the original planted spike support.

---

### CoSaMP vs Table 1 comparison

File:

```text
experiments/cosamp_vs_table1.py
```

This script checks the strict-sparse `k = 25` setting and compares CoSaMP/HTP against Table-1-style baselines when the old `phase_1` result JSON is available.

If the old `phase_1` JSON is missing, the script still runs and saves fresh CoSaMP/HTP results.

Output:

```text
results/cosamp/cosamp_vs_table1.json
```

Current interpretation:

```text
n = 256, m = 128, k = 25
```

CoSaMP achieves near-oracle recovery on both Fourier and Gaussian operators. This confirms that below-transition strict-sparse recovery is not the right setting for claiming a learned advantage.

---

### Diagnostic structured experiment

File:

```text
experiments/diagnostic_structured.py
```

This experiment checks whether the sensing operators, signal generation process, and recovery baselines behave as expected.

Outputs:

```text
results/diagnostics/diagnostic_structured.json
figures/diagnostics/diagnostic_structured.png
```

---

### Learned detector above the phase transition

File:

```text
experiments/learned_above_pt.py
```

This experiment trains a coordinate-wise MLP support detector at the strict-sparse `k = 55` setting.

The detector uses operator-aware coordinate features:

- ISTA estimate magnitude,
- residual correlation,
- direct measurement correlation,
- Gram diagonal,
- local coherence.

The model predicts support probabilities for each coordinate. The top-scoring coordinates are selected as the support, and amplitudes are recovered using least squares restricted to that support.

---

### Learned detector under compressibility

File:

```text
experiments/learned_compressible.py
```

This experiment evaluates the learned support detector on compressible Gaussian signals.

Signals contain:

- `k` large coefficients,
- Gaussian off-support tail entries.

The target support is the top-`k` entries of the full signal magnitude:

```text
TopK(|x|)
```

This setting is where the learned detector currently shows its strongest advantage over CoSaMP.

---

### Robust compressible-signal validation

File:

```text
experiments/secure_compressible.py
```

This script checks whether the compressible-signal result is robust across multiple seeds.

It runs:

```text
5 operator seeds × 3 initialization seeds = 15 runs
```

and computes paired bootstrap confidence intervals for the CoSaMP-vs-learned NRMSE gap.

Output:

```text
results/secure_compressible/secure_compressible_summary.json
```

---

### Gram-Aware Support Transformer

File:

```text
experiments/gast_quick.py
```

This experiment compares:

- coordinate-wise MLP support detector,
- Gram-Aware Support Transformer, or GAST.

GAST treats coordinates as tokens and adds Gram-matrix bias into the attention logits. It tests whether set-level attention helps beyond independent coordinate scoring.

---

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the core packages manually:

```bash
pip install numpy scipy matplotlib pandas scikit-learn torch tqdm pytest
```

---

## How to Run

### Run the main CoSaMP stress test

```bash
bash scripts/reproduce_cosamp.sh
```

This regenerates:

```text
results/cosamp/cosamp_stress_test.json
figures/cosamp/cosamp_stress_test.png
```

You can also run it manually:

```bash
python experiments/cosamp_stress_test.py \
  --out-dir results/cosamp \
  --out-prefix cosamp_stress_test

mkdir -p figures/cosamp
mv -f results/cosamp/cosamp_stress_test.png figures/cosamp/
```

---

### Run the CoSaMP vs Table 1 comparison

```bash
python experiments/cosamp_vs_table1.py
```

Output:

```text
results/cosamp/cosamp_vs_table1.json
```

---

### Run diagnostic structured experiment

```bash
python experiments/diagnostic_structured.py

mkdir -p results/diagnostics figures/diagnostics
mv -f experiments/diagnostic_structured.json results/diagnostics/
mv -f experiments/diagnostic_structured.png figures/diagnostics/
```

---

### Run learned strict-sparse experiment

```bash
python experiments/learned_above_pt.py

mkdir -p results/learned_above_pt figures/learned_above_pt
mv -f experiments/learned_above_pt.json results/learned_above_pt/
mv -f experiments/learned_above_pt.png figures/learned_above_pt/
```

---

### Run learned compressible experiment

```bash
python experiments/learned_compressible.py

mkdir -p results/learned_compressible figures/learned_compressible
mv -f experiments/learned_compressible*.json results/learned_compressible/
mv -f experiments/learned_compressible*.png figures/learned_compressible/
```

---

### Run robust compressible validation

```bash
python experiments/secure_compressible.py

mkdir -p results/secure_compressible results/learned_compressible figures/learned_compressible
mv -f experiments/secure_compressible_summary.json results/secure_compressible/ 2>/dev/null || true
mv -f experiments/learned_compressible*.json results/learned_compressible/ 2>/dev/null || true
mv -f experiments/learned_compressible*.png figures/learned_compressible/ 2>/dev/null || true
```

---

### Run GAST experiment

```bash
python experiments/gast_quick.py

mkdir -p results/gast
mv -f experiments/gast_quick*.json results/gast/
```

---

## Current CoSaMP Stress-Test Interpretation

After fixing the CoSaMP stopping condition, the easy regimes behave as expected:

```text
k = 15 or k = 25:
CoSaMP ≈ oracle
```

Harder regimes show support-identification failure:

```text
Fourier:  k = 40, 55, 70
Gaussian: k = 55, 70
```

These are important because oracle support least squares remains accurate while CoSaMP degrades, suggesting that support identification rather than amplitude estimation is the main bottleneck.

---

## Known Issue

`experiments/fig_support_bottleneck.py` currently depends on an old missing file:

```text
phase_1/results/ista_comparison_T30_lam0.05.json
```

Because that file is not part of the cleaned repository, this script may fail unless the old `phase_1` results are restored or the script is rewritten to use the current result files.

This does not affect the main CoSaMP stress test or the cleaned repository workflow.

---

## Paper Draft

The paper draft is stored in:

```text
paper/
```

Important files include:

```text
paper/asilomar_abstract.tex
paper/asilomar_abstract.pdf
paper/asilomar_abstract_v2.tex
paper/asilomar_abstract_v2.pdf
paper/research_strategy_summary.tex
paper/research_strategy_summary.pdf
```

Common paper figures include:

```text
figures/cosamp/cosamp_stress_test.png
figures/gast/alpha_trajectories.png
figures/paper/fig_support_bottleneck.png
```

If compiling LaTeX from inside `paper/`, either copy required figures into `paper/figures/` or update the figure paths accordingly.

---

## Future Work

Planned directions include:

1. **Unknown cardinality**
   - Replace fixed top-k selection with learned thresholding.
   - Compare against CoSaMP with misspecified `k`.

2. **Structured priors**
   - Test block-sparse, cluster-sparse, and transform-domain compressible signals.
   - Measure whether learned recovery benefits from amortized prior information.

3. **Iterative learned recovery**
   - Build a learnable analogue of CoSaMP's merge-refit-prune loop.
   - Add operator-aware scoring or attention at each iteration.

4. **Broader phase diagrams**
   - Sweep over `m/n`, `k/m`, operator families, and signal distributions.
   - Identify where learned methods truly outperform classical baselines.

---

## Authors

- Di An, Johns Hopkins University
- Dylan Poppert, Johns Hopkins University
- Taewoon Choi, Johns Hopkins University
- Trac D. Tran, Johns Hopkins University

---

## Citation

If referencing this project, please cite the accompanying draft:

```text
Sparse Recovery Above the Phase Transition:
A Learned Operator-Aware Approach
```
