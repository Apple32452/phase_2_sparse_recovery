# Sparse Recovery Above the Phase Transition: A Learned Operator-Aware Approach

This repository contains Phase 2 experiments for studying **learned sparse recovery above the classical phase transition**. The project compares classical compressed-sensing algorithms such as OMP and CoSaMP against learned operator-aware support detectors under strict-sparse, noisy, and compressible-signal regimes.

The central question is:

> When do learned sparse-recovery methods have a real advantage over classical algorithms?

The main conclusion is that classical algorithms dominate below the phase transition, but learned operator-aware support detection has meaningful headroom in harder regimes, especially when signals are over-sparse or compressible.

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

The sparsity level `k` is varied to test both easy and difficult recovery regimes.

The key idea is that when oracle-support least squares succeeds but CoSaMP or OMP fails, the recovery problem is still possible in principle. The failure is mainly in **support identification**, which motivates using learned support detectors.

---

## Main Findings

### 1. Below the transition, classical methods win

At `k = 25`, with `n = 256` and `m = 128`, CoSaMP achieves nearly exact recovery on both partial Fourier and Gaussian operators. In this regime, learned recovery does not provide a meaningful advantage because classical recovery is already very strong.

### 2. Above the transition, learning has headroom

At `k = 55`, which is near or above the Donoho–Tanner weak threshold, greedy polynomial-time methods such as OMP and CoSaMP begin to fail. However, oracle-support least squares still recovers accurately.

This suggests that the difficult part is not amplitude recovery. The difficult part is choosing the correct support.

### 3. Operator-aware support detection improves over simple baselines

The learned detector uses coordinate-wise features that depend on both the signal estimate and the sensing operator. These include:

- ISTA estimate magnitude,
- residual correlation,
- direct measurement correlation,
- Gram diagonal information,
- local coherence.

A coordinate-wise MLP predicts support probabilities for each coordinate. The top-scoring indices are selected as the support, and amplitudes are recovered using least squares restricted to that support.

### 4. Compressible signals show the strongest learned advantage

For Gaussian compressible signals, the learned detector trained on mixed tail amplitudes beats CoSaMP at larger off-support tail amplitudes. In the project draft, the learned detector outperforms CoSaMP by about **8–10% relative NRMSE** at tail amplitudes `0.3–0.4`.

This is the first concrete regime in the project where the learned method consistently beats CoSaMP.

### 5. Attention is explored as a set-level mechanism

The project also explores a Gram-Aware Support Transformer, or **GAST**, which uses attention over coordinate tokens and adds Gram-matrix information into the attention logits.

The motivation is that CoSaMP performs set-level reasoning through its merge-refit-prune loop, while a coordinate-wise MLP scores each index independently.

Preliminary results suggest that attention modestly improves the compressible regime but does not fully solve the strict-sparse `k = 55` regime.

---

## Repository Structure

```text
phase_2/
├── cosamp_stress_test.py
├── cosamp_stress_test.png
├── cosamp_stress_test.json
├── diagnostic_structured.py
├── diagnostic_structured.png
├── diagnostic_structured.json
├── cosamp_vs_table1.py
├── cosamp_vs_table1.json
├── exp1_family_transfer.py
├── exp1b_cross_m_transfer.py
├── exp2_attention_block.py
├── exp2_cross_family.py
├── exp3_gaussian_family.py
├── exp3a_attention_option_a.py
├── alpha_trajectories.png
├── fig_pipeline_preview-1.png
├── asilomar_abstract.tex
├── asilomar_abstract.pdf
├── asilomar_abstract_v2.tex
├── asilomar_abstract_v2.pdf
└── README.md
```

---

## Main Files

### `cosamp_stress_test.py`

Runs the main CoSaMP stress test across three perturbation axes:

1. **Sparsity sweep**
   - Varies `k`
   - Tests where OMP and CoSaMP begin to fail

2. **Noise sweep**
   - Varies SNR
   - Tests robustness under measurement noise

3. **Approximate-sparsity sweep**
   - Varies off-support tail amplitude
   - Tests behavior when signals are compressible instead of exactly sparse

This script compares:

- naive top-k correlation + least squares,
- OMP,
- CoSaMP,
- oracle support + least squares.

It writes:

```text
cosamp_stress_test.png
cosamp_stress_test.json
```

### `diagnostic_structured.py`

Runs diagnostic experiments for structured and Gaussian sensing operators. This is useful for checking whether the sensing matrices, signal generation process, and recovery baselines behave correctly.

### `cosamp_vs_table1.py`

Generates table-style comparisons for the strict-sparse regime, especially around the difficult `k = 55` setting.

### `exp1_family_transfer.py`

Tests transfer behavior across sensing-matrix families.

### `exp1b_cross_m_transfer.py`

Tests transfer across different numbers of measurements `m`.

### `exp2_attention_block.py`

Runs the attention-based support recovery experiment. This file is related to the Gram-Aware Support Transformer idea.

### `exp2_cross_family.py`

Tests cross-family generalization of learned support recovery.

### `exp3_gaussian_family.py`

Runs Gaussian-family experiments, especially for compressible-signal settings.

### `exp3a_attention_option_a.py`

Runs an additional attention-based experiment variant.

---

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the main dependencies:

```bash
pip install numpy scipy matplotlib pandas scikit-learn
```

Some learned-model experiments may also require PyTorch:

```bash
pip install torch
```

---

## How to Run

Run the main CoSaMP stress test:

```bash
python cosamp_stress_test.py
```

Run the diagnostic experiment:

```bash
python diagnostic_structured.py
```

Run the table comparison:

```bash
python cosamp_vs_table1.py
```

Run the learned recovery and transfer experiments:

```bash
python exp1_family_transfer.py
python exp1b_cross_m_transfer.py
python exp2_attention_block.py
python exp2_cross_family.py
python exp3_gaussian_family.py
python exp3a_attention_option_a.py
```

If using JupyterLab, a script can also be run inside a notebook cell:

```python
%run cosamp_stress_test.py
```

---

## Example Result

The main stress test identifies cells where CoSaMP fails while oracle recovery remains accurate:

```text
Cells where CoSaMP NRMSE > 0.10 AND oracle NRMSE < 0.05
```

These are the most important regimes because they show that the measurement system still contains enough information for accurate recovery, but the classical algorithm fails to identify the correct support.

Example difficult regimes include:

```text
Fourier, k=40
Fourier, k=55
Fourier, k=70
Gaussian, k=55
Gaussian, k=70
```

In these settings, oracle least-squares recovery remains nearly exact, but CoSaMP has high NRMSE. This supports the motivation for learned support detection.

---

## Method Overview

The learned support detector follows a support-then-amplitude strategy:

1. Build coordinate-wise features for each index.
2. Predict a support probability for each coordinate.
3. Select the top-scoring coordinates.
4. Reconstruct amplitudes using least squares on the predicted support.

The coordinate-wise feature vector includes information from the measurement operator and the residual, allowing the model to adapt to the deployed sensing matrix.

---

## Why This Project Matters

Classical compressed-sensing methods are powerful when their assumptions hold. However, real signals are often:

- not exactly sparse,
- affected by noise,
- structured by unknown priors,
- observed through fixed deployment operators,
- not accompanied by known true sparsity `k`.

This project investigates whether learned support recovery can exploit operator structure and signal-prior information to improve recovery in those harder regimes.

---

## Current Conclusion

This project does **not** claim that learning always beats classical sparse recovery. Instead, it makes a more precise claim:

> Below the phase transition, classical methods dominate. Above the transition and under compressibility, learned operator-aware recovery has genuine headroom.

The strongest empirical win so far occurs for Gaussian compressible signals, where the learned detector consistently outperforms CoSaMP at high tail amplitudes.

---

## Future Work

Planned extensions include:

- iterative learned recovery inspired by CoSaMP’s merge-refit-prune structure,
- learned adaptive cardinality instead of fixed top-k selection,
- block-sparse and cluster-sparse signal priors,
- stronger set-level attention mechanisms,
- improved operator-aware features,
- broader testing across sensing operators and signal distributions.

---

## Paper Draft

This repository includes an Asilomar-style project draft:

```text
asilomar_abstract_v2.tex
asilomar_abstract_v2.pdf
```

The draft summarizes the motivation, experimental setup, classical baselines, learned detector, compressibility results, and preliminary attention-based extension.

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
