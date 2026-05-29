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

## Recent Diagnostic Experiments

This repository now includes several diagnostic experiments designed to make the paper more rigorous and to separate easy sparse-recovery regimes from regimes where learning or structured priors may provide real value.

---

### Small-n ceiling study

File:

```text
experiments/ceiling_study_small_n.py
```

This experiment tests whether hard sparse-recovery regimes are algorithmically difficult or information-limited. It uses small dimensions where exact (L_0) support search is computationally feasible.

The experiment compares:

```text
naive top-k correlation
OMP
CoSaMP
HTP
exact L0 search
oracle support + least squares
```

Main finding:

```text
Exact L0 remains near oracle while OMP, CoSaMP, HTP, and naive top-k degrade as sparsity increases.
```

This suggests that, in the tested small-(n) setting, the hard region is not immediately information-theoretically impossible. Instead, there remains algorithmic headroom. However, the ambiguity gap between the best and second-best supports shrinks rapidly as (k) increases, suggesting that the problem is approaching a support-identification ceiling.

Outputs:

```text
results/ceiling/
figures/ceiling/
```

Aggregate script:

```text
experiments/aggregate_ceiling_study.py
```

---

### Unknown-(k) experiment

File:

```text
experiments/unknown_k.py
```

This experiment tests sparse recovery when the true sparsity (k) is unknown.

The experiment compares:

```text
CoSaMP with true k
CoSaMP with fixed k
OMP with residual stopping
learned-k predictor + CoSaMP
```

Main finding:

```text
Learned-k CoSaMP improves over poor fixed-k and residual-stopping baselines, but it does not beat oracle-k CoSaMP.
```

This suggests that cardinality estimation helps, but the main bottleneck remains support selection.

Outputs:

```text
results/unknown_k/
figures/unknown_k/
```

---

### Structured-prior experiments

File:

```text
experiments/structured_priors.py
```

This experiment tests whether structured supports create regimes where prior-aware recovery can outperform generic sparse-recovery algorithms.

Signal families:

```text
iid_sparse
block_sparse
cluster_sparse
markov_sparse
```

Methods:

```text
naive top-k correlation
OMP
CoSaMP
HTP
smoothed_topk
block_score_topk
oracle support + least squares
```

Main finding:

```text
In easy regimes, CoSaMP is near oracle and structured priors are unnecessary.
In harder regimes, block_score_topk begins to outperform CoSaMP on block-sparse signals.
```

This suggests that structured priors create measurable algorithmic headroom, especially for block-sparse recovery.

Outputs:

```text
results/structured_priors/
figures/structured_priors/
```

Aggregate script:

```text
experiments/aggregate_structured_priors.py
```

---

### Learned structured-prior detector

File:

```text
experiments/learned_structured_prior.py
```

This experiment trains a family-specific local/context support detector using coordinate-level features.

Methods:

```text
naive top-k correlation
CoSaMP
smoothed_topk
block_score_topk
learned_structured
oracle support + least squares
```

Main finding:

```text
The learned local/context detector improves over some naive structured heuristics, but it does not consistently beat CoSaMP.
```

This suggests that coordinate-wise learning is not sufficient. The next model should use stronger block-level or sequence-level aggregation.

Outputs:

```text
results/learned_structured_prior/
figures/learned_structured_prior/
```

---

### Learned block scorer

File:

```text
experiments/learned_block_scorer.py
```

This experiment focuses on the strongest structured-prior signal: block-sparse recovery.

Instead of scoring coordinates independently, the learned block scorer predicts active blocks first, then selects coordinates inside the highest-probability blocks.

Methods:

```text
naive top-k correlation
CoSaMP
block_score_topk
learned_block_scorer
oracle support + least squares
```

Main finding:

```text
For n=256 and m=96, the learned block scorer improves over CoSaMP at both k=40 and k=55. However, the hand-designed block_score_topk baseline remains slightly stronger.
```

This confirms that block-level structure is useful, but also shows that the current learned block scorer is not yet strong enough to beat a specialized hand-designed block heuristic.

Fixed outputs:

```text
results/learned_block_scorer/learned_block_scorer_m96_k40_fixed.json
results/learned_block_scorer/learned_block_scorer_m96_k55_fixed.json
figures/learned_block_scorer/learned_block_scorer_m96_k40_fixed_nrmse.png
figures/learned_block_scorer/learned_block_scorer_m96_k55_fixed_nrmse.png
```

Aggregate script:

```text
experiments/aggregate_learned_block_scorer.py
```

Aggregate outputs:

```text
results/learned_block_scorer/aggregate_learned_block_scorer.json
figures/learned_block_scorer/aggregate_learned_block_scorer_nrmse.png
figures/learned_block_scorer/aggregate_learned_block_scorer_gains.png
```

---

## Current Research Interpretation

The current experimental evidence supports the following refined paper story:

```text
1. Easy strict-sparse regimes are not good settings for claiming a learned advantage because CoSaMP is often near oracle.

2. Small-n ceiling studies show that exact L0 can remain oracle-level even when greedy algorithms fail, suggesting algorithmic headroom.

3. Unknown-k experiments show that cardinality estimation helps, but support selection remains the main bottleneck.

4. Structured priors matter most in harder regimes, especially for block-sparse signals.

5. Learned block scoring improves over CoSaMP, but the hand-designed block_score_topk baseline remains slightly stronger.

6. The next step is to develop an iterative learned block-refinement method that combines block-level prior information with CoSaMP-style residual correction.
```

---

## Next Planned Experiment

The next experiment is:

```text
experiments/iterative_learned_block_refinement.py
```

Goal:

```text
Start from learned block scoring, then iteratively refine the support using residual correlations, block probabilities, least-squares refitting, and pruning.
```

This is motivated by the observation that:

```text
learned_block_scorer > CoSaMP
block_score_topk > learned_block_scorer
```

The next model should try to combine the strengths of both:

```text
block-level learned prior
+
iterative residual correction
+
least-squares refit/prune
```

