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

