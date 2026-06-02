# Phase 2 Sparse Recovery

This repository contains diagnostic and learned sparse-recovery experiments for studying when learning helps compressed sensing / sparse recovery.

The main goal is **not** to claim that learning always beats classical sparse-recovery algorithms. Instead, the project studies:

1. when classical methods such as CoSaMP are already near oracle,
2. when hard regimes still contain algorithmic headroom,
3. when unknown sparsity (k) matters,
4. when structured priors help,
5. how learned priors can be inserted into an algorithmic recovery loop.

The current strongest result is **one-step learned block refinement**, which combines learned block probabilities with a residual-based merge-refit-prune correction step. In hard block-sparse regimes, this method substantially improves over CoSaMP, hand-designed block-score top-(k), and one-shot learned block scoring.

---

## Repository Structure

```text
phase_2_sparse_recovery/
├── experiments/
│   ├── cosamp_stress_test.py
│   ├── cosamp_vs_table1.py
│   ├── diagnostic_structured.py
│   ├── ceiling_study_small_n.py
│   ├── aggregate_ceiling_study.py
│   ├── unknown_k.py
│   ├── structured_priors.py
│   ├── aggregate_structured_priors.py
│   ├── learned_structured_prior.py
│   ├── learned_block_scorer.py
│   ├── aggregate_learned_block_scorer.py
│   ├── iterative_learned_block_refinement.py
│   └── aggregate_iterative_block_refinement.py
│
├── results/
│   ├── cosamp/
│   ├── diagnostics/
│   ├── ceiling/
│   ├── unknown_k/
│   ├── structured_priors/
│   ├── learned_structured_prior/
│   ├── learned_block_scorer/
│   └── iterative_learned_block_refinement/
│
├── figures/
│   ├── cosamp/
│   ├── diagnostics/
│   ├── ceiling/
│   ├── unknown_k/
│   ├── structured_priors/
│   ├── learned_structured_prior/
│   ├── learned_block_scorer/
│   └── iterative_learned_block_refinement/
│
├── scripts/
│   └── reproduce_cosamp.sh
│
└── README.md
```

---

## Installation

Create an environment with standard scientific Python packages.

```bash
conda create -n sparse-recovery python=3.11 -y
conda activate sparse-recovery

pip install numpy scipy matplotlib scikit-learn
```

If using an existing environment, make sure these packages are installed:

```bash
pip install numpy scipy matplotlib scikit-learn
```
The sparsity level `k`, measurement noise, sensing operator, and compressibility level are varied to test when classical and learned recovery methods succeed or fail.

---

## Main Experimental Story

The project currently supports the following refined research claim:

> Learning does not universally beat classical sparse recovery. Corrected CoSaMP is very strong in easy strict-sparse regimes. However, hard structured regimes expose support-identification bottlenecks. In those regimes, learned priors become most useful when inserted into a residual-based merge-refit-prune recovery loop.

The strongest current method is:

```text
one-step learned block refinement
```

This method:

1. predicts active blocks using a learned block scorer,
2. forms an initial block support,
3. performs least-squares refitting,
4. computes residual correlations,
5. merges learned-prior blocks with residual-correlation blocks,
6. refits on the merged candidate support,
7. prunes back to the target sparsity (k).

---

## Key Results

### 1. Corrected CoSaMP baseline

File:

```text
experiments/cosamp_stress_test.py
```

The CoSaMP implementation was corrected and rerun. After the fix, CoSaMP behaves as expected: it is near-oracle in easy strict-sparse regimes.

This is important because it prevents an artificial learned advantage caused by an implementation issue.

Run:

```bash
bash scripts/reproduce_cosamp.sh
```

Outputs:

```text
results/cosamp/cosamp_stress_test.json
figures/cosamp/cosamp_stress_test.png
```

Main takeaway:

```text
Easy strict-sparse regimes are not enough to claim a learned advantage.
Corrected CoSaMP is already very strong there.
```

---

### 2. CoSaMP vs Table 1 diagnostic

File:

```text
experiments/cosamp_vs_table1.py
```

This experiment confirms that below-transition strict-sparse recovery is not a good regime for claiming a learned advantage. CoSaMP achieves near-oracle recovery in the strict-sparse (k=25) setting.

Run:

```bash
python experiments/cosamp_vs_table1.py
```

Output:

```text
results/cosamp/cosamp_vs_table1.json
```

Main takeaway:

```text
If CoSaMP is already near oracle, learned methods have little meaningful room to improve.
```

---

### 3. Small-n ceiling study

Files:

```text
experiments/ceiling_study_small_n.py
experiments/aggregate_ceiling_study.py
```

This experiment tests whether hard sparse-recovery regimes are information-theoretically impossible or only algorithmically difficult. Since (n) is small, exact (L_0) search is feasible.

Methods compared:

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
Exact L0 remains near oracle while greedy methods degrade as sparsity increases.
```

This suggests that some hard regimes are not immediately information-theoretically impossible. They still contain algorithmic headroom.

Run:

```bash
python experiments/ceiling_study_small_n.py
python experiments/aggregate_ceiling_study.py
```

Outputs:

```text
results/ceiling/
figures/ceiling/
```

Main takeaway:

```text
The hard region contains recoverable information, but standard greedy methods fail to find the correct support.
```

---

### 4. Unknown-k recovery

File:

```text
experiments/unknown_k.py
```

This experiment tests recovery when the true sparsity (k) is unknown.

Methods compared:

```text
CoSaMP with true k
CoSaMP with fixed k
OMP with residual stopping
learned-k predictor + CoSaMP
```

Main finding:

```text
Learned-k CoSaMP improves over poor fixed-k and residual-stopping baselines,
but it does not beat oracle-k CoSaMP.
```

Run:

```bash
python experiments/unknown_k.py
```

Outputs:

```text
results/unknown_k/
figures/unknown_k/
```

Main takeaway:

```text
Cardinality estimation helps, but support selection remains the main bottleneck.
```

---

### 5. Structured-prior experiments

Files:

```text
experiments/structured_priors.py
experiments/aggregate_structured_priors.py
```

This experiment tests whether support structure creates regimes where prior-aware recovery can outperform generic sparse-recovery algorithms.

Signal families:

```text
iid_sparse
block_sparse
cluster_sparse
markov_sparse
```

Methods compared:

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
Structured priors matter most in harder regimes.
The clearest gain appears for block-sparse signals.
```

In hard regimes such as (m=96), `block_score_topk` can outperform CoSaMP on block-sparse signals.

Run examples:

```bash
python experiments/structured_priors.py \
  --m 96 \
  --k 40 \
  --n-test 200 \
  --out-prefix structured_priors_m96_k40

python experiments/structured_priors.py \
  --m 96 \
  --k 55 \
  --n-test 200 \
  --out-prefix structured_priors_m96_k55

python experiments/aggregate_structured_priors.py
```

Outputs:

```text
results/structured_priors/
figures/structured_priors/
```

Main takeaway:

```text
Structured priors create measurable algorithmic headroom, especially for block-sparse recovery.
```

---

### 6. Learned structured-prior detector

File:

```text
experiments/learned_structured_prior.py
```

This experiment trains a coordinate-level local/context detector using features from:

```text
|A^T y|
local correlation windows
block-level pooled features
coherence features
measurement norm features
```

Methods compared:

```text
naive top-k
CoSaMP
smoothed_topk
block_score_topk
learned_structured
oracle
```

Main finding:

```text
The learned coordinate-level detector improves over some naive heuristics,
but it does not consistently beat CoSaMP.
```

Run:

```bash
python experiments/learned_structured_prior.py \
  --m 96 \
  --k 40 \
  --n-train-per-family 400 \
  --n-test-per-family 200 \
  --out-prefix learned_structured_prior_m96_k40
```

Outputs:

```text
results/learned_structured_prior/
figures/learned_structured_prior/
```

Main takeaway:

```text
One-shot coordinate-wise learning is not enough.
A learned sparse-recovery method needs set-level or block-level structure.
```

---

### 7. Learned block scorer

Files:

```text
experiments/learned_block_scorer.py
experiments/aggregate_learned_block_scorer.py
```

This experiment focuses on block-sparse recovery, where structured-prior experiments showed the clearest headroom.

The learned block scorer predicts active blocks first, then selects coordinates inside the highest-probability blocks.

Methods compared:

```text
naive top-k
CoSaMP
block_score_topk
learned_block_scorer
oracle
```

Main finding:

```text
The learned block scorer improves over CoSaMP in hard block-sparse regimes,
but it remains slightly weaker than the hand-designed block_score_topk baseline.
```

Run:

```bash
python experiments/learned_block_scorer.py \
  --m 96 \
  --k 40 \
  --n-train 1000 \
  --n-test 300 \
  --out-prefix learned_block_scorer_m96_k40_fixed

python experiments/learned_block_scorer.py \
  --m 96 \
  --k 55 \
  --n-train 1000 \
  --n-test 300 \
  --out-prefix learned_block_scorer_m96_k55_fixed

python experiments/aggregate_learned_block_scorer.py
```

Outputs:

```text
results/learned_block_scorer/
figures/learned_block_scorer/
```

Main takeaway:

```text
Block-level learning helps, but one-shot learned block scoring is still limited.
```

---

### 8. One-step learned block refinement

Files:

```text
experiments/iterative_learned_block_refinement.py
experiments/aggregate_iterative_block_refinement.py
```

This is the strongest current result.

The method starts from learned block probabilities, then performs one residual-based refinement step:

```text
learned block prior
+ least-squares refit
+ residual correlation
+ merge candidate blocks
+ prune back to k
```

Methods compared:

```text
naive top-k
CoSaMP
block_score_topk
learned_block_scorer
iterative_refinement
oracle
```

Although the script name uses `iterative`, the current best method should be interpreted as:

```text
one-step learned block refinement
```

because the ablation shows that one refinement step is best.

Run main experiments:

```bash
python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 40 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k40_iter1

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 55 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k55_iter1
```

Run additional seeds:

```bash
python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 40 \
  --n-train 1000 \
  --n-test 300 \
  --seed 1 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k40_seed1_iter1

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 55 \
  --n-train 1000 \
  --n-test 300 \
  --seed 1 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k55_seed1_iter1

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 40 \
  --n-train 1000 \
  --n-test 300 \
  --seed 2 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k40_seed2_iter1

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 55 \
  --n-train 1000 \
  --n-test 300 \
  --seed 2 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k55_seed2_iter1
```

Run iteration ablation:

```bash
python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 40 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k40_iter1

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 40 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 2 \
  --out-prefix iterative_learned_block_refinement_m96_k40_iter2

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 40 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 4 \
  --out-prefix iterative_learned_block_refinement_m96_k40_iter4

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 55 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 1 \
  --out-prefix iterative_learned_block_refinement_m96_k55_iter1

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 55 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 2 \
  --out-prefix iterative_learned_block_refinement_m96_k55_iter2

python experiments/iterative_learned_block_refinement.py \
  --m 96 \
  --k 55 \
  --n-train 1000 \
  --n-test 300 \
  --refine-iters 4 \
  --out-prefix iterative_learned_block_refinement_m96_k55_iter4
```

Aggregate:

```bash
python experiments/aggregate_iterative_block_refinement.py
```

Outputs:

```text
results/iterative_learned_block_refinement/
figures/iterative_learned_block_refinement/
```

Important aggregate figures:

```text
figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_seed_summary.png
figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_gains.png
figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_iter_ablation.png
```

Main finding:

```text
One-step learned block refinement is the strongest non-oracle method.
It beats CoSaMP, block_score_topk, and learned_block_scorer in hard block-sparse regimes.
```

Approximate aggregate summary:

```text
m=96, k=40:
CoSaMP                 ≈ 0.66 NRMSE
block_score_topk       ≈ 0.57 NRMSE
learned_block_scorer   ≈ 0.60 NRMSE
one-step refinement    ≈ 0.11 NRMSE

m=96, k=55:
CoSaMP                 ≈ 0.83 NRMSE
block_score_topk       ≈ 0.76 NRMSE
learned_block_scorer   ≈ 0.77 NRMSE
one-step refinement    ≈ 0.43 NRMSE
```

Approximate gains:

```text
m=96, k=40:
one-step refinement vs CoSaMP       ≈ +0.550 NRMSE gain
one-step refinement vs block_score  ≈ +0.462 NRMSE gain
one-step refinement vs learned      ≈ +0.484 NRMSE gain

m=96, k=55:
one-step refinement vs CoSaMP       ≈ +0.397 NRMSE gain
one-step refinement vs block_score  ≈ +0.332 NRMSE gain
one-step refinement vs learned      ≈ +0.344 NRMSE gain
```

Iteration ablation:

```text
One refinement step is best.
Additional refinement steps can degrade performance, likely because residual updates drift away from the true block support.
```

Main takeaway:

```text
Learning is most effective when used as a structured prior inside a merge-refit-prune recovery loop, not as a standalone support classifier.
```

---

## Current Best Paper Claim

The most defensible paper claim is:

```text
Learning does not universally beat classical sparse recovery.
Corrected CoSaMP is already near-oracle in easy strict-sparse regimes.

However, hard structured regimes expose support-identification bottlenecks.
In those regimes, learned priors are most useful when inserted into an algorithmic recovery loop.

One-step learned block refinement combines learned block probabilities with residual-based correction and substantially improves over CoSaMP, block_score_topk, and one-shot learned block scoring in hard block-sparse regimes.
```

---

## Recommended Paper Figures

For the final report or paper, the most important figures are:

```text
1. CoSaMP stress test after correction
2. Small-n ceiling study aggregate
3. Unknown-k recovery
4. Structured-prior aggregate heatmap
5. Learned block scorer aggregate
6. One-step learned block refinement seed summary
7. One-step learned block refinement gains
8. Refinement iteration ablation
```

Most important current figures:

```text
figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_seed_summary.png
figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_gains.png
figures/iterative_learned_block_refinement/aggregate_iterative_block_refinement_iter_ablation.png
```

For paper-ready plotting, prefer:

```text
mean ± standard error
boxplots
or clipped lower error bars
```

instead of full mean ± standard deviation, because NRMSE is nonnegative and standard-deviation error bars may visually extend below zero.

---

## Reproducibility Notes

Most experiments write results to:

```text
results/<experiment_name>/
```

and figures to:

```text
figures/<experiment_name>/
```

The JSON files contain:

```text
configuration
method list
mean NRMSE
standard deviation
median
IoU
support-size diagnostics
```

Use the aggregate scripts to generate paper-ready summaries.

---

## Development Notes

After running experiments, commit results with:

```bash
git add experiments/
git add results/
git add figures/

git commit -m "Add new sparse recovery experiment results"
git push origin cleanup/repo-structure
```

If GitHub rejects the push because the remote branch has new commits:

```bash
git fetch origin
git pull --rebase origin cleanup/repo-structure
git push origin cleanup/repo-structure
```

---

## Next Steps

The current priority is no longer adding many unrelated experiments. The next steps should be:

```text
1. Clean paper-ready figures.
2. Update the Overleaf paper with the one-step learned block refinement method.
3. Add an algorithm box for one-step learned block refinement.
4. Add a table reporting aggregate gains over CoSaMP and block_score_topk.
5. Add the iteration ablation showing that one step is best.
6. Test a few more regimes or seeds only if needed.
7. Compare against stronger structured sparse-recovery baselines if time allows.
```

Possible method improvements:

```text
adaptive stopping rule for refinement
residual-drift detection
stronger learned block aggregator
extension to cluster-sparse and Markov-sparse supports
```

---

## Summary

This repository has evolved from a basic learned sparse-recovery comparison into a diagnostic study of **when learning helps sparse recovery**.

The current conclusion is:

```text
Easy strict-sparse regimes: classical methods are strong.
Hard unstructured regimes: support selection is difficult.
Unknown k: learned cardinality helps but is insufficient.
Structured priors: block structure creates recoverable headroom.
One-shot learned support scoring: useful but limited.
One-step learned block refinement: strongest current non-oracle method.
```

The strongest experimental result is:

```text
One-step learned block refinement substantially improves over CoSaMP, block_score_topk, and learned_block_scorer in hard block-sparse regimes.
```
