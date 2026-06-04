````markdown
# phase_2_sparse_recovery

This repository contains experiments for learned and adaptive sparse recovery above the classical phase-transition regime.

The main research question is:

> When classical sparse recovery methods such as CoSaMP fail, can learned structure help recover the correct support?

The current answer is:

> One-shot learned support prediction is not enough, but learned structure combined with residual-based adaptive refinement gives large gains in hard block-sparse regimes.

---

## Main result

The strongest method in the current branch is:

```text
Residual-stop adaptive learned block refinement
````

At `n=256`, `m=96`, the final three-seed aggregate results are:

| Setting     |              Method | NRMSE mean | NRMSE SE |   IoU mean |
| ----------- | ------------------: | ---------: | -------: | ---------: |
| `m=96,k=40` |              CoSaMP |     0.6635 |   0.0053 |     0.4509 |
| `m=96,k=40` |    block_score_topk |     0.5787 |   0.0054 |     0.6110 |
| `m=96,k=40` | one_step_refinement |     0.0948 |   0.0099 |     0.9377 |
| `m=96,k=40` | adaptive_refinement | **0.0633** |   0.0039 | **0.9587** |
| `m=96,k=55` |              CoSaMP |     0.8330 |   0.0030 |     0.3587 |
| `m=96,k=55` |    block_score_topk |     0.7669 |   0.0007 |     0.5265 |
| `m=96,k=55` | one_step_refinement |     0.4380 |   0.0064 |     0.7636 |
| `m=96,k=55` | adaptive_refinement | **0.3844** |   0.0104 | **0.7970** |

The final aggregate figure is:

```text
figures/adaptive_learned_block_refinement/aggregate_residual_stop_nrmse.png
```

The final aggregate JSON is:

```text
results/adaptive_learned_block_refinement/aggregate_residual_stop.json
```

---

## Research story

The project developed through the following stages.

### 1. CoSaMP stress tests

We first tested classical sparse recovery across sparsity, noise, and approximate-sparsity regimes.

Main conclusion:

```text
Below the phase transition, classical methods win.
Above the phase transition, support identification becomes the bottleneck.
```

Relevant files:

```text
experiments/cosamp_vs_table1.py
experiments/diagnostic_structured.py
results/cosamp/
figures/cosamp/
results/diagnostics/
figures/diagnostics/
```

---

### 2. Small-n ceiling study

We then ran an exact-support search in small dimensions to test whether recovery failure is information-theoretic or algorithmic.

Main conclusion:

```text
Exact L0 and oracle support can still recover perfectly in small settings,
while greedy methods degrade. This suggests the failure is often algorithmic
support-search failure, not pure information loss.
```

Relevant files:

```text
experiments/ceiling_study_small_n.py
experiments/aggregate_ceiling_study.py
results/ceiling/
figures/ceiling/
```

---

### 3. Unknown-k recovery

Classical sparse recovery usually assumes the true sparsity level `k` is known.

Main conclusion:

```text
Fixed-k methods are fragile when the true sparsity varies.
This motivates learned or adaptive cardinality mechanisms.
```

Relevant files:

```text
experiments/unknown_k_recovery.py
results/unknown_k/
figures/unknown_k/
```

---

### 4. Structured-prior experiments

We tested whether one-shot structured heuristics can beat CoSaMP on block-sparse, cluster-sparse, and Markov-sparse signals.

Main conclusion:

```text
Structured priors help when the assumed structure matches the signal,
but one-shot structured support prediction is not reliable enough.
```

Relevant files:

```text
experiments/structured_priors.py
experiments/aggregate_structured_prior.py
results/structured_priors/
figures/structured_priors/
```

---

### 5. Learned block scorer

We trained a learned block scorer and compared it against CoSaMP and block-score top-k.

Main conclusion:

```text
Learning block scores helps, but one-shot learned support prediction
does not consistently beat structured heuristics or iterative refinement.
```

Relevant files:

```text
experiments/learned_block_scorer.py
experiments/aggregate_learned_block_scorer.py
results/learned_block_scorer/
figures/learned_block_scorer/
```

---

### 6. Iterative learned block refinement

We then inserted learned block information into a residual-refinement loop.

Main conclusion:

```text
Iterative refinement is much stronger than one-shot support prediction,
but fixed iteration counts can over-refine and hurt performance.
```

Relevant files:

```text
experiments/iterative_learned_block_refinement.py
results/iterative_learned_block_refinement/
figures/iterative_learned_block_refinement/
```

---

### 7. Adaptive learned block refinement

The final method uses residual-based adaptive stopping.

Main conclusion:

```text
Adaptive refinement keeps useful correction steps and rejects harmful ones.
This gives the strongest practical recovery in the main hard regimes.
```

Relevant files:

```text
experiments/adaptive_learned_block_refinement.py
experiments/adaptive_phase_diagram.py
experiments/adaptive_stopping_ablation.py
experiments/aggregate_residual_stop.py
results/adaptive_learned_block_refinement/
figures/adaptive_learned_block_refinement/
```

---

## Reproducing the final aggregate result

Run:

```bash
python experiments/aggregate_residual_stop.py
```

Expected output includes:

```text
Aggregate residual-stop results
-----------------------------------------------------------------------------------------------
setting      method                         NRMSE mean   NRMSE SE   IoU mean
m96_k40      naive                              0.8289     0.0009     0.2886
m96_k40      cosamp                             0.6635     0.0053     0.4509
m96_k40      block_score_topk                   0.5787     0.0054     0.6110
m96_k40      learned_block_scorer               0.5829     0.0073     0.6069
m96_k40      one_step_refinement                0.0948     0.0099     0.9377
m96_k40      fixed_iterative_refinement         0.2215     0.0066     0.8680
m96_k40      adaptive_refinement                0.0633     0.0039     0.9587
m96_k40      oracle                             0.0000     0.0000     1.0000
m96_k55      naive                              0.9257     0.0029     0.2839
m96_k55      cosamp                             0.8330     0.0030     0.3587
m96_k55      block_score_topk                   0.7669     0.0007     0.5265
m96_k55      learned_block_scorer               0.7776     0.0047     0.5182
m96_k55      one_step_refinement                0.4380     0.0064     0.7636
m96_k55      fixed_iterative_refinement         0.4563     0.0082     0.7597
m96_k55      adaptive_refinement                0.3844     0.0104     0.7970
m96_k55      oracle                             0.0000     0.0000     1.0000
```

This writes:

```text
results/adaptive_learned_block_refinement/aggregate_residual_stop.json
figures/adaptive_learned_block_refinement/aggregate_residual_stop_nrmse.png
```

---

## Recommended branch workflow

Current development branch:

```text
cleanup/repo-structure
```

Update the README on this branch first:

```bash
git checkout cleanup/repo-structure
git pull --rebase origin cleanup/repo-structure

git add README.md
git commit -m "Update README with adaptive refinement results"
git push origin cleanup/repo-structure
```

After the report and README are stable, merge into `main`:

```bash
git checkout main
git pull origin main
git merge cleanup/repo-structure
git push origin main
```

Alternative recommended workflow:

```text
Open a pull request from cleanup/repo-structure into main.
```

Do not manually edit both branches separately. Keep `cleanup/repo-structure` as the source of truth until the paper/report version is finalized.

---

## Report

The current report draft is written in an ICML-style Overleaf format.

Recommended Overleaf files:

```text
main.tex
figures/stress_test.png
figures/alpha_trajectories.png
figures/adaptive_learned_block_refinement/aggregate_residual_stop_nrmse.png
figures/adaptive_learned_block_refinement/adaptive_phase_diagram_smoke_adaptive_nrmse.png
figures/adaptive_learned_block_refinement/adaptive_phase_diagram_smoke_gain_cosamp.png
figures/adaptive_learned_block_refinement/adaptive_phase_diagram_smoke_gain_block.png
figures/adaptive_learned_block_refinement/adaptive_phase_diagram_smoke_steps.png
figures/adaptive_learned_block_refinement/adaptive_stopping_ablation_gain_cosamp.png
figures/adaptive_learned_block_refinement/adaptive_stopping_ablation_nrmse_by_weight.png
```

---

## Current paper-level claim

The cleanest claim supported by the experiments is:

```text
Classical sparse recovery fails mainly through support-identification errors.
One-shot learned support prediction helps but is insufficient.
Residual-based adaptive learned block refinement gives large gains by combining
learned structured proposals with residual-driven acceptance.
```

---

## Status

The exploration stage is mostly complete.

Remaining work:

1. Polish the Overleaf report.
2. Make sure all figure filenames match the LaTeX.
3. Update README on `cleanup/repo-structure`.
4. Push and open a pull request into `main`.
5. Add more seeds only if needed for a formal conference submission.

```
```
