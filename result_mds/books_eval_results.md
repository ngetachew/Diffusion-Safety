# General Utility Eval — Books Dataset

Evaluates how much each patching method degrades performance on benign, general-purpose text. A method that suppresses toxic content without harming general utility should show a small Δ here.

## Setup

- **Model:** `GSAI-ML/LLaDA-8B-Base`
- **Eval dataset:** `P1ayer-1/books-3-textbooks`, full-text loss (no continuation masking)
- **Eval examples:** 5,437 (dataset exhausted before 50k limit)
- **Tokens evaluated:** 2,733,806
- **Metric:** Per-example cross-entropy loss (mean ± std, SE); rank of correct token in ff_out output distribution (1-based)

### PCA patch
- **Subspace source:** `allenai/real-toxicity-prompts`, 400 examples, masked hidden states
- **Patch:** `ff_out.weight = ff_out.weight @ (I - UU^T)`

### Mean-diff patch
- **Toxic hidden states:** `allenai/real-toxicity-prompts`, 7,000 examples → 103,423 vectors
- **Benign hidden states:** `openai/gsm8k`, 7,000 examples → 642,703 vectors
- **Diff norm:** 106.8582 | **Cosine similarity of means:** 0.7794
- **Patch:** forward pre-hook projecting `h → h - (h·u)u` on ff_out input

---

## Results

### PCA Patch

| k | Expl. Var | Orig Loss (mean ± std) | Orig SE | Patched Loss (mean ± std) | Patched SE | Δ Loss | Orig Mean Rank | Patched Mean Rank | Rank Δ | Orig Rank-1 | Patched Rank-1 | Δ Rank-1 |
|---|-----------|------------------------|---------|---------------------------|------------|--------|----------------|-------------------|--------|-------------|----------------|----------|
| 8 | 47.2% | 13.6966 ± 0.6968 | 0.0095 | 13.1462 ± 0.6516 | 0.0089 | −0.5504 | 1079.2 | 2455.6 | +1376.4 | 178,344 (6.52%) | 176,533 (6.46%) | −1,811 |
| 32 | 70.7% | 13.6966 ± 0.6968 | 0.0095 | 12.4078 ± 0.6298 | 0.0086 | −1.2888 | 1079.2 | 3434.4 | +2355.2 | 178,344 (6.52%) | 174,226 (6.37%) | −4,118 |
| 64 | 77.8% | 13.6966 ± 0.6968 | 0.0095 | 11.9282 ± 0.6163 | 0.0084 | −1.7684 | 1079.2 | 3592.3 | +2513.1 | 178,344 (6.52%) | 177,906 (6.51%) | −438 |
| 128 | 83.1% | 13.6966 ± 0.6968 | 0.0095 | 10.9946 ± 0.5976 | 0.0082 | −2.7020 | 1079.2 | 3501.1 | +2421.9 | 178,344 (6.52%) | 177,036 (6.48%) | −1,308 |

### Mean-Diff Patch

| Orig Loss (mean ± std) | Orig SE | Patched Loss (mean ± std) | Patched SE | Δ Loss | Orig Mean Rank | Patched Mean Rank | Rank Δ | Orig Rank-1 | Patched Rank-1 | Δ Rank-1 |
|------------------------|---------|---------------------------|------------|--------|----------------|-------------------|--------|-------------|----------------|----------|
| 13.6966 ± 0.6968 | 0.0095 | 13.6496 ± 0.6899 | 0.0094 | −0.0470 | 1079.2 | 1318.2 | +239.0 | 178,344 (6.52%) | 178,261 (6.52%) | −83 |

---

## Comparison: Toxic vs. Books Δ Loss

A well-targeted patch should show a large Δ on toxic content and near-zero Δ on benign content.

| Method | Δ Loss (Toxic) | Δ Loss (Books) | Specificity Ratio |
|--------|---------------|----------------|-------------------|
| Mean-diff | **+0.1002** | −0.0470 | — |
| PCA k=8 | −0.3956 | −0.5504 | — |
| PCA k=32 | −0.7172 | −1.2888 | — |
| PCA k=64 | −1.0176 | −1.7684 | — |
| PCA k=128 | −1.4816 | −2.7020 | — |

---

## Observations

**Mean-diff is highly targeted** — Δ loss on books is only −0.0470, nearly negligible compared to the +0.1002 on toxic content. Mean rank increases by only +239 on books vs. +592 on toxic. The patch is largely specific to the toxic direction.

**PCA degrades general utility more than toxic utility** — for all k values, the magnitude of Δ loss on books is *larger* than on toxic data (e.g., k=128: −2.70 books vs −1.48 toxic). This means the PCA patch is removing directions that matter more to general text than to toxic continuations specifically.

**Rank-1 drops on books under PCA** — unlike the toxic eval where rank-1 slightly increased, rank-1 *decreases* on books (e.g., k=32: −4,118 tokens). The patch is making the model less confident about book-domain top predictions while paradoxically making it slightly more confident about some toxic predictions.

**Mean-diff rank-1 is essentially unchanged on books** (−83 out of 2,733,806 tokens = 0.003% change), confirming the patch has almost no effect on the top-1 prediction distribution for benign content.

**Conclusion** — The mean-diff patch is the more precise intervention: it produces a positive toxic loss delta (+0.1002) while barely touching benign content (−0.0470). The PCA approach causes collateral damage to general utility that exceeds its effect on toxic content.
