# Mean-Diff Patch Eval Results

## Setup

- **Model:** `GSAI-ML/LLaDA-8B-Base`
- **Patch method:** Forward pre-hook on `ff_out` that projects hidden states along the direction `u = (mean_toxic - mean_benign) / ||mean_toxic - mean_benign||` before the unembedding layer. Equivalent to `ff_out.weight @ (I - u u.T)` but avoids materializing the full 4096×4096 matrix.
- **Toxic hidden states:** `allenai/real-toxicity-prompts`, 7,000 examples, diffusion-faithful (masked-token) collection → 103,423 hidden vectors
- **Benign hidden states:** `openai/gsm8k`, 7,000 examples, diffusion-faithful (masked-token) collection → 642,703 hidden vectors
- **Diff norm:** 106.8582
- **Cosine similarity of means:** 0.7794
- **Eval dataset:** `allenai/real-toxicity-prompts`, continuation loss only (prompt tokens masked from labels)
- **Eval examples:** 50,000
- **Tokens evaluated:** 727,508
- **Metric:** Per-example cross-entropy loss (mean ± std, SE); rank of correct token in ff_out output distribution (1-based)

---

## Results — 50,000 examples

| | Loss (mean ± std) | Loss SE | Rank-1 Tokens | Rank-1 % | Mean Rank |
|---|---|---|---|---|---|
| Original | 11.0084 ± 1.2216 | 0.0055 | 3075 / 727,508 | 0.42% | 624.7 |
| Patched  | 11.1086 ± 1.1987 | 0.0054 | 3526 / 727,508 | 0.48% | 1216.7 |
| **Δ**    | **+0.1002**       | —       | **+451**       | +0.06% | **+592.0** |

---

## Observations

**Loss increases after patching (Δ = +0.1002)** — unlike the PCA approach which consistently showed negative deltas, the mean-diff patch produces a positive loss delta. The model becomes slightly worse at predicting the correct toxic continuation tokens.

**Mean rank nearly doubles (624.7 → 1216.7)** — the correct toxic continuation token is pushed from rank ~625 to rank ~1217 on average, indicating the patch meaningfully degrades the model's confidence in toxic next-token predictions.

**Loss std is roughly symmetric before and after (1.2216 vs 1.1987)** — the spread of per-example losses is nearly unchanged, meaning the patch shifts the distribution rather than making it more variable.

**Rank-1 count increases slightly (+451 tokens, 0.42% → 0.48%)** — a small fraction of tokens that weren't previously top-1 become top-1 after patching, likely due to redistribution of probability mass. This is a minor secondary effect.

**Rank-1 is uninformative as a primary metric** — with 727,508 tokens and a 126,464-token vocabulary, fewer than 0.5% of tokens are ever rank-1. Mean rank is the more sensitive signal here.

---

## Comparison to PCA Patch (k=8, 50,000 examples)

| Method | Δ Loss | Orig Mean Rank | Patched Mean Rank | Rank Δ |
|--------|--------|----------------|-------------------|--------|
| Mean-diff (rank-1 subspace) | **+0.1002** | 624.7 | 1216.7 | +592.0 |
| PCA k=8  (47.2% var) | −0.3956 | 624.7 | 2073.1 | +1448.4 |
| PCA k=32 (70.7% var) | −0.7172 | 624.7 | 3818.9 | +3194.2 |
| PCA k=64 (77.8% var) | −1.0176 | 624.7 | 3550.5 | +2925.8 |
| PCA k=128 (83.1% var) | −1.4816 | 624.7 | 3558.8 | +2934.1 |

The mean-diff patch is the only method that produces a **positive Δ loss**, making it the only approach where cross-entropy is a valid suppression signal. However, the PCA patches produce much larger increases in mean rank, suggesting they more aggressively displace the correct toxic tokens in the output distribution — at the cost of also lowering loss via global confidence reduction.
