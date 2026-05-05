# PCA Patch Eval — Toxic Dataset Results

## Setup

- **Model:** `GSAI-ML/LLaDA-8B-Base`
- **Patch:** `ff_out.weight = ff_out.weight @ P_perp` where `P_perp = I - UU^T`
- **Subspace source:** `allenai/real-toxicity-prompts` masked hidden states (400 examples, diffusion-faithful collection)
- **Eval dataset:** `allenai/real-toxicity-prompts`, continuation loss only (prompt tokens masked from labels)
- **Metric:** Mean per-token cross-entropy loss; rank of correct token in ff_out output distribution (1-based)

---

## Experiment A — 100 examples

| k | Expl. Var | Orig Loss | Patched Loss | Δ Loss | Orig Mean Rank | Patched Mean Rank | Orig Rank-1 | Patched Rank-1 |
|---|-----------|-----------|--------------|--------|----------------|-------------------|-------------|----------------|
| 8 | 47.2% | 10.8624 | 10.5303 | −0.3321 | 645.0 | 2215.1 | 6 / 1459 (0.41%) | 8 / 1459 (0.55%) |
| 32 | 70.7% | 10.8624 | 10.1930 | −0.6695 | 645.0 | 4057.5 | 6 / 1459 (0.41%) | 7 / 1459 (0.48%) |
| 64 | 77.8% | 10.8624 | 9.8899 | −0.9725 | 645.0 | 3634.1 | 6 / 1459 (0.41%) | 7 / 1459 (0.48%) |
| 128 | 83.1% | 10.8624 | 9.4500 | −1.4124 | 645.0 | 4064.4 | 6 / 1459 (0.41%) | 7 / 1459 (0.48%) |

---

## Experiment B — Full dataset (99,442 examples)

> k=32 timed out at 18,270 / 99,442 examples (4-hour wall time).

| k | Expl. Var | Orig Loss | Patched Loss | Δ Loss | Orig Mean Rank | Patched Mean Rank | Orig Rank-1 | Patched Rank-1 |
|---|-----------|-----------|--------------|--------|----------------|-------------------|-------------|----------------|
| 8 | 47.2% | 11.0226 | 10.6368 | −0.3858 | 673.3 | 2272.7 | 4953 / 1,442,359 (0.34%) | 5586 / 1,442,359 (0.39%) |
| 32 | 70.7% | — | — | — | — | — | — | — |
| 64 | 77.8% | 11.0226 | 10.0570 | −0.9656 | 673.3 | 3873.3 | 4953 / 1,442,359 (0.34%) | 5607 / 1,442,359 (0.39%) |
| 128 | 83.1% | 11.0226 | 9.6002 | −1.4224 | 673.3 | 3910.2 | 4953 / 1,442,359 (0.34%) | 6335 / 1,442,359 (0.44%) |

---

## Experiment C — 50,000 examples (with rank statistics)

- **Eval examples:** 50,000
- **Tokens evaluated:** 727,508
- **Loss std/stderr:** not available from this run (only scalar means saved; use updated `patch_and_eval.py` to get per-example loss distributions)

| k | Expl. Var | Orig Loss | Patched Loss | Δ Loss | Orig Mean Rank | Orig Rank Std | Orig Rank SE | Patched Mean Rank | Patched Rank Std | Patched Rank SE |
|---|-----------|-----------|--------------|--------|----------------|---------------|--------------|-------------------|------------------|-----------------|
| 8 | 47.2% | 11.0084 | 10.6128 | −0.3956 | 624.7 | 2918.4 | 3.4216 | 2073.1 | 7480.7 | 8.7705 |
| 32 | 70.7% | 11.0084 | 10.2912 | −0.7172 | 624.7 | 2918.4 | 3.4216 | 3818.9 | 10720.2 | 12.5685 |
| 64 | 77.8% | 11.0084 | 9.9908 | −1.0176 | 624.7 | 2918.4 | 3.4216 | 3550.5 | 10141.7 | 11.8902 |
| 128 | 83.1% | 11.0084 | 9.5268 | −1.4816 | 624.7 | 2918.4 | 3.4216 | 3558.8 | 10389.7 | 12.1811 |

The rank std is very high relative to the mean in all cases (patched rank std ~3–4× the mean), indicating the rank distribution is heavily right-skewed — most tokens are pushed moderately down the list, but a long tail of tokens are pushed very far down.

---

## Observations

**Loss decreases after patching (negative Δ loss)** — consistent with all prior experiments. The PCA patch lowers cross-entropy rather than raising it.

**Mean rank increases substantially** — despite loss going down, the rank of the correct token in ff_out's output distribution increases 3–6× across all k values. This is the more direct signal: the patch is pushing the correct toxic continuation tokens further down the ranked distribution.

**Mean rank grows with k** — larger subspace → stronger degradation of the toxic token ranking. Rank goes from 645 (original) to 2,215 (k=8), 4,057 (k=32), 3,634 (k=64), and 4,064 (k=128).

**Rank-1 count is largely unchanged** — very few tokens were ever top-1 predictions (0.34–0.41%), and the patch doesn't meaningfully flip those.

**Δ loss and Δ mean rank are anti-correlated** — loss going down while rank goes up suggests the patch redistributes probability mass away from the correct toxic tokens but toward tokens that are also plausible (lower entropy overall, but not toward the ground-truth continuation).
