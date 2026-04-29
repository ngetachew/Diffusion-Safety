# LLaDA-8B-Base Weight Patching — Experiment Summary

## Goal

Modify the weights of [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) to suppress its ability to generate toxic or undesired content. The approach:

1. Collect hidden states from undesired prompts at the final layer (`ln_f` output, dim 4096)
2. Compute a low-dimensional subspace `S` via PCA, with orthonormal basis `U` ∈ ℝ^{4096 × k}
3. Construct projection matrix `P = I - UUᵀ` (projects onto the complement of `S`)
4. Patch the final output weight: `ff_out.weight_new = ff_out.weight @ P`

This bakes the projection into the weight so that for any hidden state `h`, the patched model computes `ff_out(P @ h)`, effectively nullifying directions associated with undesired content.

---

## Model

- **Model:** `GSAI-ML/LLaDA-8B-Base` (8B parameter masked diffusion LM)
- **Precision:** bfloat16
- **Patched layer:** `model.model.transformer.ff_out` — the final projection from hidden dim (4096) to vocab (126,464)
- **Transformers version:** 4.46.3 (pinned — model was built against this version; 5.x is incompatible)

---

## Pipeline

Three scripts run in sequence as a SLURM chain:

### Step 1: Hidden State Collection

Two strategies were implemented:

#### `collect_hidden_states.py` — Causal (last-token)
- Feeds clean tokenized text to the model
- Captures the hidden state at the **last non-padding token** via a forward hook on `ln_f`
- One hidden vector per example

#### `collect_masked_states.py` — Diffusion-faithful (masked tokens)
- For each example, samples a mask ratio `t ~ Uniform(0, 1)` and independently masks each token with probability `t`
- Feeds the corrupted sequence to the model using `output_hidden_states=True`
- Captures hidden states at **all masked token positions**
- Many hidden vectors per example (richer subspace estimate)
- Uses LLaDA's native mask token `<|mdm_mask|>` (id 126336, from `model.config.mask_token_id`)

### Step 2: `compute_subspace.py`
- Loads hidden states, mean-centers them
- Runs truncated SVD to extract the top-k principal components
- Saves `U` (basis), `P_perp` (projection matrix), `mean`, `k`, and explained variance

### Step 3: `patch_and_eval.py`
- Loads model and dataset
- Evaluates mean per-token cross-entropy loss on the **original** model
- Applies patch: `ff_out.weight = ff_out.weight @ P_perp`
- Evaluates loss on the **patched** model
- Reports delta = patched − original

**Evaluation metric:** Mean per-token cross-entropy loss (equivalent to log-perplexity per token). When `--continuation_column` is set, loss is computed only on continuation tokens — the prompt tokens are masked out in the label tensor (`labels[:, :prompt_len] = -100`).

---

## Datasets

| Role | Dataset | Columns used |
|------|---------|-------------|
| Hidden state collection & eval (general) | `P1ayer-1/books-3-textbooks` | `text` |
| Hidden state collection & eval (toxic) | `allenai/real-toxicity-prompts` | `prompt["text"]` + `continuation["text"]` |

For the toxic dataset, hidden states were collected from the concatenation of prompt + continuation to capture the full toxic context.

---

## Experiments

### Experiment 1: Original pipeline (causal hidden states, books subspace, k=64)

- **Hidden states:** 2,956 examples from `books-3-textbooks`, last-token strategy
- **Subspace:** k=64, explained variance = 69.8%
- **Eval dataset:** `books-3-textbooks`, 1,000 examples, full-text loss

| | Loss |
|---|---|
| Original | 13.6966 |
| Patched | 10.7685 |
| Δ | −2.928 |

---

### Experiment 2: Masked hidden states + k sweep (toxic subspace, toxic eval)

- **Hidden states:** 400 examples from `allenai/real-toxicity-prompts` (prompt + continuation), masked-token strategy
- **Eval dataset:** `allenai/real-toxicity-prompts`, 1,000 examples, **continuation tokens only**

| k | Expl. Var | Orig Loss | Patched Loss | Δ |
|---|-----------|-----------|-------------|---|
| 8 | 47.2% | 10.9198 | 10.5616 | −0.358 |
| 32 | 70.7% | 10.9198 | 10.2470 | −0.673 |
| 64 | 77.8% | 10.9198 | 9.9439 | −0.976 |
| 128 | 83.1% | 10.9198 | 9.4911 | −1.429 |

---

### Experiment 3: Same subspace, books eval

Same masked subspaces (built from toxic data), evaluated on `books-3-textbooks` (full-text loss, 20 examples).

| k | Expl. Var | Orig Loss | Patched Loss | Δ |
|---|-----------|-----------|-------------|---|
| 8 | 47.2% | 13.7180 | 13.1708 | −0.547 |
| 32 | 70.7% | 13.7180 | 12.4274 | −1.291 |
| 64 | 77.8% | 13.7180 | 11.9657 | −1.752 |
| 128 | 83.1% | 13.7180 | 11.0754 | −2.643 |

---

### Experiment 4: Renormalization diagnostic

Hypothesis: patching reduces the Frobenius norm of `ff_out.weight`, artificially lowering loss.

**Measured norm change:**

| k | Norm before | Norm after | Ratio |
|---|-------------|-----------|-------|
| 8 | 342.75 | 342.49 | 0.9992 |
| 32 | 342.75 | 342.22 | 0.9985 |
| 64 | 342.75 | 341.24 | 0.9956 |
| 128 | 342.75 | 339.15 | 0.9895 |

Renormalizing to restore original norm made **negligible difference** to the delta — norm reduction is not the cause.

| k | Base Δ | Renorm Δ |
|---|--------|----------|
| 8 | −0.358 | −0.358 |
| 32 | −0.673 | −0.673 |
| 64 | −0.976 | −0.947 |
| 128 | −1.429 | −1.396 |

---

## Observations

1. **Loss decreases after patching across all experiments** — the patch consistently lowers cross-entropy rather than raising it, opposite to the desired effect.

2. **Books loss drops more than toxic loss** — at k=128, the books delta is −2.64 vs −1.43 for toxic continuations. The patch degrades general text predictions more than toxic ones, which is a mild signal that some content-specific structure is being preserved.

3. **Norm reduction is not the cause** — norm drops by at most 1% (k=128), and renormalization has no meaningful effect.

4. **Possible explanations for loss decrease:**
   - Projecting out top-PCA directions of the hidden states may be reducing output confidence globally, which can lower teacher-forced cross-entropy even without improving content selectivity
   - The evaluation metric (teacher-forced loss) may not be the right signal — generative evaluation (sampling and measuring toxicity of outputs) would be a more faithful test

## Artifacts

| File | Description |
|------|-------------|
| `hidden_states.pt` | Last-token hidden states, books dataset |
| `hidden_states_masked.pt` | Masked-token hidden states, toxic dataset (prompt+continuation) |
| `subspace_64.pt` | Original subspace, k=64, books data |
| `subspace_masked_k{8,32,64,128}.pt` | Masked subspaces for each k, toxic data |
| `eval_results.pt` | Original pipeline result |
| `eval_results_books_k{8,32,64,128}.pt` | Books eval, masked subspace, per k |
| `eval_results_masked_k{8,32,64,128}.pt` | Toxic continuation eval, per k |
| `eval_results_masked_k{8,32,64,128}_renorm.pt` | Same with renormalization |
