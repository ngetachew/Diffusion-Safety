Created by Nathaniel Getachew for CS692-TML at Purdue University

# Targeted Concept Suppression in Masked Diffusion LMs

This repository explores weight-space interventions on [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base), a masked diffusion language model, with the goal of suppressing specific concepts (e.g. toxic continuations) without degrading general language modeling ability.

The core idea: collect hidden states at the final layer (`ff_out` input) for a set of undesired prompts, estimate a low-dimensional subspace of those states, and project it out of the unembedding weight matrix so the model can no longer express those directions.

Two suppression methods are compared:

- **PCA patch** — projects out the top-*k* principal components of the toxic hidden state distribution
- **Mean-diff patch** — projects out the rank-1 direction `u = (mean_toxic − mean_benign) / ‖·‖`

---

## Setup

**Requirements:** Python 3.10+, PyTorch, HuggingFace `transformers`, `datasets`, `accelerate`, `python-dotenv`, `numpy`, `matplotlib`.

Create a `.env` file in the repo root:

```
HF_TOKEN=<your_huggingface_token>
HF_CACHE=/path/to/hf_model_cache
```

The model (`LLaDA-8B-Base`) requires ~16 GB of GPU memory in bfloat16. All GPU scripts are written for SLURM but can be run directly on any machine with a suitable GPU.

---

## Pipeline Overview

```
collect_masked_states.py   →   compute_subspace.py   →   patch_and_eval.py
       (toxic)                       (PCA)                  (PCA patch eval)

collect_masked_states.py   →   compare_means.py   →   patch_and_eval_meandiff.py
  (toxic + benign)                                        (mean-diff patch eval)
```

---

## Core Scripts

### 1. `collect_masked_states.py`

Loads LLaDA-8B-Base and runs a forward pass over a HuggingFace dataset, collecting hidden states at the `ff_out` input layer for each masked token position. Uses a diffusion-faithful masking strategy (random mask ratio per example).

**Output:** a `.pt` file with keys `hidden_states` ([N, 4096]), `texts`, `mask_positions`, `mask_ratios`.

```bash
python collect_masked_states.py \
    --dataset    allenai/real-toxicity-prompts \
    --split      train \
    --text_column prompt \
    --text_subfield text \
    --limit      7000 \
    --output     /path/to/hidden_states_toxic.pt

# For a dataset with a plain text column (no nesting):
python collect_masked_states.py \
    --dataset    openai/gsm8k \
    --dataset_config main \
    --split      train \
    --text_column question \
    --limit      7000 \
    --output     /path/to/hidden_states_benign.pt
```

**SLURM:** `collect_masked_states.slurm` (accepts the same args via environment variables `DATASET`, `TEXT_COLUMN`, `LIMIT`, `OUTPUT`, etc.)

---

### 2. `compute_subspace.py`

Runs truncated SVD on the collected hidden states to find the top-*k* principal components of the toxic distribution.

**Output:** a `.pt` file with keys `U` ([4096, k]), `P_perp` ([4096, 4096]), `k`, `explained_variance`, `singular_values`.

```bash
python compute_subspace.py \
    --hidden_states /path/to/hidden_states_toxic.pt \
    --k             64 \
    --output        /path/to/subspace_k64.pt
```

---

### 3. `compare_means.py`

Computes and compares the mean hidden state vectors for two datasets (toxic and benign). Reports the difference vector, its norm, cosine similarity, and a per-dimension breakdown of the largest differences. Used to validate that the two distributions are meaningfully separated before running the mean-diff patch.

```bash
# Edit the two file paths at the top of the script, then:
python compare_means.py
```

---

### 4. `patch_and_eval.py` — PCA patch

Applies the PCA subspace patch to `ff_out` via a forward pre-hook:

```
h_new = h − (h @ U) @ Uᵀ
```

Then evaluates both the original and patched model on any HuggingFace text dataset, reporting per-example cross-entropy loss (mean ± std, SE) and per-token rank statistics.

```bash
python patch_and_eval.py \
    --subspace    /path/to/subspace_k64.pt \
    --dataset     allenai/real-toxicity-prompts \
    --split       train \
    --text_column prompt \
    --text_subfield text \
    --continuation_column continuation \
    --continuation_subfield text \
    --limit       50000 \
    --output      /path/to/eval_results_pca_k64.pt
```

**Key flags:**
- `--continuation_column` / `--continuation_subfield` — compute loss only on continuation tokens (masks the prompt)
- `--limit` — cap the number of examples evaluated
- `--renormalize` — rescale the patched weight to the original Frobenius norm after projection

**SLURM:** `patch_and_eval.slurm` (uses env vars `SUBSPACE`, `DATASET`, `TEXT_COLUMN`, `LIMIT`, `OUTPUT`, etc.)

---

### 5. `patch_and_eval_meandiff.py` — Mean-diff patch

Same evaluation harness as above, but computes the rank-1 suppression direction from the difference of means of two hidden state files:

```
u = (mean_toxic − mean_benign) / ‖mean_toxic − mean_benign‖
h_new = h − (h · u) u
```

```bash
python patch_and_eval_meandiff.py \
    --toxic_states  /path/to/hidden_states_toxic.pt \
    --benign_states /path/to/hidden_states_benign.pt \
    --dataset       allenai/real-toxicity-prompts \
    --split         train \
    --text_column   prompt \
    --text_subfield text \
    --continuation_column continuation \
    --continuation_subfield text \
    --limit         50000 \
    --output        /path/to/eval_results_meandiff.pt
```

**SLURM:** `patch_and_eval_meandiff.slurm`

---

### 6. `plot_pca_projections.py`

Projects hidden states from multiple datasets onto the first two principal components of the toxic PCA subspace and saves scatter plots — one per *k* value. Useful for visualising how well the toxic subspace separates from benign distributions.

```bash
python plot_pca_projections.py \
    --toxic_states /path/to/hidden_states_toxic.pt \
    --gsm8k_states /path/to/hidden_states_benign.pt \
    --books_states /path/to/hidden_states_books.pt \
    --subspace_dir /path/to/subspace_dir \
    --n_samples    5000 \
    --out_dir      ./plots
```

Outputs `plots/pca_projection_k{8,32,64,128}.png`.

---

## Pipeline Shell Scripts

| Script | What it does |
|--------|-------------|
| `run_masked_pipeline.sh` | Collect hidden states → compute subspace → eval (PCA, single k) |
| `run_k_sweep.sh` | Submit PCA eval jobs for k ∈ {8, 32, 64, 128} in parallel |
| `run_meandiff_pipeline.sh` | Collect toxic + benign hidden states in parallel → mean-diff eval |

---

## Results Summary

Evaluated on `allenai/real-toxicity-prompts` (50k examples, continuation loss only) and `P1ayer-1/books-3-textbooks` (5,437 examples, full-text loss). See [`pca_eval_results.md`](pca_eval_results.md), [`meandiff_eval_results.md`](meandiff_eval_results.md), and [`books_eval_results.md`](books_eval_results.md) for full tables.

| Method | Δ Loss (Toxic) | Δ Mean Rank (Toxic) | Δ Loss (Books) |
|--------|---------------|---------------------|----------------|
| Mean-diff | **+0.10** | +592 | −0.05 |
| PCA k=8 | −0.40 | +1,448 | −0.55 |
| PCA k=32 | −0.72 | +3,194 | −1.29 |
| PCA k=64 | −1.02 | +2,926 | −1.77 |
| PCA k=128 | −1.48 | +2,934 | −2.70 |

The mean-diff patch is the only method that produces a positive toxic loss delta while leaving benign content nearly untouched. PCA patches produce larger rank displacement on toxic tokens but also cause substantial collateral damage to general utility.

---

## Model Architecture (LLaDA-8B-Base)

The patch targets `model.transformer.ff_out`, the final unembedding linear layer mapping hidden dim 4096 → vocab size 126,464.

```
LLaDAModelLM
└── model: LLaDAModel
    └── transformer: ModuleDict
        ├── wte:    Embedding(126464, 4096)
        ├── blocks: 32 × LLaDALlamaBlock
        │           (attn + FFN with SiLU, RoPE, RMSNorm)
        └── ff_out: Linear(4096 → 126464)   ← patch applied here
```
