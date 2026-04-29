#!/bin/bash
# Full mean-diff patch pipeline:
#   1. Collect masked hidden states for toxic dataset   (parallel)
#   2. Collect masked hidden states for benign dataset  (parallel)
#   3. Patch ff_out with (mean_toxic - mean_benign) direction and evaluate
#
# Usage:
#   bash run_meandiff_pipeline.sh
#
# All parameters are set via environment variables. Defaults reproduce the
# current experiment (real-toxicity-prompts vs. GSM8K, eval on toxic).
#
# Example — swap benign dataset:
#   BENIGN_DATASET=P1ayer-1/books-3-textbooks BENIGN_TEXT_COLUMN=text \
#   BENIGN_CONCAT_COLUMN= RUN_NAME=toxic_vs_books bash run_meandiff_pipeline.sh

set -e
cd "$(dirname "$0")"

# ── Identifer for output files ────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-meandiff}"
SCRATCH="${SCRATCH:-/scratch/scholar/ngetach}"

# ── Hidden state collection ───────────────────────────────────────────────────
COLLECT_LIMIT="${COLLECT_LIMIT:-7000}"   # same cap applied to both datasets

# Toxic dataset
TOXIC_DATASET="${TOXIC_DATASET:-allenai/real-toxicity-prompts}"
TOXIC_DATASET_CONFIG="${TOXIC_DATASET_CONFIG:-}"
TOXIC_SPLIT="${TOXIC_SPLIT:-train}"
TOXIC_TEXT_COLUMN="${TOXIC_TEXT_COLUMN:-prompt}"
TOXIC_TEXT_SUBFIELD="${TOXIC_TEXT_SUBFIELD:-text}"
TOXIC_CONCAT_COLUMN="${TOXIC_CONCAT_COLUMN:-continuation}"
TOXIC_CONCAT_SUBFIELD="${TOXIC_CONCAT_SUBFIELD:-text}"

# Benign dataset
BENIGN_DATASET="${BENIGN_DATASET:-openai/gsm8k}"
BENIGN_DATASET_CONFIG="${BENIGN_DATASET_CONFIG:-main}"
BENIGN_SPLIT="${BENIGN_SPLIT:-train}"
BENIGN_TEXT_COLUMN="${BENIGN_TEXT_COLUMN:-question}"
BENIGN_TEXT_SUBFIELD="${BENIGN_TEXT_SUBFIELD:-}"
BENIGN_CONCAT_COLUMN="${BENIGN_CONCAT_COLUMN:-answer}"
BENIGN_CONCAT_SUBFIELD="${BENIGN_CONCAT_SUBFIELD:-}"

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_DATASET="${EVAL_DATASET:-allenai/real-toxicity-prompts}"
EVAL_DATASET_CONFIG="${EVAL_DATASET_CONFIG:-}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
EVAL_TEXT_COLUMN="${EVAL_TEXT_COLUMN:-prompt}"
EVAL_TEXT_SUBFIELD="${EVAL_TEXT_SUBFIELD:-text}"
EVAL_CONTINUATION_COLUMN="${EVAL_CONTINUATION_COLUMN:-continuation}"
EVAL_CONTINUATION_SUBFIELD="${EVAL_CONTINUATION_SUBFIELD:-text}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"

# ── Output paths ──────────────────────────────────────────────────────────────
TOXIC_STATES="${SCRATCH}/hidden_states_toxic_${RUN_NAME}.pt"
BENIGN_STATES="${SCRATCH}/hidden_states_benign_${RUN_NAME}.pt"
EVAL_OUTPUT="${SCRATCH}/eval_results_meandiff_${RUN_NAME}.pt"

echo "=== Mean-diff pipeline: $RUN_NAME ==="
echo "  Toxic  dataset : $TOXIC_DATASET (limit=$COLLECT_LIMIT) → $TOXIC_STATES"
echo "  Benign dataset : $BENIGN_DATASET (limit=$COLLECT_LIMIT) → $BENIGN_STATES"
echo "  Eval   dataset : $EVAL_DATASET (limit=$EVAL_LIMIT)"
echo "  Output         : $EVAL_OUTPUT"
echo ""

# ── Step 1: collect toxic hidden states ──────────────────────────────────────
JOB_TOXIC=$(sbatch --parsable \
    --export=ALL,\
DATASET=$TOXIC_DATASET,\
DATASET_CONFIG=$TOXIC_DATASET_CONFIG,\
SPLIT=$TOXIC_SPLIT,\
TEXT_COLUMN=$TOXIC_TEXT_COLUMN,\
TEXT_SUBFIELD=$TOXIC_TEXT_SUBFIELD,\
CONCAT_COLUMN=$TOXIC_CONCAT_COLUMN,\
CONCAT_SUBFIELD=$TOXIC_CONCAT_SUBFIELD,\
LIMIT=$COLLECT_LIMIT,\
OUTPUT=$TOXIC_STATES \
    collect_masked_states.slurm)
echo "Step 1a (toxic  hidden states): job=$JOB_TOXIC → $TOXIC_STATES"

# ── Step 2: collect benign hidden states (parallel with step 1) ───────────────
JOB_BENIGN=$(sbatch --parsable \
    --export=ALL,\
DATASET=$BENIGN_DATASET,\
DATASET_CONFIG=$BENIGN_DATASET_CONFIG,\
SPLIT=$BENIGN_SPLIT,\
TEXT_COLUMN=$BENIGN_TEXT_COLUMN,\
TEXT_SUBFIELD=$BENIGN_TEXT_SUBFIELD,\
CONCAT_COLUMN=$BENIGN_CONCAT_COLUMN,\
CONCAT_SUBFIELD=$BENIGN_CONCAT_SUBFIELD,\
LIMIT=$COLLECT_LIMIT,\
OUTPUT=$BENIGN_STATES \
    collect_masked_states.slurm)
echo "Step 1b (benign hidden states): job=$JOB_BENIGN → $BENIGN_STATES"

# ── Step 3: patch and eval (runs after both collection jobs succeed) ──────────
JOB_EVAL=$(sbatch --parsable \
    --dependency=afterok:${JOB_TOXIC}:${JOB_BENIGN} \
    --export=ALL,\
TOXIC_STATES=$TOXIC_STATES,\
BENIGN_STATES=$BENIGN_STATES,\
DATASET=$EVAL_DATASET,\
DATASET_CONFIG=$EVAL_DATASET_CONFIG,\
SPLIT=$EVAL_SPLIT,\
TEXT_COLUMN=$EVAL_TEXT_COLUMN,\
TEXT_SUBFIELD=$EVAL_TEXT_SUBFIELD,\
CONTINUATION_COLUMN=$EVAL_CONTINUATION_COLUMN,\
CONTINUATION_SUBFIELD=$EVAL_CONTINUATION_SUBFIELD,\
LIMIT=$EVAL_LIMIT,\
OUTPUT=$EVAL_OUTPUT \
    patch_and_eval_meandiff.slurm)
echo "Step 2  (patch & eval):         job=$JOB_EVAL → $EVAL_OUTPUT"

echo ""
echo "Pipeline: [$JOB_TOXIC, $JOB_BENIGN] → $JOB_EVAL"
echo "Monitor:  squeue -u ngetach"
