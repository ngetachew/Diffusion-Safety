#!/bin/bash
# Runs the full masked pipeline once, then fans out compute_subspace + patch_and_eval
# for each k in K_VALUES. All subspace jobs run in parallel after hidden state collection.
#
# Usage:
#   bash run_k_sweep.sh
#   COLLECT_LIMIT=200 DATASET=allenai/real-toxicity-prompts ... bash run_k_sweep.sh
set -e
cd "$(dirname "$0")"

K_VALUES=(8 32 64 128)

COLLECT_LIMIT="${COLLECT_LIMIT:-3000}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
DATASET="${DATASET:-P1ayer-1/books-3-textbooks}"
SPLIT="${SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
TEXT_SUBFIELD="${TEXT_SUBFIELD:-}"
CONCAT_COLUMN="${CONCAT_COLUMN:-}"
CONCAT_SUBFIELD="${CONCAT_SUBFIELD:-}"

HIDDEN_STATES=/scratch/scholar/ngetach/hidden_states_masked.pt

# ── Step 1: collect hidden states (once) ─────────────────────────────────────
JOB1=$(sbatch --parsable \
    --export=ALL,DATASET=$DATASET,SPLIT=$SPLIT,TEXT_COLUMN=$TEXT_COLUMN,TEXT_SUBFIELD=$TEXT_SUBFIELD,CONCAT_COLUMN=$CONCAT_COLUMN,CONCAT_SUBFIELD=$CONCAT_SUBFIELD,LIMIT=$COLLECT_LIMIT,OUTPUT=$HIDDEN_STATES \
    collect_masked_states.slurm)
echo "Step 1 (collect masked hidden states): $JOB1"
echo ""

# ── Steps 2+3: chain subspace jobs serially (debug account allows 1 at a time)
# eval jobs depend only on their own subspace job and run in parallel on gpu.
PREV_SUBSPACE_JOB=$JOB1
for K in "${K_VALUES[@]}"; do
    SUBSPACE=/scratch/scholar/ngetach/subspace_masked_k${K}.pt
    EVAL_OUT=/scratch/scholar/ngetach/eval_results_masked_k${K}.pt

    JOB2=$(sbatch --parsable --dependency=afterok:$PREV_SUBSPACE_JOB \
        --wrap="module load conda/2024.09 && conda activate /home/ngetach/.conda/envs/rocky9/2024.09/CS587 && python /home/ngetach/tml/compute_subspace.py --hidden_states $HIDDEN_STATES --k $K --output $SUBSPACE" \
        --job-name=llada_subspace_k${K} --account=debug --partition=scholar-debug \
        --cpus-per-task=4 --nodes=1 --time=10:00 \
        --output=slurm_logs/subspace_k${K}_%j.out --error=slurm_logs/subspace_k${K}_%j.err)

    JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 \
        --export=ALL,SUBSPACE=$SUBSPACE,OUTPUT=$EVAL_OUT,DATASET=$DATASET,SPLIT=$SPLIT,TEXT_COLUMN=$TEXT_COLUMN,TEXT_SUBFIELD=$TEXT_SUBFIELD,LIMIT=$EVAL_LIMIT \
        patch_and_eval.slurm)

    echo "  k=$K: subspace=$JOB2, eval=$JOB3"
    echo "         subspace → $SUBSPACE"
    echo "         results  → $EVAL_OUT"

    PREV_SUBSPACE_JOB=$JOB2
done

echo ""
echo "Monitor: squeue -u ngetach"
