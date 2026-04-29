#!/bin/bash
set -e
cd "$(dirname "$0")"

COLLECT_LIMIT="${COLLECT_LIMIT:-3000}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
DATASET="${DATASET:-P1ayer-1/books-3-textbooks}"
SPLIT="${SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
TEXT_SUBFIELD="${TEXT_SUBFIELD:-}"
CONCAT_COLUMN="${CONCAT_COLUMN:-}"
CONCAT_SUBFIELD="${CONCAT_SUBFIELD:-}"

JOB1=$(sbatch --parsable \
    --export=ALL,DATASET=$DATASET,SPLIT=$SPLIT,TEXT_COLUMN=$TEXT_COLUMN,TEXT_SUBFIELD=$TEXT_SUBFIELD,CONCAT_COLUMN=$CONCAT_COLUMN,CONCAT_SUBFIELD=$CONCAT_SUBFIELD,LIMIT=$COLLECT_LIMIT,OUTPUT=/scratch/scholar/ngetach/hidden_states_masked.pt \
    collect_masked_states.slurm)
echo "Step 1 (collect masked hidden states): $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
    --wrap="module load conda/2024.09 && conda activate /home/ngetach/.conda/envs/rocky9/2024.09/CS587 && python /home/ngetach/tml/compute_subspace.py --hidden_states /scratch/scholar/ngetach/hidden_states_masked.pt --output /scratch/scholar/ngetach/subspace_masked_64.pt" \
    --job-name=llada_subspace_masked --account=debug --partition=scholar-debug \
    --cpus-per-task=4 --nodes=1 --time=10:00 \
    --output=slurm_logs/subspace_masked_%j.out --error=slurm_logs/subspace_masked_%j.err)
echo "Step 2 (compute masked subspace):      $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 \
    --export=ALL,SUBSPACE=/scratch/scholar/ngetach/subspace_masked_64.pt,OUTPUT=/scratch/scholar/ngetach/eval_results_masked.pt,DATASET=$DATASET,SPLIT=$SPLIT,TEXT_COLUMN=$TEXT_COLUMN,TEXT_SUBFIELD=$TEXT_SUBFIELD,LIMIT=$EVAL_LIMIT \
    patch_and_eval.slurm)
echo "Step 3 (patch and eval):               $JOB3"

echo ""
echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
echo "Monitor:  squeue -u ngetach"
