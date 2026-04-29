#!/bin/bash
set -e
cd "$(dirname "$0")"

JOB1=$(sbatch --parsable collect_hidden_states.slurm)
echo "Step 1 (collect hidden states): $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
    --wrap="module load conda/2026.03 && conda activate /home/ngetach/.conda/envs/rocky9/2024.09/CS587 && python /home/ngetach/tml/compute_subspace.py --output /scratch/scholar/ngetach/subspace_64.pt" \
    --job-name=llada_subspace --account=debug --partition=scholar-debug \
    --cpus-per-task=4 --nodes=1 --time=10:00 \
    --output=slurm_logs/subspace_%j.out --error=slurm_logs/subspace_%j.err)
echo "Step 2 (compute subspace):      $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 patch_and_eval.slurm)
echo "Step 3 (patch and eval):        $JOB3"

echo ""
echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
echo "Monitor:  squeue -u ngetach"
