#!/bin/bash
set -e
cd "$(dirname "$0")"

JOB1=$(sbatch --parsable \
    --export=ALL,DATASET=allenai/real-toxicity-prompts,SPLIT=train,TEXT_COLUMN=prompt,TEXT_SUBFIELD=text,LIMIT=3000,OUTPUT=/scratch/scholar/ngetach/hidden_states_toxic.pt \
    collect_hidden_states.slurm)
echo "Step 1 (collect toxic hidden states): $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
    --wrap="module load conda/2024.09 && conda activate /home/ngetach/.conda/envs/rocky9/2024.09/CS587 && python /home/ngetach/tml/compute_subspace.py --hidden_states /scratch/scholar/ngetach/hidden_states_toxic.pt --output /scratch/scholar/ngetach/subspace_toxic_64.pt" \
    --job-name=llada_subspace_toxic --account=debug --partition=scholar-debug \
    --cpus-per-task=4 --nodes=1 --time=10:00 \
    --output=slurm_logs/subspace_toxic_%j.out --error=slurm_logs/subspace_toxic_%j.err)
echo "Step 2 (compute toxic subspace):      $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 \
    --export=ALL,SUBSPACE=/scratch/scholar/ngetach/subspace_toxic_64.pt,OUTPUT=/scratch/scholar/ngetach/ppl_eval_results.pt \
    eval_perplexity.slurm)
echo "Step 3 (perplexity eval):             $JOB3"

echo ""
echo "Pipeline: $JOB1 → $JOB2 → $JOB3"
echo "Monitor:  squeue -u ngetach"
