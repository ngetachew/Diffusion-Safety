sbatch --wrap="module load conda/2024.09 && conda activate /home/ngetach/.conda/envs/rocky9/2024.09/CS587 && python /home/ngetach/tml/compare_means.py" \
    --account=debug --partition=scholar-debug --cpus-per-task=4 --nodes=1 --time=10:00 \
    --output=slurm_logs/compare_means_%j.out
