#!/bin/bash
#SBATCH --job-name=qnn-full
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00   # 7 DAYS (168 hours)
#SBATCH --output=qnn_full_%j.out
#SBATCH --error=qnn_full_%j.err
#SBATCH --mail-user=emmanuel.obeng@student.univaq.it
#SBATCH --mail-type=ALL
  # More memory for full dataset
source ~/.bashrc
conda activate qnn-310
cd /NFSHOME/eobeng/src.disim
echo "Job started at $(date)"
echo "Running FULL 7-DAY QNN experiment (multiclass PennyQNN)"
echo "Partition: cuda"
echo "Time: 7 days"
# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
# Run the full experiment
python -u alg_experiment_qnn_fixed_V2_multiclass.py
echo "Job finished at $(date)"
