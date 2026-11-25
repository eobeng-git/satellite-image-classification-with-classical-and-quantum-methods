#!/bin/bash
#SBATCH --job-name=qnn-v4
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=240:00:00   # 10 days = maximum allowed on the cluster
#SBATCH --output=qnn_v4_%j.out
#SBATCH --error=qnn_v4_%j.err
source ~/.bashrc
conda activate qnn-310
cd /NFSHOME/eobeng/src.disim
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"
# the script must compile before the experiment starts (review point 1)
python -m py_compile alg_experiment_qnn_fixed_V4.py || { echo "COMPILE ERROR - stopping"; exit 1; }
python -u alg_experiment_qnn_fixed_V4.py
echo "Job finished at $(date) with exit code $?"
