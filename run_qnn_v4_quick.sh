#!/bin/bash
#SBATCH --job-name=qnn-v4-quick
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=qnn_v4_quick_%j.out
#SBATCH --error=qnn_v4_quick_%j.err
source ~/.bashrc
conda activate qnn-310
cd /NFSHOME/eobeng/src.disim
echo "Job started at $(date)"
# the script must compile before anything runs
python -m py_compile alg_experiment_qnn_fixed_V4.py || { echo "COMPILE ERROR - stopping"; exit 1; }
# small three-fold check: 3000 pixels, 2 QNN epochs (review point 2)
python -u alg_experiment_qnn_fixed_V4.py --quick
echo "Job finished at $(date) with exit code $?"
