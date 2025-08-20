#!/bin/bash
#SBATCH -J ESN48       # Job name (no spaces or quote it)
#SBATCH -o ESN48.%j.out                   # Stdout (include jobid to avoid overwrite)
#SBATCH -e ESN48.%j.err                   # Stderr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 2-14:01:00
#SBATCH -p wiss
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
#SBATCH --exclude=anatolien,airmont,montevideo,monaco,montreal

# If you need a GPU, uncomment the next line (and ensure the partition supports it)
# #SBATCH --gres=gpu:1

# Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sionna_test

HOSTNAME=$(hostname)

nvidia-smi || true

srun echo "$HOSTNAME"
srun python3 Demo_MIMO_4x8_Sionna_CDL_ESN_v2.py
srun echo "$HOSTNAME"

echo "$(date): Finished jobs!"

