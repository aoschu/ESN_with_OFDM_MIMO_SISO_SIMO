#!/bin/bash

#SBATCH -J wesn_job                  # Job name
#SBATCH -o wesn_%j.out               # Output file with job ID
#SBATCH -e wesn_%j.err               # Error file with job ID
#SBATCH -N 1                         # Number of nodes
#SBATCH -n 1                         # Number of tasks
#SBATCH -c 4                         # CPUs per task
#SBATCH -t 2-14:01:00                # Time limit (days-hrs:min:sec)

#SBATCH -p wiss                      # Partition/queue
#SBATCH --mem=24G                    # Total memory
#SBATCH --mail-type=ALL
#SBATCH --exclude=anatolien,airmont,montevideo,monaco,montreal

# Load Conda and activate the required environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wesn

# Load hostname and GPU info
HOSTNAME=$(hostname)
nvidia-smi || echo "No GPU available or nvidia-smi not found"

# Navigate to the cloned repo
cd ~/Windowed-ESN-for-MIMO-OFDM

# Run the script
echo "Running on host: $HOSTNAME"
srun python3 main.py
echo "Finished running on: $HOSTNAME"

wait
echo "$(date): Finished jobs!"

