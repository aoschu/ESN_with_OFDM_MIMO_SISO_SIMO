#!/bin/bash

#SBATCH -J sionna_blockfading                  # Job name
#SBATCH -o sionna_blockfading.out              # Output file (%j expands to job ID)
#SBATCH -e sionna_blockfading.err              # Error file (%j expands to job ID)
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 1                        # Number of tasks
#SBATCH -c 4                        # CPUs per task
#SBATCH -t 2-14:01:00               # Time limit (days-hrs:min:sec)

#SBATCH -p wiss                     # Queue or partition name
#SBATCH --mem=24G
#SBATCH --mail-type=ALL            # Select type of email messages to be sent (all)
#SBATCH --exclude=anatolien,airmont,montevideo,monaco,montreal  # Exclude problematic nodes

# Load Conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sionna_test

# Load hostname
HOSTNAME=$(hostname)

nvidia-smi

# Execute the job
srun echo $HOSTNAME
srun python3  sionna_blockfading_timechannel_esn.py
srun echo $HOSTNAME

wait
echo "$(date): Finished jobs!"

