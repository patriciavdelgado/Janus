#!/bin/bash
#
#SBATCH --partition=gpu_min32gb     # Reserved partition
#SBATCH --qos=gpu_min32gb         # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=janus    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Running job in reserved partition"

# Commands / scripts to run (e.g., python3 train.py)

python3 JP1B_SUIM_10iterations.py
