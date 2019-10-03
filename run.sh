#!/bin/bash
#SBATCH --job-name="DL-MAI EMBEDDINGS"
#SBATCH --workdir=.
#SBATCH --qos="debug"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=logs/stdout.log
#SBATCH --error=logs/stderr.log
#SBATCH --time=01:30:00

echo "Run with params: $@"
source initialize.sh "$@"
python train.py "$@"
