#!/bin/bash
#SBATCH --job-name=ml_a100_job
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1           # Request 1 A100 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00             # Max time: 12 hours
#SBATCH --mail-type=END
#SBATCH --mail-user=pl7682@princeton.edu

# Load necessary modules
module purge
module load anaconda3

# Activate conda environment
source activate ml_env

# Run your Python script
python run.py