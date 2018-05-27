#!/bin/bash
#SBATCH --job-name=hw3
#SBATCH --output=particle-filter.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kcdasilv@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:1

module load cuda/cuda-9.1

./particle-filter.exe
