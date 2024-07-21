#!/bin/bash
#SBATCH --job-name="msn soft training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/storage/scratch/amokhtar/msn-shape-completion-modified/logs/soft/slurm-%j.out



module load cuda/10.0

/home/stud/ahah/miniconda3/envs/msn/bin/python -u train.py

# watch -n 0.5 'tail -n 2 /storage/scratch/amokhtar/msn-shape-completion-modified/logs/soft/slurm-%j.out'
#2024-07-20T20:34:03.871882