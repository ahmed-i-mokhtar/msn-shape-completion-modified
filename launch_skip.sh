#!/bin/bash
#SBATCH --job-name="msn skip training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=/storage/scratch/amokhtar/msn-shape-completion-modified/logs/skip/slurm-%j.out



module load cuda/10.0

/home/stud/ahah/miniconda3/envs/msn/bin/python -u train.py

# watch -n 0.5 'tail -n 2 /storage/scratch/amokhtar/msn-shape-completion-modified/logs/original/slurm-%j.out'
#2024-07-21T12:54:44.024329