#!/bin/bash
#SBATCH --job-name=mnist-sif
#SBATCH --partition=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

DIR=/mnt/appl/singularity_images
IMG=${IMG:-$DIR/pytorch-24.07-py3.sif}

[ -f data/mnist.pt ] || apptainer exec "$IMG" python src/data.py
apptainer exec --nv "$IMG" python src/train.py
