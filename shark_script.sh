#!/bin/bash
#SBATCH --job-name=barlow-twins-training
#SBATCH --partition=PATHgpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --chdir=/exports/path-nefro-hpc/gbuzzanca_/Foundation-Models-Pretraining/
#SBATCH --output=/exports/path-nefro-hpc/gbuzzanca_/Foundation-Models-Pretraining/%x-%j.out
#SBATCH --error=/exports/path-nefro-hpc/gbuzzanca_/Foundation-Models-Pretraining/%x-%j.err

export PYTHONUNBUFFERED=TRUE
export NCCL_SOCKET_IFNAME=team0

module load library/cuda/11.6.1/gcc.8.3.1
module load library/cudnn/11.6/cudnn

source ~/.bashrc
conda activate /exports/path-nefro-hpc/gbuzzanca_/.conda/envs/barlow_twins

srun python train.py --images-folders=/exports/path-nefro-hpc/PATCHES_REST/AMC --images-folders=/exports/path-nefro-hpc/PATCHES_REST/KUL --images-folders=/exports/path-nefro-hpc/PATCHES_REST/LUMC --images-folders=/exports/path-nefro-hpc/PATCHES_REST/UMCU --images-folders=/exports/path-nefro-hpc/PATCHES_LANCET/TRAIN --batch-size=1152
