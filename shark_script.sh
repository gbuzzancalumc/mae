#!/bin/bash
#SBATCH --job-name=mae-training
#SBATCH --partition=PATHgpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb

module load library/cuda/11.6.1/gcc.8.3.1
module load library/cudnn/11.6/cudnn

source ~/.bashrc
conda activate /exports/path-nefro-hpc/gbuzzanca_/conda_envs/mae_pretraining

srun python main_pretrain.py --data_dir /exports/path-nefro-hpc/gbuzzanca_/train_amc_umcu_lumc_kul_test_uka_runmc/train/