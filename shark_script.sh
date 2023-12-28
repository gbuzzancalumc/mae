#!/bin/bash
#SBATCH --job-name=mae-pretraining
#SBATCH --partition=PATHgpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --chdir=/exports/path-nefro-hpc/gbuzzanca_/MAE/
#SBATCH --output=/exports/path-nefro-hpc/gbuzzanca_/MAE/%x-%j.out
#SBATCH --error=/exports/path-nefro-hpc/gbuzzanca_/MAE/%x-%j.err

export PYTHONUNBUFFERED=TRUE
export NCCL_SOCKET_IFNAME=team0

module load library/cuda/11.6.1/gcc.8.3.1
module load library/cudnn/11.6/cudnn

source ~/.bashrc
conda activate /exports/path-nefro-hpc/gbuzzanca_/.conda_envs/mae

srun python submit_pretrain.py --job_dir ${JOB_DIR} --norm_pix_loss --data_path /export/path-nefro-hpc/gbuzzanca_/train_amc_umcu_lumc_kul_test_uka_runmc/train
