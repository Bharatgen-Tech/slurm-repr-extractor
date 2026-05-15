#!/bin/bash
#SBATCH --job-name=iasr_infer
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --nodelist=ip-10-20-160-224,ip-10-20-164-92,ip-10-20-179-184,ip-10-20-198-82
#SBATCH --output=logs1/%x_%j.out
#SBATCH --error=logs1/%x_%j.err

set -euo pipefail

mkdir -p logs1

# ---- Conda activation (CORRECT WAY) ----
source /home/ashutosh.adhikari/miniconda3/etc/profile.d/conda.sh
conda activate onnx_runner 

export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH

# ---- NCCL / Torch settings ----
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# ---- Launch DDP ----
srun python /fsx/ashutosh.adhikari/representations/scripts/recompute_representations_slurm_logging.py

