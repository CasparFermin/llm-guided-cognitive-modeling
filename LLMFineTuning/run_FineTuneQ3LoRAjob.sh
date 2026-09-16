#!/bin/bash
#SBATCH --job-name=FineTuneLoRAQ3
#SBATCH --partition=gpu_h100        # request H100 partition
#SBATCH --gres=gpu:3                # request 3 GPUs
#SBATCH --time=3:00:00              # runtime
#SBATCH --output=results/%x_%j.log  # log file
#SBATCH --error=error/%x_%j.txt     # error file

# load module
module load 2025 Python/3.13.1-GCCcore-14.2.0 NVHPC/25.3-CUDA-12.8.0

# load environemtn
source FT/bin/activate

export HF_ENABLE_PARALLEL_LOADING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run python file
python LLMFineTuning/FineTuneQ3LoRA.py