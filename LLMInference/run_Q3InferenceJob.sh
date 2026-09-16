#!/bin/bash
#SBATCH --job-name=LLMInference
#SBATCH --partition=gpu_h100        # request H100 partition
#SBATCH --gres=gpu:3                # request 3 GPUs
#SBATCH --time=3:00:00              # runtime
#SBATCH --output=results/%x_%j.log  # log file
#SBATCH --error=error/%x_%j.txt     # error file

# load module
module load 2025 Python/3.13.1-GCCcore-14.2.0 NVHPC/25.3-CUDA-12.8.0

# load environment
source FT/bin/activate

export HF_ENABLE_PARALLEL_LOADING=1

# run the inference script, use 1 for fine-tuned model (merged), 0 for base model
python  LLMInference/LLMPredict.py "1"