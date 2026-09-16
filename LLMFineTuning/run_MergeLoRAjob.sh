#!/bin/bash
#SBATCH --job-name=qwen3_merge
#SBATCH --partition=genoa                  # Your specified partition
#SBATCH --nodes=1                         
#SBATCH --exclusive                      # CRITICAL: Gives you the entire node and its full memory
#SBATCH --time=03:00:00
#SBATCH --output=result/merge_output_%j.log
#SBATCH --error=error/%x_%j.txt

# load module
module load 2025 Python/3.13.1-GCCcore-14.2.0 NVHPC/25.3-CUDA-12.8.0

# load environment
source FT/bin/activate

# run python file
python LLMFineTuning/mergeLoRA.py
