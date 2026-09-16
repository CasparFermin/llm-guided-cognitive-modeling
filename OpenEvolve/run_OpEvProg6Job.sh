#!/bin/bash
#SBATCH --job-name=OpEvProg6
#SBATCH --partition=gpu_h100        # request H100 GPU
#SBATCH --gres=gpu:2                # request 2 GPU's
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # request 32 cpus per task (only one call to openevolve, which utilises multiple process pools)
#SBATCH --time=3:35:00              # short runtime for testing
#SBATCH --output=/gpfs/scratch1/shared/cfermin/Logs/results_%x_%j.log  # log file
#SBATCH --error=/gpfs/scratch1/shared/cfermin/Logs/errors_%x_%j.txt     # error file
#SBATCH --export=ALL

# load necessary modules
module load 2025 Python/3.13.1-GCCcore-14.2.0 NVHPC/25.3-CUDA-12.8.0

source FT/bin/activate

# copy model from scratch-shared to the the faster scratch-node
# export MODEL_DIR=$TMPDIR/Qwen3-Coder-Next-Merged
# time cp -r /gpfs/scratch1/shared/cfermin/Models/Qwen3-Coder-Next-Merged "$MODEL_DIR/"

# echo "Copy finished. Forcing disk sync..."
# time sleep 5

# # 1. Create a workspace on the local scratch-node
# export WORK_DIR="/gpfs/scratch1/shared/cfermin/workspace"
# mkdir -p "$WORK_DIR"

# set the OpenAI API Key
# export OPENAI_API_KEY="AIzaSyDdEpY5VDhglwLxk9p_5S30Z6dV9Ai4j-M"
export OPENAI_API_KEY="sk-no-key"

# export VLLM_CACHE_ROOT="$WORK_DIR/.vllm_cache"
# mkdir -p "$VLLM_CACHE_ROOT"

# cd "$WORK_DIR"

export MODEL_DIR="/gpfs/scratch1/shared/cfermin/Models/Qwen3-Coder-Next-Merged"

# python OpEvProg6/dir_change.py \
#   --directory "$MODEL_DIR" \
#   --config OpEvProg6/config.yaml

# create log file for vllm logs
export LOG_FILE="/gpfs/scratch1/shared/cfermin/Logs/vllm_${SLURM_JOB_ID}.log"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# host a local vllm server
vllm serve $MODEL_DIR \
  --port 11434 \
  --chat-template "$MODEL_DIR/chat_template.jinja" \
  --quantization fp8 \
  --max-model-len 131072 \
  --download-dir $MODEL_DIR \
  --tensor-parallel-size 2 \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait until server is ready
echo "Waiting for vLLM server to be ready..."
until curl -s localhost:11434/v1/models | grep -q "Qwen"; do
  sleep 15
done
echo "vLLM server ready!"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the openevolve test process
openevolve-run OpEvProg6/initial_program.py OpEvProg6/evaluator.py --config OpEvProg6/config.yaml --output /gpfs/scratch1/shared/cfermin/results/OpEvQ3/

# close server
kill $SERVER_PID
kill $OPTILLM_PID


# optillm \
#   --base-url http://localhost:11434/v1 \
#   --port 8000 \
#   --model "$MODEL_DIR" \
#   > optillm.log 2>&1 &

# OPTILLM_PID=$!
# echo "DONE WITH SETTING UP OPTILLM"