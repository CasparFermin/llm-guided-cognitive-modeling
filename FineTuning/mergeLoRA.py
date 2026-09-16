import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


optimized_threads = 96
os.environ["OMP_NUM_THREADS"] = str(optimized_threads)
os.environ["MKL_NUM_THREADS"] = str(optimized_threads)
torch.set_num_threads(optimized_threads)
torch.set_num_interop_threads(2)

# ---------- PATHS ----------
base_model_path = "/gpfs/scratch1/shared/cfermin/Models/Qwen3-Coder-Next"
lora_adapter_path = "/gpfs/scratch1/shared/cfermin/Data/FineTune/checkpoint-40/"
merged_output_path = "/gpfs/scratch1/shared/cfermin/Models/Qwen3-Coder-Next-Merged"


# ---------- LOAD TOKENIZER ----------
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True,
)

# ---------- LOAD BASE MODEL ----------
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",   # safer for merge
    local_files_only=True,
)

# ---------- LOAD LORA ----------
model = PeftModel.from_pretrained(
    base_model,
    lora_adapter_path,
    torch_dtype=torch.bfloat16,
)

print("Loaded PEFT adapter.")

# ---------- MERGE LORA INTO BASE MODEL ----------
model = model.merge_and_unload()

print("Merged adapter into base model.")

# ---------- SAVE MERGED MODEL ----------
model.save_pretrained(
    merged_output_path,
    safe_serialization=True,
    max_shard_size="10GB"  # Optimal for Snellius Lustre/Staging filesystems
)

tokenizer.save_pretrained(merged_output_path)

print(f"Merged model saved to: {merged_output_path}")