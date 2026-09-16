import torch
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig#, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
import pandas as pd
from datasets import Dataset, concatenate_datasets
from collections import defaultdict, Counter
import re

# ------- Create dataset from dataframe, filled based on text with features -------
def get_dataset(df):

    # extract Features from the data and specifically text for each participant
    features = []
    for _, row in df.iterrows():
        text = row['text'].strip()

        # extract all answers in <<...>> for this participant
        choices = re.findall(r'<<(.*?)>>', text)

        # extract the rest of the sentence after 'labeled'
        list_segment = re.search(r"labeled\s+([^.]+)", text).group(1)

        # get all capital letters
        options = re.findall(r"\b[A-Z]\b", list_segment)

        # define mapping of character to number (e.g., 'E':0, 'M':1)
        choice_arm_mapping = {opt: i for i, opt in enumerate(options)}

        # tokenize the letters
        option_ids = [tokenizer(f"<<{opt}", add_special_tokens=False).input_ids[1] for opt in options]

        # define mapping of token_id to number
        option_token_mapping = {opt: i for i, opt in enumerate(option_ids)}

        # transform choices into 0 and 1
        bin_choices = [choice_arm_mapping[c] for c in choices]

        # tokenize the whole text
        tokenized = tokenizer(text, add_special_tokens=False, truncation=False)

        #  get the input_ids and attention mask
        input_ids = tokenized["input_ids"]

        # create default labels, -100 is don't compute nll over it
        completion_mask = [0] * len(input_ids)

        # set the token id in the labels where there are human choices
        for pos, token in enumerate(input_ids):
            if token in option_ids:
                completion_mask[pos] = 1

        # create the features
        features.append({
            # model data
            "participant": row['participant'],
            "input_ids": input_ids,
            "completion_mask": completion_mask,

            # metadata
            "option_ids": option_ids,
            "option_mapping": option_token_mapping,
            "human_choices": bin_choices
        })

    # transform features to a dataframe
    features = pd.DataFrame(features)

    # convert to HF Dataset
    dataset = Dataset.from_pandas(features[["input_ids", "completion_mask"]])

    return dataset

# ---------- DEFINE CONFIGURATIONS ----------
model_path = "/gpfs/scratch1/shared/cfermin/Models/Qwen3-Coder-Next"
data_path = "/gpfs/scratch1/shared/cfermin/Data/"
max_seq_length = 17555 # Adjust based on your needs
seed=3407

# ---------- DEFINE SFT, LORA CONFIG ----------
sft_config = SFTConfig(
    # ---- Data Preprocessing
    packing = True,
    padding_free = True,
    dataset_text_field=None,
    max_length = max_seq_length,
    shuffle_dataset=True,
    seed=seed,
    bf16=True,
    # ---- Regiment
    per_device_train_batch_size = 1,  # one batch at a time to avoid OOM
    per_device_eval_batch_size = 1,   # same for evaluation
    gradient_accumulation_steps = 8, # set to 8 after debugging
    gradient_checkpointing=True,
    warmup_steps = 2,              # each step is 8 forward & backward, and 1 optimizer call
    num_train_epochs = 3,
    # ----- Kernel  
    prediction_loss_only = True,
    # ----- Optimizer
    optim = "paged_adamw_8bit", 
    learning_rate = 2e-4,
    # --- Logging
    logging_strategy="steps",
    logging_steps=1,
    report_to="none",
    include_num_input_tokens_seen=True,    # produces error somehow
    # --- Evaluation
    eval_strategy="steps",
    eval_steps=10,                 # evaluate every 100 steps
    # ---- Saving
    save_strategy="steps",
    save_steps=10,
    save_total_limit=8,
    output_dir=f"{data_path}FineTune/",
)

# ---------- LOAD TOKENIZER & MODEL & APPLY CONFIG ----------
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

apply_liger_kernel_to_qwen3_moe(
    rope=True,
    swiglu=True,
    cross_entropy=False,
    fused_linear_cross_entropy=True,
    rms_norm=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="kernels-community/flash-attn3",    # Use flash-attention for speed and avoiding cross-attention
)

# ---------- LOAD DATA ----------
experiments = [
    {'name': 'HSo', 'experiment': 'exp0', 'split': 'Train'},
    {'name': 'HW', 'experiment': 'exp0', 'split': 'Train'},
    {'name': 'TB', 'experiment': 'exp1', 'split': 'Train'},
    {'name': 'TB', 'experiment': 'exp2', 'split': 'Train'},
    {'name': 'DB', 'experiment': 'exp0', 'split': 'Train'},
    # {'name': 'HF', 'experiment': 'exp0', 'split': 'OOD'},
    # {'name': 'HSa', 'experiment': 'exp0', 'split': 'OOD'},
    # {'name': 'CB', 'experiment': 'exp0', 'split': 'OOD'},
    # {'name': 'MF', 'experiment': 'exp0', 'split': 'OOD'},
]

train_dats = []
eval_dats = []
for exp in experiments:

    train = pd.read_csv(f"{data_path}{exp['name']}/Train_text_{exp['experiment']}.csv")
    eval = pd.read_csv(f"{data_path}{exp['name']}/Val_text_{exp['experiment']}.csv")

    # get the dataset for this experiment
    train_dats.append(get_dataset(train))
    eval_dats.append(get_dataset(eval))

train_data = concatenate_datasets(train_dats)
eval_data = concatenate_datasets(eval_dats)

# ----- BASELINE EVALUATION PERFORMANCE -----
baseline_trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=sft_config
)

baseline_metrics = baseline_trainer.evaluate()
pd.DataFrame([baseline_metrics]).to_csv(f"{data_path}FineTune/baseline_metrics.csv", index=False)

# ------- DEFINE LORA -------
# load config
config = AutoConfig.from_pretrained(model_path)

# define standard rank, and compute appropriate rank for each expert to avoid LoRA bloat
base_r = 16
num_experts = getattr(config, "num_local_experts", None) or config.num_experts
effective_r = max(1, base_r // num_experts)
print("effective_r:",effective_r)
lora_config = LoraConfig(
    r=base_r,
    lora_alpha=32,
    target_modules=[
        # There are 48 Layers in Qwen3-Coder-Next 80B
        # --- Classical Self-Attention, only appear 12 times ---
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # --- Linear Attention/DeltaNet Modules, appears 32 times ---
        "in_proj_qkvz",                   # Catches the fused QKVP linear attention
        "in_proj_ba",                     # Catches the secondary linear attention projection
        "out_proj",                       # Catches the attention output projections
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    target_parameters=[
        "gate_up_proj",
        "down_proj",
    ],
    rank_pattern={ # Chokes rank to 1 safely
        r".*mlp\.experts\.down_proj": effective_r,
        r".*mlp\.experts\.gate_up_proj": effective_r,
    },
    alpha_pattern={ # Ensure alpha is twice the size of effective_r
        r".*mlp\.experts\.down_proj": effective_r * 2,
        r".*mlp\.experts\.gate_up_proj": effective_r * 2,
    },
    bias="none",
    task_type="CAUSAL_LM"
)

# merge model
model = get_peft_model(model, lora_config)

# ------ CHECK IF TRAINABLE PARAMETERS IS PROPERLY SET (SHOULD BE 1% -  LOW NUMBER)
model.print_trainable_parameters()

# ---------- CREATE SFTTRAINER and TRAIN ----------
trainer = SFTTrainer(
    model = model,
    train_dataset = train_data,
    eval_dataset = eval_data,
    args = sft_config
)

# start training process
trainer.train(resume_from_checkpoint=f"{data_path}FineTune/checkpoint-40")

# Run final evaluation on last LoRA state and store it
trainer.evaluate()
model.save_pretrained(f"{data_path}FineTuned/Final_LoRA")

# store log history
pd.DataFrame(trainer.state.log_history).to_csv(f"{data_path}FineTune/SFTTrainer_logs.csv", index=False)

