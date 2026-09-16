from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
import torch
import pandas as pd
import re
from datasets import Dataset
import sys
import numpy as np
import time


# ------- Costum Preprocess_Logits function -------
def preprocess_logits_for_metrics(logits, labels, option_ids):
    """
    Reduces logits[batch, seq, vocab_size] to logits[batch, trials, 2 + num_options] to avoid transferring a really large tensor from the GPU to the CPU.
    The main purpose of this function is preprocessing the logits on the GPU with tensors instead of the CPU with np.arrays.
    """
    # shift logits and labels
    logits = logits[0, :-1, :]
    labels = labels[0, 1:]

    # precreate the mask
    mask = labels != -100

    # filter only positions that matter, as only trial logits are relevant (reduce sequence)
    trial_logits = logits[mask]  # [num_valid_positions, vocab_size]
    trial_labels = labels[mask]  # [num_valid_positions]

    # delete logits already to avoid keeping full vocabulary in VRAM
    del logits

    # Compute LSE over full vocabulary
    lse = torch.logsumexp(trial_logits, dim=-1, keepdim=True)  # [num_valid_positions, 1]

    # extract valid token IDs for this participant (reduce vocabulary)
    option_ids_array = option_ids.view(1, -1).expand(trial_logits.shape[0], -1)
    
    # gather logits for these two options
    f_logits = trial_logits.gather(dim=-1, index=option_ids_array) # [num_valid_positions, n_options]
    
    # identify target logits and compute nll
    target_logits = trial_logits.gather(dim=-1, index=trial_labels.unsqueeze(-1))
    nll = -(target_logits - lse)


    # return option_logits and LSE, and append
    return torch.cat([lse, nll, f_logits], dim=-1).float().detach().cpu()  # [seq, 2+n_options]

# ------- Create dataset from dataframe, filled based on text with features -------
def get_dataset(exp, data_path):
    # load the data
    df = pd.read_csv(f"{data_path}{exp['name']}/Test_text_{exp['experiment']}.csv")

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
        labels = [-100] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # set the token id in the labels where there are human choices
        for pos, token in enumerate(input_ids):
            if token in option_ids:
                labels[pos] = token

        # create the features
        features.append({
            # model data
            "participant": row['participant'],
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,

            # metadata
            "option_ids": option_ids,
            "option_mapping": option_token_mapping,
            "human_choices": bin_choices
        })

    # transform features to a dataframe
    features = pd.DataFrame(features)

    # convert to HF Dataset
    dataset = Dataset.from_pandas(features[["participant", "input_ids", "labels", "attention_mask", "option_ids"]])

    return dataset, features

# ------- Run the SFTTrainer Predict for each experiment -------
if __name__=="__main__":
    FineTuned = "FineTuned" if len(sys.argv) > 1 and sys.argv[1] == "1" else "Base"
    print(f"Running Inference on the {FineTuned} Model!")

    # ------- Load Model, Tokenizer, and Create SFTConfig -------
    model_path = "/gpfs/scratch1/shared/cfermin/Models/Qwen3-Coder-Next-Merged"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    # define experiments
    experiments = [
        {'name': 'TB', 'experiment': 'exp1', 'split': 'Train'},
        {'name': 'DB', 'experiment': 'exp0', 'split': 'Train'},
        {'name': 'MF', 'experiment': 'exp0', 'split': 'OOD'},
        {'name': 'HSo', 'experiment': 'exp0', 'split': 'Train'},
        {'name': 'HW', 'experiment': 'exp0', 'split': 'Train'},
        {'name': 'TB', 'experiment': 'exp2', 'split': 'Train'},
        {'name': 'HF', 'experiment': 'exp0', 'split': 'OOD'},
        {'name': 'HSa', 'experiment': 'exp0', 'split': 'OOD'},
        {'name': 'CB', 'experiment': 'exp0', 'split': 'OOD'},
    ]
    data_path = "/gpfs/scratch1/shared/cfermin/Data/"

    apply_liger_kernel_to_qwen3_moe(
        rope=True,
        swiglu=True,
        cross_entropy=False,              # False to be able to extract logits
        fused_linear_cross_entropy=False, # False to be able to extract logits
        rms_norm=True,                    
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="kernels-community/flash-attn3",  # saves GPU memory
        local_files_only=True,
    )
    model.eval()

    # loop for each entry in experiments
    tot_results = []
    for exp in experiments:

        # get the dataset for this experiment
        dataset, features = get_dataset(exp, data_path)
        total_tokens = sum(len(x["input_ids"]) for x in dataset)
        n_participants = features.shape[0]

        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=default_data_collator
        )

        print(f"Experiment: {exp['name'] + exp['experiment']}", flush=True)
        print(f"Number of iterations (i.e., participants): {n_participants}", flush=True)

        # storage
        results = []

        start_time = time.time()

        # -------- INFERENCE BLOCK --------
        with torch.no_grad():

            # loop over participants
            for batch in dataloader:

                # forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                # preprocess logits ON GPU
                BatchResults = preprocess_logits_for_metrics(
                    outputs.logits,
                    batch["labels"],
                    batch["option_ids"]
                )

                # store results
                results.append({
                    "participant": batch['participant'].item(),
                    "lse": BatchResults[:, 0],
                    "nll": BatchResults[:, 1],
                    "logits": BatchResults[:, 2:],
                    "labels": batch['labels'].cpu()
                })

                # aggressively clear references
                del outputs
                del BatchResults

        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed
        print(f"Finished Inference for Experiment: {exp['name']}_{exp['experiment']}\nTotal Time: {elapsed}, with tokens/s: {tokens_per_sec}", flush=True)

        # transform to dataframe
        res_df = pd.DataFrame(results)

        total_nll = np.concatenate([r["nll"] for r in results]).mean()

        # ------- Store Results Summary and Trial-Based Data -------
        # store results summary
        tot_results.append([exp['name'], exp['experiment'], n_participants,  (time.time() - start_time), total_nll])

        data_proc_time = time.time()
        # store a single file for each experiment containing the individual logits
        logits_rows = []
        for res in results:

            # get the results of the current participant
            participant = res["participant"]
            lse_array = res["lse"]          # [T]
            nll_array = res["nll"]          # [T]
            gpu_logits = res["logits"]      # [T, K]
            labs = res["labels"]          # [T]

            # get the features per participant
            par_features = features[features['participant'] == participant].iloc[0]
            human_choices = par_features["human_choices"]

            # Create a dictionary where i is the arm index
            log_data = [{f"LLMLog_{log_op}": gpu_logits[trial, log_op] for log_op in range(gpu_logits.shape[1])} for trial in range(len(human_choices))]

            # get participant results
            for trial in range(len(human_choices)):
                logits_rows.append({
                    "participant": participant,
                    "trial": trial + 1,
                    "human_choice": human_choices[trial],
                    **log_data[trial],
                    "lse": lse_array[trial],
                    "nll": nll_array[trial]
                })

        # store trial-based results
        logits_df = pd.DataFrame(logits_rows)
        logits_df.to_csv(f"{data_path}Results/{FineTuned}/{exp['name']}_{exp['experiment']}.csv", index=False)
        print("Data Process Time:", time.time() - data_proc_time)
    # store total results
    pd.DataFrame(tot_results, columns=['name', 'experiment', 'n-par', 'test_time', 'test_nll']).to_csv(f"{data_path}Results/{FineTuned}/LLMPredict_summary.csv", index=False)
