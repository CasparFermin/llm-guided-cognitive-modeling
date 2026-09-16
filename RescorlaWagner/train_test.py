from CogModel import RescorlaWagnerModel
from trainer import Trainer
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Gets the absolute path to the file currently running
script_dir = Path(__file__).resolve().parent
results_dir = Path(f"{script_dir}/../Results/RescorlaWagnerResults/")

# create main results folders 
Path(f"{results_dir}").mkdir(parents=True, exist_ok=True)

def save_loss_curve(losses, save_path="loss_curve.png", window=None):
    """
    Save a graph of loss over training iterations.

    Parameters
    ----------
    losses : list or np.ndarray
        Sequence of mean NLL values.
    save_path : str
        Output image path.
    window : int or None
        Optional moving-average smoothing window.
    """

    losses = np.asarray(losses, dtype=float)

    plt.figure(figsize=(10, 5))

    # raw losses
    plt.plot(losses, label="Mean NLL")

    # optional smoothing
    if window is not None and window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(losses, kernel, mode="valid")

        smooth_x = np.arange(window - 1, len(losses))
        plt.plot(smooth_x, smooth, label=f"Moving Avg ({window})")

    plt.xlabel("Iteration")
    plt.ylabel("Mean NLL")
    plt.title("Training Loss Over Iterations")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

experiments = [
    {'name': 'TwoBandit', 'experiment': 'exp1', 'model': RescorlaWagnerModel(num_options=2)},
    {'name': 'TwoBandit', 'experiment': 'exp2', 'model': RescorlaWagnerModel(num_options=2)},
    {'name': 'DriftingBandit', 'experiment': 'exp0', 'model': RescorlaWagnerModel(num_options=4)},
    {'name': 'HorizonFeng', 'experiment': 'exp0', 'model': RescorlaWagnerModel(num_options=2)},
    {'name': 'HorizonWaltz', 'experiment': 'exp0', 'model': RescorlaWagnerModel(num_options=2)},
    {'name': 'ChangingBandit', 'experiment': 'exp0','model': RescorlaWagnerModel(num_options=2)},
    {'name': 'HorizonSomer', 'experiment': 'exp0', 'model': RescorlaWagnerModel(num_options=2)},
    {'name': 'HorizonSade', 'experiment': 'exp0', 'model': RescorlaWagnerModel(num_options=2)},
    {'name': 'MaggiesFarm', 'experiment': 'exp0', 'model': RescorlaWagnerModel(num_options=3)},
]

tot_data = []
col_names = ['participant', 'game', 'horizon', 'trial', 'forced', 'choice', 'reward', 'hazard_rate']

for exp in experiments:

    # read the data
    train_np = np.load(f"{script_dir}/../Data/PreProcData/{exp['name']}/proc_Train_{exp['experiment']}.npy")
    test_np = np.load(f"{script_dir}/../Data/PreProcData/{exp['name']}/proc_Test_{exp['experiment']}.npy")

    # transform from numpy to dataframes
    train_df = pd.DataFrame(train_np, columns=col_names)
    test_df = pd.DataFrame(test_np, columns=col_names)

    # construct the trainer, based on the model
    trainer = Trainer(exp['model'])

    print("Fitting experiment:", exp['name'])

    # apply fit and evaluate to get nll of both datasets
    train_nlls, test_nll, logits = trainer.fit_and_evaluate(train_df, test_df)
    tot_data.append([exp['name'], exp['experiment'], np.mean(train_nlls), test_nll])

    # Optionally create and save image of loss over training iterations
    # save_loss_curve(train_nlls, save_path=f"Images/LossCurves/RescolaWagner{exp['name']}_{exp['experiment']}.png")

    # remove the rows where the answers are forced, and thus have no logits
    test_df = test_df[test_df['forced'] == 0]

    # get probabilities of the model picking the right answer
    probs = torch.softmax(logits, dim=1)
    choices_tensor = torch.tensor(test_df['choice'].values)

    # get the probability that the model is correct
    prob_correct = probs[torch.arange(len(choices_tensor), dtype=torch.long), choices_tensor.long()]

    # create dictionary of logits for dynamic handling
    log_data = {f"ResWagLog_{log_op}": logits[:, log_op] for log_op in range(logits.shape[1])}

    # store logits and per-trial data for each experiment
    inv_df = pd.DataFrame({'participant': test_df['participant'], 'game': test_df['game'], 'horizon': test_df['horizon'],  'trial': test_df['trial'], 'reward': test_df['reward'], 
                        'human_choice': test_df['choice'], 'EvolvedMod': prob_correct, **log_data})
    inv_df.to_csv(f'{results_dir}/{exp["name"]}_{exp["experiment"]}.csv', index=False)

df = pd.DataFrame(tot_data, columns=['name', 'experiment', 'train_nll', 'test_nll'])
print(df)
print(df['test_nll'].mean())
df.to_csv(f'{results_dir}/summary.csv', index=False)