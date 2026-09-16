import importlib.util
import numpy as np
from openevolve.evaluation_result import EvaluationResult
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.optimize import minimize
import ast
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ----- Load data globally so that it can be accessed by all processes -----
# Gets the absolute path to the file currently running
script_dir = Path(__file__).resolve().parent
results_dir = f"{script_dir}/../Results/EvolvedCogModels/"

# create main results folder
Path(results_dir).mkdir(parents=True, exist_ok=True)

experiments = [
    {'name': 'TwoBandit', 'experiment': 'exp1', 'num_options': 2},
    {'name': 'TwoBandit', 'experiment': 'exp2', 'num_options': 2},
    {'name': 'DriftingBandit', 'experiment': 'exp0', 'num_options': 4},
    {'name': 'HorizonSomer', 'experiment': 'exp0', 'num_options': 2},
    {'name': 'HorizonWaltz', 'experiment': 'exp0', 'num_options': 2},
    {'name': 'HorizonSade', 'experiment': 'exp0', 'num_options': 2},
    {'name': 'HorizonFeng', 'experiment': 'exp0', 'num_options': 2},
    {'name': 'ChangingBandit', 'experiment': 'exp0', 'num_options': 2},
    {'name': 'MaggiesFarm', 'experiment': 'exp0', 'num_options': 3},
]

data_path = f"{script_dir}/../Data/"

EXPERIMENT_DATA = []

for exp in experiments:
    train_data = np.load(f"{data_path}{exp['name']}/proc_Train_{exp['experiment']}.npy")
    test_data = np.load(f"{data_path}{exp['name']}/proc_Test_{exp['experiment']}.npy")

    EXPERIMENT_DATA.append({
        "train": train_data,
        "test": test_data
    })
    
# ----- Helper functions for model evaluation -----

def compute_source_complexity(program_path):
    """Compute a simple AST-based complexity score for the source file."""
    with open(program_path, "r", encoding="utf-8", errors="replace") as source_file:
        source = source_file.read()

    tree = ast.parse(source, filename=program_path)
    complexity = sum(1 for _ in ast.walk(tree))
    return complexity

def save_loss_curve(loss_history, save_path="loss_curve.png", window=None):
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

    losses = np.asarray(loss_history, dtype=float)

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

# ---- Model Runner -----
def evalModel(x, data, num_op, param_keys, model, full_logits = False):
    """
    Evaluate the model. First, load in the data. Second, create the model. Third, loop over each trial, and apply initilize, when a new participant appears.
    
    All experiments contain the data columns:
    participant:    int32   - The participant ID
    game:           int32   - The number of the game, starts at 1
    horizon:        int32   - The number of trials per game, -1 if not known by the participant
    trial:          int32   - The trial number, resets every game and starts at 1
    forced:         int32   - Either 0 or 1, with 1 representing a non-trial, which won't be taken into account during computation of nll
    human_choice:   int32   - The human choice that has to be modelled
    reward:         int32   - The reward as a consequence of the human_choice
    hazard_rate:    int32   - Indicates the degree of abrupt expected point change, it ranges from 0-10, with 1 representing a 10% change
    """
    # create the parameter dictionary from the static keys and dynamic values
    params_dict = dict(zip(param_keys, x))

    # create bare-bones model object
    model.set_params(num_op, params_dict)

    nll = 0.0
    prev_participant = None
    prev_game = None

    # store logits
    all_logits = []

    all_participant = data[:, 0]
    all_game = data[:, 1]
    all_horizon = data[:, 2]
    all_trial = data[:, 3]
    all_forced = data[:, 4]
    all_human_choice = data[:, 5]
    all_reward = data[:, 6]
    all_hazard_rate = data[:, 7]

    # loop over each row (i.e., trial) in the data
    for i in range(len(data)):

        # extract the data from the row
        participant = all_participant[i]
        game = all_game[i]
        horizon = all_horizon[i]
        trial = all_trial[i]
        forced = all_forced[i]
        human_choice = all_human_choice[i]
        reward = all_reward[i]
        hazard_rate = all_hazard_rate[i]

        # Initialize model state per participant
        if participant != prev_participant:
            model.participant_reset()
            prev_participant = participant
            prev_game = None

        # Reset game-specific variables
        if game != prev_game:
            model.game_reset(game, horizon, hazard_rate)
            prev_game = game

        # Predict action probabilities
        logits = model.predict(game, trial, horizon, hazard_rate, forced)

        if forced == 0:
            # store logits for vectorized computations later
            all_logits.append(logits)

        # Update model based on human choice and reward
        model.update(game, trial, horizon, hazard_rate, forced, human_choice, reward)

    # transform list of logits to numpy array
    npall_logits = np.array(all_logits)

    # prepare for slicing the logits
    choices = data[data[:, 4] == 0, 5].astype(int)
    row_indices = np.arange(len(choices))

    # get safe logits to prevent under/overflow in logsumexp
    safe_logits = np.clip(npall_logits, -10000, 10000)

    # compute nll
    nll = np.mean(scipy.special.logsumexp(safe_logits, axis = 1) - safe_logits[row_indices, choices])

    # set best_nll and best_params as global so they are consistent across functions
    global best_nll, best_params

    if full_logits:
        return nll, safe_logits
    else: 
        loss_history.append(nll)
        if nll < best_nll:
            best_nll = nll
            best_params = params_dict

        return nll

# ---- Model Orchestrator -----
def run_model(program_path, max_eval, model_path):
    """
    This function is called once for each iteration by OpenEvolve, which includes:
    1. training the model by optimizing the adjustable parameters.
    2. returning feedback metrics.
    """

    # load program
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # get source_complexity based on model AST for combined_score penalization
    source_complexity = compute_source_complexity(program_path)

    # get parameter bounds, the keys, and number of parameters
    param_bounds = program.define_parameters_and_bounds()
    param_keys = list(param_bounds.keys())
    parameter_count = len(param_bounds)

    # compute model complexity
    # model_complexity = 2e-9 * (source_complexity ** 2) + 2e-4 * (parameter_count ** 2)
    model_complexity = float(source_complexity + (parameter_count * 150))

    # construct the model
    model = program.Model()

    # initialize arrays for train and test nlls
    train_nlls = []
    test_nlls = []

    # loop over experiments and store results
    results = []
    for i, exp in enumerate(experiments):

        start_time = time.time()

        # set best_nll and best_params as global so they are consistent across functions
        global best_nll, best_params
        best_nll = np.inf
        best_params = {}

        # get data for this experiment
        train_data = EXPERIMENT_DATA[i]["train"]
        test_data = EXPERIMENT_DATA[i]["test"]

        # get number of options
        num_options = exp['num_options']

        # extract parameters from the helper function defined by the LLM
        init_param_dict = program.get_init_param(i+1)

        # raise an error if there is a mismatch between the keys in the bounds and the keys in the initial parameters
        if set(param_keys) != set(init_param_dict.keys()):
            raise ValueError(f"Mismatch between parameter keys in bounds and initial values. "
                            f"Bounds keys: {set(param_bounds.keys())}, Initial values keys: {set(init_param_dict.keys())}")
        
        # get initial parameter values in the correct order for optimization
        param_init = np.array([init_param_dict[k] for k in param_keys], dtype=np.float64)
            
        global loss_history
        loss_history = []

        # run optimization of trainable parameters
        result = minimize(
            evalModel,
            x0=param_init,
            args=(train_data, num_options, param_keys, model, False),
            options={'maxfev': max_eval},
            method='Powell',
            bounds=list(param_bounds.values())
        )
        nEval = result.nfev

        # print("Best loss:", min(loss_history), flush=True)
        # save_loss_curve(loss_history, save_path=f"Models/{model_path}/Train_loss_curve_{exp["name"]}_{exp["experiment"]}.png", window=None)
        loss_history = []

        # evaluate test_data with the best parameters found during training, and get the logits for further analysis
        nll, all_logits = evalModel(list(best_params.values()), test_data, num_options, param_keys, model, True)
        test_nlls.append(nll)
        train_nlls.append(best_nll)

        # compute combined_score
        comb_score = np.exp(-best_nll)

        # remove the rows that have 'forced' in them
        test_data = test_data[test_data[:, 4] != 1]
        
        # get probabilities of the model picking the right answer
        human_choices = test_data[:, 5].astype(int)
        row_indices = np.arange(len(human_choices))
        prob_correct = np.exp(all_logits[row_indices, human_choices] - scipy.special.logsumexp(all_logits, axis=1))

        # Create a dictionary where i is the arm index
        log_data = {f"EvCogModLog_{log_op}": all_logits[:, log_op] for log_op in range(all_logits.shape[1])}

        # store results
        pd.DataFrame({'participant': test_data[:, 0], 'game': test_data[:, 1], 
                      'horizon': test_data[:, 2], 'trial': test_data[:, 3], 
                      'reward': test_data[:, 6], 'human_choice': human_choices, 
                      'HCProb': prob_correct, **log_data
                      }).to_csv(f'{results_dir}{model_path}/{exp['name']}_{exp['experiment']}.csv', index=False)

        print(f"Experiment: {exp['name']}, Combined_score: {comb_score}, train-nll: {best_nll}, test-nll: {nll}, n_eval: {nEval}, model_complexity: {model_complexity}")

        # store all the data in a list of dictionaries
        results.append({"Experiment": i+1, "combined_score": comb_score, "nll": best_nll, "model_complexity": model_complexity, "parameters": best_params})
        print("Time For Model Evaluate:", time.time() - start_time)
            
    # store total results
    tot_data = [[experiments[i]['name'], experiments[i]['experiment'], train_nlls[i], test_nlls[i]] for i in range(len(train_nlls))]
    df = pd.DataFrame(tot_data, columns=['name', 'experiment', 'train_nll', 'test_nll'])
    print(df)
    print(df['test_nll'].mean())
    df.to_csv(f'{results_dir}{model_path}/Results_summary.csv', index=False)

    # return the results as a dataframe
    return pd.DataFrame(results)

def full_evaluate(model_path):
    program_path = f"{script_dir}/BestModels/{model_path}/best_program.py"
    return run_model(program_path, 100, model_path) # minimize with 100 different parameter evaluations

if __name__ == "__main__":
    # define model paths to evaluate
    models = ["Base/1000", "Base/1500", "Base/2000", "Base/Best", "FineTuned/1000", "FineTuned/1500", "FineTuned/2000", "FineTuned/Best"]

    with ProcessPoolExecutor(max_workers=8) as executor:
    # executor.map automatically runs evaluate_single_model for each model in parallel
        results = list(executor.map(full_evaluate, models))
    
    print(results)
    
