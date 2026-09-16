import importlib.util
import numpy as np
from openevolve.evaluation_result import EvaluationResult
import scipy
import pandas as pd
from scipy.optimize import minimize, OptimizeWarning
import warnings
import traceback
import sys
import ast

# ------ Load Data Globally ------
experiments = [
    {'name': 'TB', 'experiment': 'exp1', 'num_options': 2},
    {'name': 'TB', 'experiment': 'exp2', 'num_options': 2},
    {'name': 'DB', 'experiment': 'exp0', 'num_options': 4},
    {'name': 'HSo', 'experiment': 'exp0', 'num_options': 2},
    {'name': 'HW', 'experiment': 'exp0', 'num_options': 2}
] # different names are used here to ensure the LLM cannot get the names of the experiments

data_path = "Path/to/your/data/"  # Update this path to your actual data directory
prog_path = "Path/to/your/programs/"  # Update this path to your actual programs directory

TRAIN_DATA = [np.load(f"{data_path}{exp['name']}/struc_Train_{exp['experiment']}.npy") for exp in experiments]


def compute_source_complexity(program_path):
    """Compute a simple AST-based complexity score for the source file."""
    with open(program_path, "r", encoding="utf-8", errors="replace") as source_file:
        source = source_file.read()

    tree = ast.parse(source, filename=program_path)
    complexity = sum(1 for _ in ast.walk(tree))
    return complexity

def evalModel(x, data, num_op, param_keys, model):
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

    # compute nll
    nll = np.mean(scipy.special.logsumexp(npall_logits, axis = 1) - npall_logits[row_indices, choices])

    # set best_nll and best_params as global so they are consistent across functions
    global best_nll, best_params

    # store best nll and params
    if nll < best_nll:
        best_nll = nll
        best_params = params_dict

    return nll

def run_model(program_path, max_eval):
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
    
    # initialize numpy to consider all warnings as errors.
    np.seterr(all='raise')
    warnings.simplefilter("error", OptimizeWarning)

    # get parameter bounds, the keys, and number of parameters
    param_bounds = program.define_parameters_and_bounds()
    param_keys = list(param_bounds.keys())
    parameter_count = len(param_bounds)

    # compute model complexity
    model_complexity = float(source_complexity + (parameter_count * 150))

    # construct the model
    model = program.Model()

    # loop over experiments and store results
    results = []
    for i, exp in enumerate(experiments):

        # set best_nll and best_params as global so they are consistent across functions
        global best_nll, best_params
        best_nll = np.inf
        best_params = {}

        # get data for this experiment
        train_data = TRAIN_DATA[i]

        # get number of options
        num_options = exp['num_options']

        # catch SciPy and Numpy warnings
        try:
            
            # extract parameters from the helper function defined by the LLM
            init_param_dict = program.get_init_param(i+1)

            # raise an error if there is a mismatch between the keys in the bounds and the keys in the initial parameters
            if param_keys != list(init_param_dict.keys()):
                raise ValueError(f"Mismatch between parameter keys in bounds and initial values"
                                 f"Bounds keys: {list(param_bounds.keys())}, Initial values keys: {list(init_param_dict.keys())}")

            # get initial parameter values in the correct order for optimization
            param_init = np.array([init_param_dict[k] for k in param_keys], dtype=np.float64)

            # run optimization of trainable parameters
            minimize(
                evalModel,
                x0=param_init,
                args=(train_data, num_options, param_keys, model),
                options={'maxfev': max_eval},
                method='Powell',
                bounds=list(param_bounds.values())
            )

        # catch all errors (and warnings raised as errors)
        except (FloatingPointError, OptimizeWarning, Exception) as e:
            # extract traceback
            _, _, tb = sys.exc_info()

            # get the last call, for exact location of the error
            last_call = traceback.extract_tb(tb)[-1]
            
            # create format of the error, and return as tuple
            error_type = type(e).__name__
            error_message = e
            error_location =  f"line {last_call.lineno}, in {last_call.name}: {last_call.line}"
            return {'Experiment': i+1, 'error_type': error_type, 'error_message': str(error_message), 'error_location':error_location}

        # store all the data in a list of dictionaries
        results.append({"Experiment": i+1, "combined_score": np.exp(-best_nll), "nll": best_nll, "model_complexity": model_complexity, "parameters": best_params})
            
    # return the results as a dataframe
    return pd.DataFrame(results)

# Stage-based evaluation for cascade evaluation
def evaluate(program_path, max_eval=20, stage=2):
    """First stage evaluation with fewer trials"""

    try:
        # Run a single trial with timeout
        result = run_model(program_path, max_eval)

        # catch caused by wrong implementations
        if isinstance(result, dict):

            # create a single error message
            error_summary = f"[Exp {result['Experiment']}] {result['error_type']}: {result['error_message']} @ {result['error_location']}"

            error_artifacts = {
                'errors:': error_summary,
                "suggestion": ("Ensure all required methods of the class Model are exist, namely: __init__, set_params, participant_reset, game_reset, predict, update."
                               "Ensure define_parameters_and_bounds and get_init_param exists, and that all these functions return the correct format."
                               "Also ensure math is handled correctly, and that bounds and initial guesses are properly set for the trainable parameters."
                               )
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": result['error_message']
                },
                artifacts=error_artifacts
            )

        # Handle different result formats
        if isinstance(result, pd.DataFrame):
            if result.shape[1] == 5:
                pass
            else:
                error_artifacts = {
                    "error_type": "InvalidReturnFormat",
                    "error_message": f"Stage {stage}: Invalid result format, expected got n-results: {result.shape[1]}",
                    "suggestion": "Ensure predict returns the same amount of logits as num_options."
                }
                
                return EvaluationResult(
                    metrics={
                        "runs_successfully": 0.0, 
                        "combined_score": 0.0,
                        "error": "Invalid result format"
                    },
                    artifacts=error_artifacts
                )
        else:
            # print(f"Stage {stage}: Invalid result format, expected a dictionary (experiment, combined_score, nll, model_complexity, parameters) but got: {type(result)}")
            
            error_artifacts = {
                "error_type": "InvalidReturnType",
                "error_message": f"Stage {stage}: Function returned {type(result)}, expected dictionary (experiment, combined_score, nll, model_complexity, parameters)",
                "suggestion": "run_model() must return a dictionary with the items (experiment, combined_score, nll, model_complexity, parameters)."
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "Invalid result format"
                },
                artifacts=error_artifacts
            )

        # extract data from the results            
        nll_df = result[['Experiment', 'nll']]

        # Check if the result is valid
        if (
            any(np.isnan(nll_df['nll']))
            or any(np.isinf(nll_df['nll']))
        ):
            # print(f"Stage {stage} validation: Invalid result: {nll_df.to_string()}")
            
            error_artifacts = {
                "error_type": "InvalidResultValues",
                "error_message": f"Stage {stage}: Got invalid values: {nll_df.to_string()}",
                "suggestion": "Function returned NaN or infinite values. Check for division by zero, invalid math operations, or uninitialized variables"
            }
            
            return EvaluationResult(
                metrics={
                    "runs_successfully": 0.1, 
                    "combined_score": 0.0,
                    "error": "Invalid result values"
                },
                artifacts=error_artifacts
            )
        
        # compute mean combined score
        best_stats = result[['combined_score', 'nll', 'model_complexity']].mean().to_dict()
        result.drop(columns=['model_complexity'], inplace=True)
        
        # Add artifacts for successful stage 1
        evaluation_artifacts = {
            "Experiment History\n": result.to_dict(orient='records')
        }

        return EvaluationResult(
            metrics={
                "runs_successfully": 1.0,
                "combined_score": best_stats['combined_score'],
                "nll": best_stats['nll'],
                "model_complexity": best_stats['model_complexity']
            },
            artifacts=evaluation_artifacts
        )
    except TimeoutError as e:
        # print(f"Stage {stage} evaluation timed out: {e}")
        
        error_artifacts = {
            "error_type": "TimeoutError",
            "error_message": "Stage {stage}: Function execution exceeded 1000 second timeout",
            "suggestion": "Function is likely stuck in infinite loop or doing too much computation. Try reducing iterations or adding early termination conditions"
        }
        
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": "Timeout"
            },
            artifacts=error_artifacts
        )
    except IndexError as e:
        # Specifically handle IndexError which often happens with early termination checks
        # print(f"Stage {stage} evaluation failed with IndexError: {e}")
        # print("This is likely due to a list index check before the list is fully populated.")
        
        error_artifacts = {
            "error_type": "IndexError",
            "error_message": f"Stage {stage}: {str(e)}",
            "suggestion": "List index out of range - likely accessing empty list or wrong index. Check list initialization and bounds"
        }
        
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": f"IndexError: {str(e)}"
            },
            artifacts=error_artifacts
        )
    except Exception as e:
        # print(f"Stage {stage} evaluation failed: {e}")
        # print(traceback.format_exc())
        
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": f"Stage {stage}: {str(e)}",
            "full_traceback": traceback.format_exc(),
            "suggestion": "Unexpected error occurred. Check the traceback for specific issue"
        }
        
        return EvaluationResult(
            metrics={
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": str(e)
            },
            artifacts=error_artifacts
        )

# define the cascade evaluation functions with differing amounts of times to evaluate the program for (due to non-determinism of minimize)
def evaluate_stage1(program_path):
    return evaluate(program_path, 3, 1)

def evaluate_stage2(program_path):
    return evaluate(program_path, 20, 2)

if __name__ == "__main__":
    print(evaluate_stage2(f"{prog_path}initial_program.py")) # just for testing, the other print statements are also purely for testing purposes