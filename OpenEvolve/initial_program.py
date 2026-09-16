# EVOLVE-BLOCK START
import numpy as np


class Model:
    # Avoid unnecessary randomness inside predict/update.
    # Prefer deterministic mechanisms unless stochasticity is essential.
    # Expensive random sampling significantly slows optimization.

    def __init__(self):
        """
        This initialization function is called once when the model is created. 
        It is only meant to avoid initialize the model object multiple times.
        """
        pass
    
    def set_params(self, num_options, params_dict):
        """
        This function is called during parameter optimization per experiment, and should update the model's parameters based on the input dictionary.
        """
        self.num_options = num_options

        # extract parameters
        self.alpha = params_dict["alpha"]      # learning rate

    def participant_reset(self):
        # Reset all participant-level latent state variables here, this is called whenever a new participant begins.
        pass

    def game_reset(self, game, horizon, hazard_rate):
        # Reset or update game-specific latent variables here, this is called whenever a new game begins.
        pass

    def predict(self, game, trial, horizon, hazard_rate, forced):
        # Returns raw action logits of shape (num_options,), do NOT apply softmax or convert to probabilities.
        logits = [self.alpha] * self.num_options
        return logits

    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        # update state based on trial feedback
        pass


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.

    Returns:
    param_bounds_dictionary: Dictionary of parameter names and their (min, max) bounds for optimization
    """

    # define prameters and their bounds
    return {
        'alpha': (0.1, 2.0)    # alpha: Some parameter
    }

def get_init_param(experiment):
    """
    This function is called at the start of optimization to set the initial parameters of the model.
    Note, these initial parameters are matched against the bounds dictionary.
    An error will be raised if any additional parameters are added or some are unused for a particular experiment.
    """
    default_params = {'alpha': 1}
    
    # set initial parameters values depending on the experiment
    match experiment:
        case 1:
            return {'alpha': 1}
        case 2:
            return {'alpha': 1}
        case 3:
            return {'alpha': 1}
        case 4:
            return {'alpha': 1}
        case 5:
            return {'alpha': 1}
        case _:
            return default_params

# EVOLVE-BLOCK END