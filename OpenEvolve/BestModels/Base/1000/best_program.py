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
        self.values = None  # learned values for each option
    
    def set_params(self, num_options, params_dict):
        """
        This function is called during parameter optimization per experiment, and should update the model's parameters based on the input dictionary.
        """
        self.num_options = num_options

        # extract parameters
        self.alpha = params_dict["alpha"]      # learning rate
        self.beta = params_dict["beta"]        # inverse temperature for softmax
        self.decay = params_dict["decay"]      # value decay to prevent saturation

    def participant_reset(self):
        # Reset all participant-level latent state variables here, this is called whenever a new participant begins.
        self.values = None

    def game_reset(self, game, horizon, hazard_rate):
        # Reset or update game-specific latent variables here, this is called whenever a new game begins.
        # Initialize values for all options to zero
        self.values = np.zeros(self.num_options)
        # Normalize hazard rate to [0, 1] for use in learning rate modulation
        self.hazard_factor = hazard_rate / 10.0
        # Reset trial counter for exploration decay
        self.trial_in_game = 0

    def predict(self, game, trial, horizon, hazard_rate, forced):
        # Returns raw action logits of shape (num_options,), do NOT apply softmax or convert to probabilities.
        if forced:
            # For forced trials, return neutral logits (will be ignored in NLL anyway)
            return [0.0] * self.num_options
        else:
            # Compute exploration bonus that decays with trial number
            # More exploration early in games, especially with high hazard rate
            exploration_bonus = self.beta * (1.0 / (1.0 + self.trial_in_game)) * (1.0 + self.hazard_factor)
            
            # Add exploration to values as logits
            logits = self.values + exploration_bonus
            return logits.tolist()

    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        # update state based on trial feedback
        # Increment trial counter for exploration decay
        self.trial_in_game += 1
        
        if not forced and self.values is not None:
            # Q-learning update with hazard-adaptive learning rate
            prediction_error = r_points - self.values[h_choice]
            
            # Higher hazard rate → faster learning (adapt to potential changes)
            adaptive_alpha = self.alpha * (1.0 + self.hazard_factor * 0.5)
            self.values[h_choice] += adaptive_alpha * prediction_error
            
            # Apply decay to all values to prevent saturation
            # Higher hazard → faster decay (more forgetting of outdated information)
            decay_rate = self.decay * (1.0 + self.hazard_factor)
            self.values = self.values * (1.0 - decay_rate)
            
        elif forced and self.values is not None:
            # On forced trials, still learn but with reduced learning rate
            # Participants learn from instructions but may not fully engage
            prediction_error = r_points - self.values[h_choice]
            learning_rate_on_forced = self.alpha * 0.3 * (1.0 + self.hazard_factor * 0.5)
            self.values[h_choice] += learning_rate_on_forced * prediction_error


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.

    Returns:
    param_bounds_dictionary: Dictionary of parameter names and their (min, max) bounds for optimization
    """

    # define parameters and their bounds
    return {
        'alpha': (0.01, 1.0),    # learning rate
        'beta': (0.1, 10.0),     # inverse temperature for choice stochasticity
        'decay': (0.0, 0.5)      # value decay rate
    }

def get_init_param(experiment):
    """
    This function is called at the start of optimization to set the initial parameters of the model.
    Note, these initial parameters are matched against the bounds dictionary.
    An error will be raised if any additional parameters are added or some are unused for a particular experiment.
    """
    default_params = {'alpha': 0.2, 'beta': 1.0, 'decay': 0.1}
    
    # set initial parameters values depending on the experiment
    match experiment:
        case 1:
            return {'alpha': 0.3, 'beta': 1.0, 'decay': 0.05}
        case 2:
            return {'alpha': 0.3, 'beta': 1.0, 'decay': 0.05}
        case 3:
            return {'alpha': 0.2, 'beta': 1.5, 'decay': 0.1}
        case 4:
            return {'alpha': 0.2, 'beta': 1.0, 'decay': 0.1}
        case 5:
            return {'alpha': 0.2, 'beta': 1.0, 'decay': 0.1}
        case _:
            return default_params

# EVOLVE-BLOCK END