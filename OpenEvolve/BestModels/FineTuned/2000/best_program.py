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
        self.alpha = params_dict["alpha"]        # base learning rate
        self.beta = params_dict["beta"]          # decision temperature
        self.init_value = params_dict["init_value"]  # initial value estimate
        self.eps = params_dict.get("epsilon", 0.0)   # exploration bonus for less-chosen options
        self.hazard_sens = params_dict.get("hazard_sensitivity", 0.0)  # sensitivity to hazard rate
        self.choice_bias = params_dict.get("choice_bias", 0.0)  # bias toward first option in 2-option games
        self.trial_decay = params_dict.get("trial_decay", 0.0)  # how much temperature decreases with trials
    
    def participant_reset(self):
        # Reset all participant-level latent state variables here, this is called whenever a new participant begins.
        # Initialize value estimates for all options and track choice frequencies
        self.value_estimates = np.full(self.num_options, self.init_value)
        self.choice_counts = np.zeros(self.num_options)

    def game_reset(self, game, horizon, hazard_rate):
        # Reset or update game-specific latent variables here, this is called whenever a new game begins.
        # Reset value estimates and choice counts at the start of each game
        self.participant_reset()
        self.game_hazard_rate = hazard_rate
        self.trials_in_game = 0
        self.last_choice = None  # Track previous choice for sticky bias
        self.instruction_phase = False  # Track if in forced instruction phase (exp 4&5)

    def predict(self, game, trial, horizon, hazard_rate, forced):
        # Returns raw action logits of shape (num_options,), do NOT apply softmax or convert to probabilities.
        # For forced trials, return 0 for all options (will be ignored in NLL)
        if forced:
            return [0] * self.num_options
        
        # Track if in instruction phase (games 4&5, trials 1-4)
        self.instruction_phase = (game >= 4 and trial <= 4)
        
        # Adjust temperature based on trial progression (more exploitation as game progresses)
        trial_progress = min(trial / max(horizon, 1), 1.0)
        trial_factor = 1.0 - self.trial_decay * trial_progress
        effective_beta = self.beta * max(0.1, trial_factor)
        
        # Adjust temperature based on hazard rate - higher hazard may need more exploration
        hazard_factor = 1.0 + self.hazard_sens * hazard_rate / 15.0
        effective_beta *= hazard_factor
        
        # Normalize value estimates to prevent extreme logits
        value_range = np.max(self.value_estimates) - np.min(self.value_estimates)
        value_scale = min(1.0, 5.0 / max(value_range, 0.1))
        
        # Compute logits as value estimates with temperature scaling and exploration bonus
        logits = []
        for i in range(self.num_options):
            # Scale value estimates by decision temperature parameter and normalization
            value_component = self.value_estimates[i] * effective_beta * value_scale
            
            # Add exploration bonus based on how rarely this option has been chosen
            # Use exponential decay for more realistic exploration patterns
            base_exploration = self.eps * np.exp(-0.1 * trial)
            hazard_exploration = base_exploration * (1 + hazard_rate / 25.0) if hazard_rate else base_exploration
            exploration_bonus = hazard_exploration / (self.choice_counts[i] + 1)
            
            # Add choice stability bonus - prefer previously chosen options
            # This captures human tendency to stick with familiar choices
            choice_stability = 0.12 * effective_beta
            if self.choice_counts[i] > 0:
                stability_bonus = choice_stability * np.log(self.choice_counts[i] + 1)
            else:
                stability_bonus = 0.0
            
            # Add choice persistence - tendency to repeat previous choice
            if self.last_choice is not None and i == self.last_choice:
                persistence_bonus = 0.18 * effective_beta * (1 - trial_progress)
                stability_bonus += persistence_bonus
            
            # Add choice bias for first option in 2-option games (early trials)
            bias = 0.0
            if self.num_options == 2 and trial <= 3:
                bias = self.choice_bias if i == 0 else -self.choice_bias * 0.5
            
            # Adjust instruction trials to have more exploration and less stability
            if self.instruction_phase:
                stability_bonus *= 0.5  # Reduced stability during instruction phase
                exploration_bonus *= 0.8
            
            logit = value_component + exploration_bonus + stability_bonus + bias
            logits.append(logit)
        
        return logits

    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        # update state based on trial feedback
        
        # Determine learning dynamics based on trial type and experiment context
        if forced and game < 4:
            # For experiments 1-3, forced trials don't provide learning opportunity
            # Only update when participants make their own choice
            return
        
        # Adjust learning rate based on hazard rate and trial type
        # Instruction trials (forced in exp 4&5) have different learning dynamics
        if forced and self.instruction_phase:
            # Reduced learning during instruction trials, but still learn from outcomes
            # Participants may be more focused on learning the instruction value than following instructions
            adjusted_alpha = self.alpha * 0.4 * (1 + self.hazard_sens * self.game_hazard_rate / 10.0)
        else:
            # Free-choice trials and later forced trials use higher learning rates
            adjusted_alpha = self.alpha * (1 + self.hazard_sens * self.game_hazard_rate / 10.0)
        
        # Update value estimate using Rescorla-Wagner rule
        prediction_error = r_points - self.value_estimates[h_choice]
        self.value_estimates[h_choice] += adjusted_alpha * prediction_error
        
        # Apply value decay to unchosen options - stronger in high hazard environments
        # But instruction trials don't cause strong forgetting
        if not self.instruction_phase and self.game_hazard_rate > 2:
            decay_rate = 0.03 + 0.005 * self.game_hazard_rate
            for i in range(self.num_options):
                if i != h_choice:
                    self.value_estimates[i] *= (1 - decay_rate)
        else:
            # Minimal decay for instruction trials or low hazard environments
            for i in range(self.num_options):
                if i != h_choice:
                    self.value_estimates[i] *= 0.99
        
        # Update choice counts for exploration bonus
        # For forced trials in instruction phase, use smaller increments
        if not forced:
            self.choice_counts[h_choice] += 1
            self.trials_in_game += 1
            self.last_choice = h_choice  # Track for sticky bias
        elif forced and game >= 4:
            # Update choice counts for forced trials in instruction phase with smaller increments
            # This allows instruction trials to influence later free choices without overwhelming the model
            self.choice_counts[h_choice] += 0.3
            # Track instruction trial choices for potential consistency effects
            if not hasattr(self, 'instruction_choices'):
                self.instruction_choices = {}
            if game not in self.instruction_choices:
                self.instruction_choices[game] = {}
            self.instruction_choices[game][trial] = h_choice


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.

    Returns:
    param_bounds_dictionary: Dictionary of parameter names and their (min, max) bounds for optimization
    """

    # Define parameters with meaningful interpretations:
    # - alpha: base learning rate (0.01 to 1) for updating value estimates
    # - beta: decision temperature (0.1 to 5) - higher = more deterministic choices
    # - init_value: initial expected value for all options (-5 to 5)
    # - eps: exploration bonus (0 to 1) - incentives choosing less-frequent options
    # - hazard_sensitivity: sensitivity to hazard rate (0 to 2) - higher = more focus on recent outcomes
    # - choice_bias: bias toward first option in 2-option games (-2 to 2)
    # - trial_decay: how much temperature decreases across trials (0 to 0.8)
    
    return {
        'alpha': (0.01, 1.0),           # base learning rate
        'beta': (0.1, 5.0),             # decision temperature
        'init_value': (-5.0, 5.0),      # initial value estimate
        'eps': (0.0, 1.0),              # exploration bonus parameter
        'hazard_sensitivity': (0.0, 2.0),  # sensitivity to hazard rate
        'choice_bias': (-2.0, 2.0),     # bias toward first option in 2-option games
        'trial_decay': (0.0, 0.8)       # temperature decay across trials
    }

def get_init_param(experiment):
    """
    This function is called at the start of optimization to set the initial parameters of the model.
    Note, these initial parameters are matched against the bounds dictionary.
    """
    # Base parameters that should work reasonably well across all experiments
    base_params = {
        'alpha': 0.2, 
        'beta': 1.0, 
        'init_value': 0.0,
        'eps': 0.0,
        'hazard_sensitivity': 0.0,
        'choice_bias': 0.0,
        'trial_decay': 0.2
    }
    
    # Experiment-specific parameter adjustments based on task characteristics
    experiment_params = {
        1: {
            'alpha': 0.25, 
            'beta': 1.0, 
            'init_value': 0.0,
            'eps': 0.0,  # No need for exploration bonus in simple 2-option task
            'hazard_sensitivity': 0.4,  # Some sensitivity to hazard rate
            'choice_bias': 0.5,  # Bias toward first option in early trials
            'trial_decay': 0.3
        },
        2: {
            'alpha': 0.3, 
            'beta': 0.9, 
            'init_value': 0.0,
            'eps': 0.0,  # Both options vary, focus on value differences
            'hazard_sensitivity': 0.5,  # Good sensitivity to hazard rate
            'choice_bias': 0.0,
            'trial_decay': 0.3
        },
        3: {
            'alpha': 0.35, 
            'beta': 0.8, 
            'init_value': 0.0,
            'eps': 0.6,  # Need exploration bonus for 4 options
            'hazard_sensitivity': 0.2,
            'choice_bias': 0.0,
            'trial_decay': 0.4
        },
        4: {
            'alpha': 0.3, 
            'beta': 0.7, 
            'init_value': 0.0,
            'eps': 0.2,  # Some exploration needed after forced trials
            'hazard_sensitivity': 0.6,  # High sensitivity for adapting to changing environments
            'choice_bias': 0.3,
            'trial_decay': 0.5
        },
        5: {
            'alpha': 0.25, 
            'beta': 0.7, 
            'init_value': 0.0,
            'eps': 0.3,  # More exploration for longer free-choice periods
            'hazard_sensitivity': 0.7,  # High sensitivity to adapt to changing environments
            'choice_bias': 0.2,
            'trial_decay': 0.5
        }
    }
    
    if experiment in experiment_params:
        return experiment_params[experiment]
    else:
        return base_params

# EVOLVE-BLOCK END