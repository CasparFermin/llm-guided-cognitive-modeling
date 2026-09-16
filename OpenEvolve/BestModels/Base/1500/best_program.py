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
        self.values = None  # Value estimates for each option
        self.trial_count = 0  # Track trial count for exploration
        self.last_choice = None  # Track last choice for persistence bias
        self.in_instructed_phase = False  # Track instruction phase for experiments 4 & 5
    
    def set_params(self, num_options, params_dict):
        """
        This function is called during parameter optimization per experiment, and should update the model's parameters based on the input dictionary.
        """
        self.num_options = num_options

        # extract parameters
        self.alpha = params_dict["alpha"]      # learning rate
        self.beta = params_dict["beta"]        # inverse temperature for softmax (exploitation/exploration)
        self.init_val = params_dict["init_val"]  # initial value estimate
        self.hazard_sens = params_dict.get("hazard_sens", 0.0)  # hazard rate sensitivity
        self.instr_decay = params_dict.get("instr_decay", 0.0)  # decay during instruction phase

    def participant_reset(self):
        # Reset all participant-level latent state variables here, this is called whenever a new participant begins.
        # Initialize value estimates for each option
        self.values = [self.init_val] * self.num_options
        self.trial_count = 0
        self.last_choice = None
        self.in_instructed_phase = False

    def game_reset(self, game, horizon, hazard_rate):
        # Reset or update game-specific latent variables here, this is called whenever a new game begins.
        # Reset values at start of each game (since new slot machines each game)
        self.values = [self.init_val] * self.num_options
        self.trial_count = 0
        self.last_choice = None
        self.in_instructed_phase = False
        self.instructed_trials_remaining = 0
    
    def _compute_exploration_bonus(self, game, trial, horizon, hazard_rate):
        """Compute trial-dependent exploration bonus with hazard rate adaptation"""
        if horizon > 0:
            # More exploration early in game, but modulate by hazard rate
            trial_progress = min(1.0, trial / max(1, horizon))
            
            # Base exploration that decays with trial progress
            base_exploration = 0.3 * (1.0 - trial_progress)
            
            # Modulate by hazard rate: higher hazard → more exploration needed due to instability
            if self.in_instructed_phase and game in [4, 5]:
                # During instruction phase, reduce exploration significantly
                hazard_modulation = 0.5 + 0.5 * (1.0 - trial_progress) * self.hazard_sens * hazard_rate / 10.0
            else:
                # Free-choice phase: more exploration for higher hazard rates
                hazard_modulation = 1.0 + 0.5 * self.hazard_sens * hazard_rate / 10.0
                
            return base_exploration * hazard_modulation
        else:
            return 0.3  # Default exploration when horizon unknown

    def predict(self, game, trial, horizon, hazard_rate, forced):
        # Returns raw action logits of shape (num_options,), do NOT apply softmax or convert to probabilities.
        if forced:
            # For forced trials, return constant logits (participants have no choice)
            return [0.0] * self.num_options
        else:
            # Use values with exploration bonus to create logits
            exploration_bonus = self._compute_exploration_bonus(game, trial, horizon, hazard_rate)
            
            # Compute logits with persistence bias for last choice
            logits = []
            for i, v in enumerate(self.values):
                logit = self.beta * v + exploration_bonus
                
                # Add persistence bias for last choice (especially important for stable environments)
                # Only add bias for free-choice trials, not during instruction phase
                if self.last_choice is not None and i == self.last_choice and not self.in_instructed_phase:
                    logit += 0.5 * (1.0 + self.hazard_sens * hazard_rate / 10.0)
                    
                logits.append(logit)
            
            return logits

    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        # update state based on trial feedback
        
        # Track transition from instructed to free-choice phase in experiments 4 & 5
        if game in [4, 5] and trial == 1:
            self.in_instructed_phase = True
            self.instructed_trials_remaining = 4
        elif self.in_instructed_phase:
            self.instructed_trials_remaining -= 1
            if self.instructed_trials_remaining <= 0:
                self.in_instructed_phase = False
        
        # Skip learning on forced trials in experiments 4/5 initial instructed trials
        if forced and game in [4, 5] and trial <= 4:
            self.last_choice = h_choice  # Still track last choice for bias
            return
            
        # Adjust learning rate based on hazard rate - higher hazard means more learning
        adjusted_alpha = self.alpha
        if self.hazard_sens > 0 and hazard_rate > 0:
            # Higher hazard rate → faster updating
            adjusted_alpha = min(1.0, self.alpha * (1.0 + self.hazard_sens * hazard_rate / 10.0))
        
        # Q-learning style update: V(s) = V(s) + alpha * (reward - V(s))
        old_value = self.values[h_choice]
        prediction_error = r_points - old_value
        self.values[h_choice] = old_value + adjusted_alpha * prediction_error
        
        # Apply small decay during instruction phase to prevent value drift
        if self.in_instructed_phase and self.instr_decay > 0:
            self.values = [v * (1 - self.instr_decay) for v in self.values]
        
        # Update last choice for persistence bias
        self.last_choice = h_choice
        self.trial_count += 1


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.

    Returns:
    param_bounds_dictionary: Dictionary of parameter names and their (min, max) bounds for optimization
    """

    # define parameters and their bounds - reduced from 5 to 4 parameters
    return {
        'alpha': (0.01, 1.0),      # learning rate: how quickly values are updated
        'beta': (0.01, 10.0),      # inverse temperature: determines choice determinism
        'init_val': (-5.0, 5.0),   # initial value estimate
        'hazard_sens': (0.0, 0.5),  # sensitivity to hazard rate for adaptive learning
        'instr_decay': (0.0, 0.2)   # value decay during instruction phase
    }

def get_init_param(experiment):
    """
    This function is called at the start of optimization to set the initial parameters of the model.
    Note, these initial parameters are matched against the bounds dictionary.
    An error will be raised if any additional parameters are added or some are unused for a particular experiment.
    """
    
    # set initial parameters values depending on the experiment
    # Base values that work well across most reinforcement learning tasks
    match experiment:
        case 1:
            # Simple 2-option with one always zero: moderate learning, high exploitation
            return {'alpha': 0.3, 'beta': 3.0, 'init_val': 0.0, 'hazard_sens': 0.0, 'instr_decay': 0.0}
        case 2:
            # Simple 2-option with both varying: similar to exp1 but slightly more exploration
            return {'alpha': 0.25, 'beta': 3.0, 'init_val': 0.0, 'hazard_sens': 0.0, 'instr_decay': 0.0}
        case 3:
            # 4 options: higher exploration to encourage discovery
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'hazard_sens': 0.1, 'instr_decay': 0.05}
        case 4:
            # Instructed trials first: moderate learning with instruction decay to prevent drift
            return {'alpha': 0.3, 'beta': 1.5, 'init_val': 0.0, 'hazard_sens': 0.1, 'instr_decay': 0.1}
        case 5:
            # Similar to exp4 but potentially more exploration needed
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'hazard_sens': 0.15, 'instr_decay': 0.05}
        case _:
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'hazard_sens': 0.1, 'instr_decay': 0.05}

# EVOLVE-BLOCK END