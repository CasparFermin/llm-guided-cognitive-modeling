# EVOLVE-BLOCK START
import numpy as np


class Model:
    def __init__(self):
        """Initialize the model with basic structures."""
        pass
    
    def set_params(self, num_options, params_dict):
        """Set model parameters for optimization."""
        self.num_options = num_options
        
        # Extract parameters
        self.alpha = params_dict["alpha"]      # learning rate
        self.beta = params_dict["beta"]        # decision noise/temperature
        self.init_value = params_dict["init_value"]  # initial value estimate
    
    def participant_reset(self):
        """Reset participant-level state variables."""
        # Initialize value estimates for all options
        self.value_estimates = np.full(self.num_options, self.init_value)
    
    def game_reset(self, game, horizon, hazard_rate):
        """Reset or update game-specific variables."""
        # Reset value estimates at the start of each game
        self.participant_reset()
        
        # Track trial progression within game
        self.trial_in_game = 0
        self.game_horizon = horizon
        self.game_hazard_rate = hazard_rate
    
    def predict(self, game, trial, horizon, hazard_rate, forced):
        """Return raw logits for each option."""
        # For forced trials, return 0 for all options (will be ignored in NLL)
        if forced:
            return [0] * self.num_options
        
        # Compute logits as value estimates scaled by inverse temperature
        # Add exploration bonus that decreases with trial number and choice bias
        logits = []
        for i in range(self.num_options):
            # Base value estimate scaled by inverse temperature
            base_value = self.value_estimates[i] * self.beta
            
            # Add exploration bonus that decays with trial number
            # Use more gradual decay and experiment-specific scaling
            if self.num_options == 4:  # experiment 3
                # More exploration needed for 4-armed bandit
                exploration_bonus = 1.5 * np.exp(-0.08 * self.trial_in_game)
            else:  # experiments 1, 2, 4, 5
                exploration_bonus = 0.8 * np.exp(-0.15 * self.trial_in_game)
            
            logit = base_value + exploration_bonus
            
            # Add choice bias (prefer earlier options)
            if i < 2:  # prefer first two options
                logit += 0.1
            elif self.num_options == 4 and i == 2:  # prefer first half of 4 options
                logit += 0.05
            
            logits.append(logit)
        
        return logits
    
    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        """Update value estimates based on trial feedback."""
        # Update value estimates for both forced and free trials in experiments 4 & 5
        # In these experiments, participants observe outcomes even on forced trials
        
        # Determine if we should update based on trial type and experiment
        update_free = not forced
        update_forced = (game >= 4) and forced and (trial <= 4)  # forced trials in exp 4 & 5
        
        if not (update_free or update_forced):
            return
        
        # Update value estimate using Rescorla-Wagner rule
        prediction_error = r_points - self.value_estimates[h_choice]
        self.value_estimates[h_choice] += self.alpha * prediction_error
        
        # For experiments 4 & 5 with high hazard rates, add value decay to capture adaptation
        # High hazard rate indicates sudden changes, so we should discount old values
        if (self.game_hazard_rate is not None and self.game_hazard_rate >= 6 and 
            not forced and self.trial_in_game > 4):
            decay_factor = 1.0 - (self.game_hazard_rate / 25.0)  # 0.76 decay at hazard_rate=6
            for i in range(self.num_options):
                if i != h_choice:
                    self.value_estimates[i] *= decay_factor
        
        # Update trial counter
        self.trial_in_game += 1
    
    def get_params(self):
        """Get current parameters."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "init_value": self.init_value
        }


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.
    """
    return {
        'alpha': (0.01, 1.0),        # learning rate: how quickly values are updated
        'beta': (0.1, 5.0),          # decision temperature: higher = more deterministic choices
        'init_value': (-5.0, 5.0)    # initial value estimate for all options
    }


def get_init_param(experiment):
    """
    Set initial parameter values depending on experiment.
    Different experiments may have different optimal starting points.
    """
    match experiment:
        case 1:
            # Simple 2-armed bandit with one sure zero option
            return {'alpha': 0.2, 'beta': 1.0, 'init_value': 0.0}
        case 2:
            # 2-armed bandit with both options varying
            return {'alpha': 0.2, 'beta': 1.0, 'init_value': 0.0}
        case 3:
            # 4-armed bandit - need more exploration initially
            return {'alpha': 0.4, 'beta': 0.6, 'init_value': 1.0}
        case 4:
            # Free-choice after forced trials - need to account for instruction trials
            return {'alpha': 0.2, 'beta': 1.0, 'init_value': 0.0}
        case 5:
            # Similar to 4 but different trial structure
            return {'alpha': 0.2, 'beta': 1.0, 'init_value': 0.0}
        case _:
            return {'alpha': 0.2, 'beta': 1.0, 'init_value': 0.0}

# EVOLVE-BLOCK END