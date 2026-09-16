# EVOLVE-BLOCK START
import numpy as np


class Model:
    def __init__(self):
        """Initialize the model with basic structures."""
        pass
    
    def set_params(self, num_options, params_dict):
        """Set model parameters for optimization."""
        self.num_options = num_options
        
        # Extract parameters - simplified to 6 core parameters
        self.alpha = params_dict["alpha"]      # base learning rate
        self.beta = params_dict["beta"]        # decision temperature
        self.init_value = params_dict["init_value"]  # initial value estimate
        self.hazard_sensitivity = params_dict.get("hazard_sensitivity", 0.0)  # sensitivity to hazard rate
        self.choice_bias = params_dict.get("choice_bias", 0.0)  # bias toward first option
        self.trial_decay = params_dict.get("trial_decay", 0.0)  # how much temperature decreases with trial number
    
    def participant_reset(self):
        """Reset participant-level state variables."""
        # Initialize value estimates for all options
        self.value_estimates = np.full(self.num_options, self.init_value)
        self.option_counts = np.zeros(self.num_options)  # track how often each option is chosen
    
    def game_reset(self, game, horizon, hazard_rate):
        """Reset or update game-specific variables."""
        # Reset value estimates at the start of each game
        self.participant_reset()
        self.game_hazard_rate = hazard_rate
        self.trials_in_game = 0
    
    def predict(self, game, trial, horizon, hazard_rate, forced):
        """Return raw logits for each option."""
        # For forced trials, return 0 for all options (will be ignored in NLL)
        if forced:
            return [0] * self.num_options
        
        # Adjust learning rate based on hazard rate - higher hazard = more weight to recent outcomes
        adjusted_alpha = self.alpha * (1 + self.hazard_sensitivity * hazard_rate / 10.0)
        
        # Temperature decreases over trials (more exploitation as game progresses)
        trial_factor = 1.0 - self.trial_decay * min(trial / max(horizon, 1), 1.0)
        effective_beta = self.beta * trial_factor
        
        # Compute logits based on value estimates with choice bias and exploration
        logits = []
        for i in range(self.num_options):
            # Add exploration bonus for less-chosen options (simplified)
            exploration_bonus = 0.1 * np.sqrt(trial + 1) / (self.option_counts[i] + 1)
            
            # Add choice bias for first option in 2-option games
            bias = self.choice_bias if i == 0 and self.num_options == 2 else 0.0
            
            # Combine value, exploration, and bias components
            logit = self.value_estimates[i] * effective_beta + exploration_bonus + bias
            logits.append(logit)
        
        return logits
    
    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        """Update value estimates based on trial feedback."""
        # Only update on free choices (forced trials don't inform learning)
        if forced:
            return
        
        # Adjust learning rate based on hazard rate
        adjusted_alpha = self.alpha * (1 + self.hazard_sensitivity * hazard_rate / 10.0)
        
        # Update value estimate using Rescorla-Wagner rule with adjusted learning rate
        prediction_error = r_points - self.value_estimates[h_choice]
        self.value_estimates[h_choice] += adjusted_alpha * prediction_error
        
        # Update option counts for exploration bonus
        self.option_counts[h_choice] += 1
        self.trials_in_game += 1


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.
    """
    return {
        'alpha': (0.01, 1.0),        # base learning rate: how quickly values are updated
        'beta': (0.1, 5.0),          # decision temperature: higher = more deterministic choices
        'init_value': (-5.0, 5.0),   # initial value estimate for all options
        'hazard_sensitivity': (0.0, 2.0),  # sensitivity to hazard rate (higher = more adaptive learning)
        'choice_bias': (-2.0, 2.0),  # bias toward first option in 2-option games
        'trial_decay': (0.0, 0.8)    # how much temperature decreases as game progresses (more exploitation over time)
    }


def get_init_param(experiment):
    """
    Set initial parameter values depending on experiment.
    Different experiments may have different optimal starting points.
    """
    # Base parameters that should work reasonably well across all experiments
    base_params = {
        'alpha': 0.2,
        'beta': 1.0,
        'init_value': 0.0,
        'hazard_sensitivity': 0.0,
        'choice_bias': 0.0,
        'trial_decay': 0.2
    }
    
    # Experiment-specific adjustments
    match experiment:
        case 1:
            # Simple 2-armed bandit with one sure zero option
            # May benefit from learning rate modulation by hazard rate
            return {
                'alpha': 0.2,
                'beta': 1.0,
                'init_value': 0.0,
                'hazard_sensitivity': 0.3,
                'choice_bias': 0.5,  # slight bias toward first option
                'trial_decay': 0.3
            }
        case 2:
            # 2-armed bandit with both options varying
            # Need good sensitivity to hazard rate for adaptation
            return {
                'alpha': 0.25,
                'beta': 1.0,
                'init_value': 0.0,
                'hazard_sensitivity': 0.5,
                'choice_bias': 0.0,
                'trial_decay': 0.2
            }
        case 3:
            # 4-armed bandit - need more exploration initially
            # Higher temperature for exploration, moderate trial decay
            return {
                'alpha': 0.35,
                'beta': 0.8,
                'init_value': 0.0,
                'hazard_sensitivity': 0.2,
                'choice_bias': 0.0,
                'trial_decay': 0.4
            }
        case 4:
            # Free-choice after forced trials - need to account for instruction trials
            # Higher sensitivity to hazard rate for adapting learning
            return {
                'alpha': 0.25,
                'beta': 1.0,
                'init_value': 0.5,
                'hazard_sensitivity': 0.6,
                'choice_bias': 0.2,
                'trial_decay': 0.5
            }
        case 5:
            # Similar to 4 but different trial structure (1-6 free trials)
            # Need more forgetting as trials are shorter
            return {
                'alpha': 0.2,
                'beta': 1.2,
                'init_value': 0.3,
                'hazard_sensitivity': 0.7,
                'choice_bias': 0.1,
                'trial_decay': 0.6
            }
        case _:
            return base_params

# EVOLVE-BLOCK END