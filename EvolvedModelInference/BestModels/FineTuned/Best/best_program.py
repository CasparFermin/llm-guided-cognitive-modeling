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
        self.value_vars = None  # Uncertainty estimates for each option
        self.trial_count = 0  # Track trial count for exploration
        self.last_choice = None  # Track last choice for stickiness effect
        self.choice_history = []  # Track choice history for habit modeling
        self.rpe_history = []  # Track RPE history for volatility estimation
        self.last_hazard = 0.0  # Track previous hazard for volatility detection
        self.confidence = 0.6  # Track confidence for exploration modulation
        self.trial_in_game = 0  # Track trial number within current game
        self.value_range = (0, 10)  # Expected value range for normalization
    
    def set_params(self, num_options, params_dict):
        """
        This function is called during parameter optimization per experiment, and should update the model's parameters based on the input dictionary.
        """
        self.num_options = num_options

        # extract parameters
        self.alpha = params_dict["alpha"]      # learning rate
        self.beta = params_dict["beta"]        # inverse temperature for softmax (exploitation/exploration)
        self.init_val = params_dict["init_val"]  # initial value estimate
        self.stickiness = params_dict.get("stickiness", 0.0)  # tendency to repeat previous choice
        self.asymmetry = params_dict.get("asymmetry", 0.2)  # asymmetric learning weight (positive PE > negative PE)
        self.risk_sens = params_dict.get("risk_sens", 1.0)  # risk sensitivity parameter
        self.base_bias = params_dict.get("base_bias", 0.0)  # base bias for option preferences
        # Fixed hazard parameters for improved volatility detection
        self.hazard_explore_weight = 0.12  # Fixed weight for hazard-based exploration
        self.hazard_update_weight = 0.05   # Fixed weight for hazard-based learning rate scaling
        self.unchosen_uncertainty_growth = 0.04  # Fixed growth rate for unchosen options
        # Simplified forgetting mechanism
        self.forgetting_rate = 0.008       # Fixed value decay rate

    def participant_reset(self):
        # Reset all participant-level latent state variables here, this is called whenever a new participant begins.
        # Initialize value estimates for each option
        self.values = [self.init_val] * self.num_options
        # Initialize uncertainty estimates (higher initial uncertainty encourages exploration)
        self.value_vars = [1.0] * self.num_options
        self.trial_count = 0
        self.last_choice = None
        self.choice_history = []  # Track choice history for habit modeling
        self.rpe_history = []  # Track RPE history for volatility estimation
        self.last_hazard = 0.0  # Track previous hazard for volatility detection
        self.confidence = 0.6  # Track confidence for exploration modulation
        self.trial_in_game = 0  # Track trial number within current game

    def game_reset(self, game, horizon, hazard_rate):
        # Reset or update game-specific latent variables here, this is called whenever a new game begins.
        # Reset values at start of each game (since new slot machines each game)
        self.values = [self.init_val] * self.num_options
        
        # Reset uncertainty estimates for new game with experiment-appropriate initial values
        if game in [1, 2]:  # 2-option experiments
            self.value_vars = [0.65] * self.num_options
        elif game == 3:  # 4-option experiment
            self.value_vars = [1.0] * self.num_options
        else:  # Experiments 4/5
            self.value_vars = [0.8] * self.num_options
        
        self.trial_count = 0
        self.trial_in_game = 0  # Reset trial counter within game
        self.last_choice = None
        # Calibrate initial confidence based on hazard rate and experiment structure
        base_confidence = 0.6
        if game in [4, 5]:
            base_confidence = 0.5
        # Use smoother confidence calibration
        self.confidence = max(0.3, min(0.8, base_confidence - hazard_rate * 0.02))
        self.choice_history = []  # Reset choice history
        self.rpe_history = []  # Reset RPE history
        self.last_hazard = 0.0  # Reset hazard tracking for new game

    def predict(self, game, trial, horizon, hazard_rate, forced):
        # Returns raw action logits of shape (num_options,), do NOT apply softmax or convert to probabilities.
        if forced:
            # For forced trials, return neutral logits (will be ignored in NLL anyway)
            return [0.0] * self.num_options
        else:
            # Apply risk sensitivity to values before computing logits
            # Risk sensitivity: concave transformation for gains, convex for losses
            risk_adjusted_values = []
            for v in self.values:
                if v >= 0:
                    risk_adjusted_values.append(np.power(v + 1e-6, self.risk_sens))
                else:
                    risk_adjusted_values.append(-np.power(-v + 1e-6, self.risk_sens))
            
            # Normalize values to prevent drift (helps with generalization across experiments)
            value_range = self.value_range[1] - self.value_range[0]
            normalized_values = [(v - self.value_range[0]) / value_range for v in risk_adjusted_values]
            
            # Compute base logits from risk-adjusted values with beta scaling
            logits = [self.beta * v for v in normalized_values]
            
            # Add base bias for option preferences (e.g., position bias)
            for i in range(self.num_options):
                # Use normalized position bias that works well across different numbers of options
                position_bias = self.base_bias * (0.5 - i / (self.num_options - 1 + 1e-6))
                logits[i] += position_bias
            
            # Add exploration bonus based on uncertainty (more uncertainty → more exploration)
            uncertainty_bonus = [0.18 * np.sqrt(var) for var in self.value_vars]
            logits = [l + u for l, u in zip(logits, uncertainty_bonus)]
            
            # Add trial-dependent exploration bonus (more exploration early)
            # Use experiment-specific exploration decay rates for better generalization
            if horizon > 0:
                trial_progress = trial / max(1, horizon)
                # Base exploration decay rate with improved stability
                decay_rate = 0.8
                
                # Experiment-specific adjustments
                if game in [1, 2]:
                    decay_rate = 0.85  # Slower decay in simple 2-option tasks
                elif game == 3:
                    decay_rate = 0.75  # Faster decay with more options
                elif game in [4, 5]:
                    # Different decay for instruction vs free choice phases
                    decay_rate = 0.85 if trial <= 4 else 1.0
                
                exploration_bonus = 0.28 * np.exp(-decay_rate * trial_progress)
            else:
                exploration_bonus = 0.28
            
            logits = [l + exploration_bonus for l in logits]
            
            # Add stickiness for previous choice with improved inverted-U hazard response
            if self.last_choice is not None and self.stickiness != 0:
                # Use a smoother inverted-U function that peaks at moderate hazard (4-6)
                hazard_stickiness = min(1.0, 0.18 * hazard_rate * (10.0 - hazard_rate) / 25.0)
                trial_mod = 1.0 - min(1.0, trial / max(1, horizon)) if horizon > 0 else 0.0
                logits[self.last_choice] += self.stickiness * hazard_stickiness * trial_mod
            
            # Add habit component for repeated choices (more pronounced in stable environments)
            if len(self.choice_history) >= 2:
                # Count how many times the current option was chosen recently
                recent_choices = self.choice_history[-3:] if len(self.choice_history) >= 3 else self.choice_history
                choice_count = recent_choices.count(self.last_choice)
                # Add habit bonus that scales with choice frequency, reduced in volatile environments
                hazard_mod = max(0.7, 1.0 - hazard_rate/12.0)
                habit_bonus = self.stickiness * choice_count * hazard_mod
                logits[self.last_choice] += habit_bonus
            
            # Normalize logits to avoid extreme values for numerical stability
            max_logit = max(logits)
            logits = [l - max_logit for l in logits]
            
            # Add small baseline exploration to prevent premature convergence
            # Consistent baseline across all experiments
            baseline_exploration = 0.012
            logits = [l + baseline_exploration for l in logits]
            
            # Add confidence-modulated exploration with trial-based decay
            # Simplified confidence exploration function for better generalization
            if self.confidence < 0.75 and horizon > 0:
                # Direct confidence exploration without trial phase adjustments
                # Adjust exploration magnitude based on experiment structure
                exp_factor = 1.0 + 0.05 * (1 if game in [4, 5] else 0)
                confidence_exploration = 0.22 * (1.0 - self.confidence) * (1.0 - min(1.0, trial/max(1, horizon))) * exp_factor
                logits = [l + confidence_exploration for l in logits]
            
            return logits

    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        # update state based on trial feedback
        
        # Track last choice for stickiness effect (even on forced trials)
        self.last_choice = h_choice
        self.trial_in_game += 1  # Increment trial counter within game
        
        # Store choice for habit tracking
        self.choice_history.append(h_choice)
        # Keep only recent choices for habit calculation (last 3)
        if len(self.choice_history) > 3:
            self.choice_history.pop(0)
        
        # Skip learning on forced trials in experiments 4/5 initial instructed trials
        if forced and game in [4, 5] and trial <= 4:
            return
            
        # Adaptive learning rate based on hazard rate
        adjusted_alpha = self.alpha
        if hazard_rate > 0:
            # Use smoother hazard sensitivity with bounded effect
            hazard_factor = np.tanh(hazard_rate * 0.12)
            adjusted_alpha = min(1.0, self.alpha * (1.0 + 0.15 * hazard_factor))
        
        # Q-learning style update: V(s) = V(s) + alpha * (reward - V(s))
        old_value = self.values[h_choice]
        prediction_error = r_points - old_value
        self.values[h_choice] = old_value + adjusted_alpha * prediction_error
        
        # Add forgetting mechanism to prevent value saturation
        # Simplified forgetting that works across experiments
        for i in range(self.num_options):
            # Apply forgetting to all values
            self.values[i] *= (1 - 0.005)
        
        # Normalize values to prevent drift with experiment-specific scaling
        if len(self.values) > 0:
            # Store original mean for reference
            original_mean = np.mean(self.values)
            # Apply experiment-specific normalization with trial-dependent decay
            # More aggressive normalization in later trials to prevent drift
            trial_factor = min(1.0, trial / max(1, horizon)) if horizon > 0 else 0.5
            norm_strength = 0.08 * (0.5 + 0.5 * trial_factor)  # More normalization in later trials
            
            # Adjust normalization based on experiment structure
            if game in [4, 5] and trial > 4:
                norm_strength *= 1.2  # Stronger normalization after instruction phase
            
            self.values = [v - original_mean * norm_strength for v in self.values]
        
        # Update uncertainty estimates - reduce uncertainty after feedback
        self.value_vars[h_choice] = max(0.1, self.value_vars[h_choice] * (1 - adjusted_alpha))
        
        # Uncertainty growth for unchosen options
        if hazard_rate > 0:
            # Use bounded additive growth with hazard sensitivity
            uncertainty_growth = 0.025 * np.tanh(hazard_rate * 0.18)
            for i in range(self.num_options):
                if i != h_choice:
                    self.value_vars[i] = min(1.0, self.value_vars[i] + uncertainty_growth)
        
        # Update confidence estimate based on prediction accuracy
        # Simplified confidence update with trial-dependent decay for better generalization
        prediction_error_abs = abs(r_points - old_value)
        
        # Direct confidence update with bounded sensitivity
        # Use trial progress to modulate confidence update magnitude
        trial_progress = min(1.0, trial / max(1, horizon)) if horizon > 0 else 0.5
        
        if prediction_error_abs < 0.8:
            confidence_change = 0.015 * (1.0 - 0.2 * trial_progress)
        elif prediction_error_abs < 1.8:
            confidence_change = -0.005 * (1.0 - 0.1 * trial_progress)
        else:
            confidence_change = -0.02 * (1.0 - 0.15 * trial_progress)
        
        # Modulate by hazard rate: higher hazard → slower confidence updates
        confidence_change *= (1.0 - 0.03 * hazard_rate)
        
        self.confidence = max(0.3, min(0.85, self.confidence + confidence_change))
        
        # Uncertainty decay to prevent unbounded growth
        for i in range(self.num_options):
            # More sophisticated decay that considers hazard rate and trial progress
            trial_progress = min(1.0, trial / max(1, horizon)) if horizon > 0 else 0.5
            
            # Base decay rate with hazard-dependent adjustment
            # More volatile environments require faster uncertainty decay
            base_decay = 0.002
            if game in [4, 5]:
                base_decay = 0.0022 if trial > 4 else 0.0012  # More aggressive decay after instruction
            elif game == 3:
                base_decay = 0.0015  # More conservative decay for 4-option experiment
            
            # Hazard-modulated decay: higher hazard → faster uncertainty decay
            hazard_decay_mod = 1.0 + 0.03 * hazard_rate * (1.0 - trial_progress)
            trial_decay = base_decay * (0.5 + 0.5 * trial_progress) * hazard_decay_mod
            
            self.value_vars[i] = max(0.1, min(1.0, self.value_vars[i] * (1.0 - trial_decay)))
        
        # Track hazard rate change for next trial's volatility detection
        self.last_hazard = hazard_rate
        
        self.trial_count += 1


def define_parameters_and_bounds():
    """
    Define trainable parameters and their optimization bounds.
    Parameter definitions are shared across experiments, but parameter values are optimized independently per experiment.

    Returns:
    param_bounds_dictionary: Dictionary of parameter names and their (min, max) bounds for optimization
    """

    # Simplified parameter set with improved generalization across experiments
    # Removed asymmetry parameter for better stability and generalization
    return {
        'alpha': (0.05, 1.0),       # learning rate: how quickly values are updated
        'beta': (0.01, 10.0),       # inverse temperature: determines choice determinism
        'init_val': (-2.0, 2.0),    # initial value estimate
        'stickiness': (-0.8, 1.5),  # tendency to repeat previous choice
        'risk_sens': (0.6, 1.8),    # risk sensitivity parameter for utility curvature
        'base_bias': (-0.25, 0.25)  # narrower range for base_bias to improve generalization
    }

def get_init_param(experiment):
    """
    This function is called at the start of optimization to set the initial parameters of the model.
    Note, these initial parameters are matched against the bounds dictionary.
    An error will be raised if any additional parameters are added or some are unused for a particular experiment.
    """
    
    # Simplified parameter initialization for better generalization
    match experiment:
        case 1:
            # Simple 2-option with one always zero: moderate learning, high exploitation
            return {'alpha': 0.4, 'beta': 3.0, 'init_val': 0.0, 'stickiness': 0.4, 'risk_sens': 0.75, 'base_bias': 0.15}
        case 2:
            # Simple 2-option with both varying: similar to exp1 but slightly more risk seeking
            return {'alpha': 0.35, 'beta': 2.8, 'init_val': 0.0, 'stickiness': 0.45, 'risk_sens': 0.85, 'base_bias': 0.1}
        case 3:
            # 4 options: higher exploration to encourage discovery, moderate risk seeking
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'stickiness': 0.3, 'risk_sens': 0.85, 'base_bias': 0.0}
        case 4:
            # Instructed trials first: moderate learning with room for exploration after instructed phase
            return {'alpha': 0.25, 'beta': 1.5, 'init_val': 0.0, 'stickiness': 0.15, 'risk_sens': 0.8, 'base_bias': 0.05}
        case 5:
            # Similar to exp4 but potentially more exploration needed
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'stickiness': 0.3, 'risk_sens': 0.9, 'base_bias': 0.05}
        case _:
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'stickiness': 0.25, 'risk_sens': 0.85, 'base_bias': 0.1}

# EVOLVE-BLOCK END