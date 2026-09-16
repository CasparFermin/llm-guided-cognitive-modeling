# EVOLVE-BLOCK START
import numpy as np


class Model:
    def __init__(self):
        self.num_options = None
        self.alpha = None
        self.beta = None
        self.init_val = 0.0
        self.stickiness = 0.0
        self.risk_sens = 1.0  # risk sensitivity parameter
        self.choice_history = []  # Track recent choices for habit modeling
        self.confidence = 0.6  # Start with moderate confidence

    def set_params(self, num_options, params_dict):
        self.num_options = num_options
        self.alpha = params_dict["alpha"]
        self.beta = params_dict["beta"]  # inverse temperature for choice determinism
        self.init_val = params_dict.get("init_val", 0.0)
        self.stickiness = params_dict.get("stickiness", 0.0)
        self.risk_sens = params_dict.get("risk_sens", 1.0)  # risk sensitivity for utility curvature
        # Fixed parameters for hazard sensitivity and uncertainty growth
        self.hazard_explore_weight = 0.15  # hazard-based exploration
        self.uncertainty_growth = 0.05  # growth rate for uncertainty in unchosen options

    def participant_reset(self):
        # Reset participant-level state: persistent memory or biases
        self.values = None
        self.value_vars = None
        self.last_choice = None
        self.choice_history = []  # Reset choice history
        self.confidence = 0.6  # Reset confidence

    def game_reset(self, game, horizon, hazard_rate):
        # Reset game-specific state: action values and exploration bonus
        self.values = [self.init_val] * self.num_options
        self.value_vars = [1.0] * self.num_options  # uncertainty estimates
        self.trials_in_game = 0
        self.last_choice = None
        self.choice_history = []  # Reset choice history
        # Adjust initial confidence based on hazard rate (lower confidence in volatile environments)
        self.confidence = max(0.3, 0.6 - 0.03 * hazard_rate)

    def predict(self, game, trial, horizon, hazard_rate, forced):
        # Returns raw action logits (unnormalized scores)
        if forced:
            return [0.0] * self.num_options
        
        # Apply risk sensitivity to values before computing logits
        # Risk sensitivity: concave transformation for gains, convex for losses
        risk_adjusted_values = []
        for v in self.values:
            if v >= 0:
                risk_adjusted_values.append(np.power(v + 1e-6, self.risk_sens))
            else:
                risk_adjusted_values.append(-np.power(-v + 1e-6, self.risk_sens))
        
        # Base logits from risk-adjusted values
        logits = [self.beta * v for v in risk_adjusted_values]
        
        # Add uncertainty-based exploration bonus (more uncertainty → more exploration)
        uncertainty_bonus = [0.3 * np.sqrt(var) for var in self.value_vars]
        logits = [l + u for l, u in zip(logits, uncertainty_bonus)]
        
        # Add trial progress exploration (more exploration early in game)
        if horizon > 0:
            trial_progress = trial / max(1, horizon)
            exploration_bonus = 0.2 * (1.0 - trial_progress)
            logits = [l + exploration_bonus for l in logits]
        
        # Add stickiness for previous choice
        if self.last_choice is not None and self.stickiness != 0:
            logits[self.last_choice] += self.stickiness
        
        # Add habit component for repeated choices (more pronounced in stable environments)
        if len(self.choice_history) >= 2 and self.stickiness != 0:
            # Count how many times the current option was chosen recently
            recent_choices = self.choice_history[-3:] if len(self.choice_history) >= 3 else self.choice_history
            choice_count = recent_choices.count(self.last_choice)
            # Add habit bonus that scales with choice frequency
            habit_bonus = self.stickiness * 0.3 * choice_count * (1.0 - hazard_rate/10.0)
            logits[self.last_choice] += habit_bonus
        
        # Enhanced hazard-modulated exploration with inverted-U response
        # Humans show peak sensitivity to changes at moderate hazard levels (5-7)
        if hazard_rate > 0 and self.last_choice is not None:
            # Create an inverted-U response to hazard rate that peaks around hazard=6
            hazard_factor = hazard_rate * (10 - hazard_rate) / 25.0  # Peaks at hazard=5
            logits[self.last_choice] += self.hazard_explore_weight * hazard_factor
        
        # Add confidence-modulated exploration
        # Lower confidence → more exploration, especially in early trials
        if self.confidence < 0.8 and horizon > 0:
            confidence_exploration = 0.2 * (1.0 - self.confidence) * (1.0 - min(1.0, trial/max(1, horizon)))
            logits = [l + confidence_exploration for l in logits]
        
        # Normalize logits to avoid extreme values
        max_logit = max(logits)
        logits = [l - max_logit for l in logits]
        
        return logits

    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        # Update action values only on free trials (or learning may be suppressed in forced)
        if forced and game in [4, 5] and trial <= 4:
            return  # Skip learning on initial forced trials in experiments 4/5
        
        # Track last choice for stickiness effect (even on forced trials)
        self.last_choice = h_choice
        
        # Store choice for habit tracking
        self.choice_history.append(h_choice)
        # Keep only recent choices for habit calculation (last 3)
        if len(self.choice_history) > 3:
            self.choice_history.pop(0)
        
        # Rescorla-Wagner update: value += alpha * (reward - value)
        old_value = self.values[h_choice]
        prediction_error = r_points - old_value
        
        # Hazard-modulated learning rate
        adjusted_alpha = self.alpha
        if hazard_rate > 0:
            # Smooth hazard sensitivity using sigmoid for bounded effect
            hazard_factor = np.tanh(hazard_rate * 0.15)
            adjusted_alpha = min(1.0, self.alpha * (1.0 + 0.15 * hazard_factor))
        
        # Asymmetric learning: amplify positive prediction errors more
        # This captures the human tendency to update more strongly for better-than-expected outcomes
        if prediction_error > 0:
            # Amplify positive PEs, especially in volatile environments
            pe_weight = 1.0 + 0.1 * hazard_rate / 10.0
        else:
            # Dampen negative PEs slightly more than positive ones
            pe_weight = max(0.8, 1.0 - 0.05 * hazard_rate / 10.0)
        
        effective_alpha = min(1.0, adjusted_alpha * pe_weight)
        self.values[h_choice] += effective_alpha * prediction_error
        
        # Update uncertainty estimates - reduce uncertainty after feedback
        self.value_vars[h_choice] = max(0.1, self.value_vars[h_choice] * (1 - adjusted_alpha))
        
        # Slightly increase uncertainty for unchosen options with smooth hazard response
        uncertainty_growth = self.uncertainty_growth * np.tanh(hazard_rate * 0.2)
        for i in range(self.num_options):
            if i != h_choice:
                # More sophisticated uncertainty growth that considers current uncertainty level
                current_uncertainty = self.value_vars[i]
                uncertainty_boost = 1.0 + 0.1 * current_uncertainty
                self.value_vars[i] = min(1.0, self.value_vars[i] * (1.0 + uncertainty_growth * uncertainty_boost))
        
        # Update confidence estimate based on prediction accuracy and environmental stability
        prediction_error_abs = abs(r_points - old_value)
        # Update running average of prediction errors
        self.avg_prediction_error = 0.9 * getattr(self, 'avg_prediction_error', 0.0) + 0.1 * prediction_error_abs
        
        # Update confidence: lower confidence for larger prediction errors and higher hazard
        confidence_update = 0.05 * (1.0 - min(1.0, self.avg_prediction_error / 5.0)) * (1.0 - hazard_rate/10.0)
        self.confidence = max(0.3, min(0.9, self.confidence + confidence_update))
        
        self.trials_in_game += 1


def define_parameters_and_bounds():
    return {
        'alpha': (0.01, 1.0),       # learning rate: how quickly values are updated
        'beta': (0.01, 10.0),       # inverse temperature: determines choice determinism
        'init_val': (-5.0, 5.0),    # initial value estimate
        'stickiness': (-1.0, 2.0),  # tendency to repeat previous choice
        'risk_sens': (0.5, 2.0)     # risk sensitivity parameter for utility curvature
    }

def get_init_param(experiment):
    # Set initial parameters values depending on the experiment
    # Base values that work well across most reinforcement learning tasks
    match experiment:
        case 1:
            # Simple 2-option with one always zero: moderate learning, high exploitation, slight risk aversion
            return {'alpha': 0.3, 'beta': 3.0, 'init_val': 0.0, 'stickiness': 0.5, 'risk_sens': 0.7}
        case 2:
            # Simple 2-option with both varying: similar to exp1 but slightly more risk seeking
            return {'alpha': 0.3, 'beta': 3.0, 'init_val': 0.0, 'stickiness': 0.5, 'risk_sens': 0.8}
        case 3:
            # 4 options: higher exploration to encourage discovery, moderate risk seeking
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'stickiness': 0.3, 'risk_sens': 0.9}
        case 4:
            # Instructed trials first: moderate learning with room for exploration after instructed phase
            return {'alpha': 0.35, 'beta': 1.5, 'init_val': 0.0, 'stickiness': 0.0, 'risk_sens': 0.85}
        case 5:
            # Similar to exp4 but potentially more exploration needed, slightly more risk seeking
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'stickiness': 0.2, 'risk_sens': 0.9}
        case _:
            return {'alpha': 0.3, 'beta': 2.0, 'init_val': 0.0, 'stickiness': 0.3, 'risk_sens': 0.85}
# EVOLVE-BLOCK END