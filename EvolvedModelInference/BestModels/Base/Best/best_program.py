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
        self.last_instruction_choice = None  # Track last instruction trial choice
        self.instruction_reference_value = 0.0  # Reference value for instruction-following
        self.instruction_following_strength = 0.0  # Dynamic instruction-following strength
        self.trials_after_instruction = 0  # Track trials after instruction phase
        self.aspiration = 0.0  # Reference point for outcome evaluation
        self.aspiration_trials = 0  # Track trials used for aspiration calculation
        self.positive_pe_weight = 0.3  # Initial weight for positive prediction errors
        self.negative_pe_weight = -0.1  # Initial weight for negative prediction errors
        self.instruction_confidence = 0.0  # Confidence in instruction-following behavior
        self.instruction_outcomes = []  # Track instruction trial outcomes for confidence estimation
        self.instruction_consistency = 0  # Track consistency with instruction recommendations

    def predict(self, game, trial, horizon, hazard_rate, forced):
        # Returns raw action logits of shape (num_options,), do NOT apply softmax or convert to probabilities.
        # For forced trials, return 0 for all options (will be ignored in NLL)
        if forced:
            return [0] * self.num_options
        
        # Track if in instruction phase based on experiment and trial
        # Experiments 4&5 have instruction trials, but different lengths (4 vs 1-6 free choices)
        self.instruction_phase = (game >= 4 and trial <= 4)
        
        # Adjust temperature based on trial progression (more exploitation as game progresses)
        trial_progress = min(trial / max(horizon, 1), 1.0)
        trial_factor = 1.0 - self.trial_decay * trial_progress
        effective_beta = self.beta * max(0.15, trial_factor)  # Ensure minimum temperature
        
        # Adjust temperature based on hazard rate - higher hazard may need more exploration
        hazard_factor = 1.0 + self.hazard_sens * hazard_rate / 20.0
        effective_beta *= hazard_factor
        
        # Compute logits with value normalization based on current instruction phase and context
        if self.instruction_phase:
            # During instruction trials, focus more on relative value differences
            # and instruction-following behavior
            value_range = np.max(self.value_estimates) - np.min(self.value_estimates)
            # Use more conservative normalization to prevent extreme logits
            value_scale = min(0.8, 3.0 / max(value_range, 0.1))
            
            # Find best and worst options for instruction-following logic
            best_option = int(np.argmax(self.value_estimates))
            worst_option = int(np.argmin(self.value_estimates))
            value_diff = self.value_estimates[best_option] - self.value_estimates[worst_option]
            
            # Instruction-following strength depends on value uncertainty and trial progress
            # Use sigmoid function to model accelerating instruction-following confidence
            instruction_progress = trial / 4.0
            self.instruction_following_strength = 0.25 * (1 - np.exp(-1.8 * instruction_progress))
            
            # Strength increases when instruction-recommended option has clearer value
            instruction_strength = self.instruction_following_strength * (1 + 0.4 * self.hazard_sens) * (0.5 + value_diff / 10.0)
            
            logits = []
            for i in range(self.num_options):
                # Scale value estimates by decision temperature parameter
                value_component = self.value_estimates[i] * effective_beta * value_scale
                
                # Add exploration bonus based on how rarely this option has been chosen
                exploration_bonus = self.eps / np.sqrt(self.choice_counts[i] + 1)
                
                # Add aspiration-based evaluation - options above aspiration get extra boost
                aspiration_bonus = 0.0
                if self.aspiration_trials > 2:
                    aspiration_diff = self.value_estimates[i] - self.aspiration
                    aspiration_bonus = 0.08 * effective_beta * min(1.0, max(-1.0, aspiration_diff / 4.0))
                
                # Add choice stability bonus - prefer previously chosen options
                choice_stability = 0.08 * effective_beta
                if self.choice_counts[i] > 0:
                    stability_bonus = choice_stability * np.log(self.choice_counts[i] + 1)
                else:
                    stability_bonus = 0.0
                
                # Add choice persistence - tendency to repeat previous choice
                if self.last_choice is not None and i == self.last_choice:
                    persistence_bonus = 0.12 * effective_beta * (1 - trial_progress)
                    stability_bonus += persistence_bonus
                
                # Add instruction-following consistency - prefer instruction-chosen options
                if self.instruction_phase and self.last_instruction_choice is not None and i == self.last_instruction_choice:
                    instruction_bonus = instruction_strength * effective_beta * (1 - trial_progress * 0.25)
                    stability_bonus += instruction_bonus
                
                # Add choice bias for first option in 2-option games (early trials)
                bias = 0.0
                if self.num_options == 2 and trial <= 4:
                    bias = self.choice_bias if i == 0 else -self.choice_bias * 0.3
                
                # Adjust instruction trials to have more exploration and different stability
                if self.instruction_phase:
                    stability_bonus *= 0.75  # Reduced stability during instruction phase
                    exploration_bonus *= 0.8
                
                logit = value_component + exploration_bonus + stability_bonus + aspiration_bonus + bias
                logits.append(logit)
        else:
            # Normal free-choice trials
            # Normalize value estimates to prevent extreme logits
            value_range = np.max(self.value_estimates) - np.min(self.value_estimates)
            value_scale = min(0.9, 5.0 / max(value_range, 0.1))
            
            # Compute logits as value estimates with temperature scaling and multiple bias components
            logits = []
            for i in range(self.num_options):
                # Scale value estimates by decision temperature parameter and normalization
                value_component = self.value_estimates[i] * effective_beta * value_scale
                
                # Add exploration bonus based on how rarely this option has been chosen
                # Use inverse square root for more realistic exploration decay
                exploration_bonus = self.eps / np.sqrt(self.choice_counts[i] + 1)
                
                # Add aspiration-based evaluation - options above aspiration get extra boost
                aspiration_bonus = 0.0
                if self.aspiration_trials > 2:
                    aspiration_diff = self.value_estimates[i] - self.aspiration
                    aspiration_bonus = 0.1 * effective_beta * min(1.0, max(-1.0, aspiration_diff / 4.0))
                
                # Add choice stability bonus - prefer previously chosen options
                # This captures human tendency to stick with familiar choices
                choice_stability = 0.1 * effective_beta
                if self.choice_counts[i] > 0:
                    stability_bonus = choice_stability * np.log(self.choice_counts[i] + 1)
                else:
                    stability_bonus = 0.0
                
                # Add choice persistence - tendency to repeat previous choice
                if self.last_choice is not None and i == self.last_choice:
                    persistence_bonus = 0.15 * effective_beta * (1 - trial_progress)
                    stability_bonus += persistence_bonus
                
                # Add instruction-following consistency bonus for games 4&5 after instruction phase
                if game >= 4 and self.trials_after_instruction > 0 and self.last_instruction_choice is not None:
                    # Instruction-following consistency decays over trials after instruction
                    instruction_decay = np.exp(-0.25 * self.trials_after_instruction)
                    if i == self.last_instruction_choice:
                        # Use instruction confidence to modulate consistency bonus
                        consistency_bonus = 0.2 * self.instruction_following_strength * self.instruction_confidence * instruction_decay * effective_beta
                        stability_bonus += consistency_bonus
                
                # Add choice bias for first option in 2-option games (early trials)
                bias = 0.0
                if self.num_options == 2 and trial <= 4:
                    bias = self.choice_bias if i == 0 else -self.choice_bias * 0.3
                
                logit = value_component + exploration_bonus + stability_bonus + aspiration_bonus + bias
                logits.append(logit)
        
        return logits

    def update(self, game, trial, horizon, hazard_rate, forced, h_choice, r_points):
        # Update state based on trial feedback
        
        # Determine learning dynamics based on trial type and experiment context
        if forced and game < 4:
            # For experiments 1-3, forced trials don't provide learning opportunity
            # Only update when participants make their own choice
            return
        
        # Track instruction trial outcomes for confidence estimation
        if self.instruction_phase:
            self.instruction_outcomes.append(r_points)
            # Update aspiration reference with instruction outcomes
            self.aspiration_trials += 1
            if self.aspiration_trials == 1:
                self.aspiration = r_points
            else:
                # Use exponential moving average for aspiration update
                aspiration_weight = min(0.3, 0.08 * self.aspiration_trials)
                self.aspiration = (1 - aspiration_weight) * self.aspiration + aspiration_weight * r_points
            
            # Update instruction confidence based on outcome consistency
            if len(self.instruction_outcomes) > 1:
                # Track how consistent instruction outcomes are
                outcome_diff = abs(r_points - self.instruction_outcomes[-2]) if len(self.instruction_outcomes) >= 2 else 0
                if outcome_diff < 1.5:  # More sensitive to outcome consistency
                    self.instruction_confidence = min(1.0, self.instruction_confidence + 0.07)
                else:  # Different outcomes reduce confidence
                    self.instruction_confidence = max(0.1, self.instruction_confidence - 0.06)
        else:
            # Update trials since last instruction for confidence decay
            if game >= 4:
                self.trials_after_instruction += 1
                # Instruction-following confidence decays after instruction phase
                instruction_decay = np.exp(-0.2 * self.trials_after_instruction)
                self.instruction_confidence *= instruction_decay
        
        # Adjust learning rate based on hazard rate and trial type
        # Instruction trials (forced in exp 4&5) have different learning dynamics
        if forced and self.instruction_phase:
            # Track instruction trial outcomes for confidence estimation
            # Use more sophisticated instruction-following model with confidence updates
            # Start with lower confidence and increase as instruction trials progress
            if trial == 1:
                self.instruction_confidence = 0.1  # Start with low confidence
            
            # Track instruction trial outcomes for confidence estimation
            # Instruction-following confidence builds up gradually
            instruction_progress = min(trial / 4.0, 1.0)
            # Use sigmoid function to model accelerating instruction-following confidence
            instruction_confidence = 0.15 + 0.75 * (1 / (1 + np.exp(-2.5 * (instruction_progress - 0.35))))
            # Adjust confidence based on outcome quality
            if len(self.instruction_outcomes) >= 2:
                avg_outcome = np.mean(self.instruction_outcomes)
                outcome_quality = max(0.1, min(1.0, (avg_outcome + 2.0) / 4.0))
                instruction_confidence *= outcome_quality
            
            # Calculate instruction-following learning rate
            # More confident participants follow instructions more strongly and learn more efficiently
            adjusted_alpha = self.alpha * instruction_confidence * (1 + self.hazard_sens * self.game_hazard_rate / 9.0)
            
            # Update instruction-specific value estimates
            current_instr_value = self.value_estimates[h_choice]
            instr_pred_error = r_points - current_instr_value
            self.value_estimates[h_choice] = current_instr_value + adjusted_alpha * instr_pred_error
            
            # Update instruction-following strength based on consistency with previous instruction choices
            if hasattr(self, 'last_instruction_choice') and self.last_instruction_choice is not None:
                if h_choice == self.last_instruction_choice:
                    # Strengthen instruction-following when consistent
                    self.instruction_following_strength = min(1.0, self.instruction_following_strength + 0.07)
                else:
                    # Weaken instruction-following when inconsistent
                    self.instruction_following_strength = max(0.0, self.instruction_following_strength - 0.06)
        else:
            # Free-choice trials and later forced trials use higher learning rates
            # Adjust learning rate based on whether outcome was better or worse than aspiration
            current_value = self.value_estimates[h_choice]
            prediction_error = r_points - current_value
            
            # Aspiration-adjusted learning: learn more when outcomes exceed reference point
            aspiration_factor = 1.0 + 0.35 * np.sign(prediction_error) * np.tanh(abs(prediction_error) / 5.0)
            adjusted_alpha = self.alpha * aspiration_factor * (1 + self.hazard_sens * self.game_hazard_rate / 10.0)
            
            # Update the value of the chosen option using Q-learning rule
            self.value_estimates[h_choice] = current_value + adjusted_alpha * prediction_error
        
        # Apply value decay to unchosen options - stronger in high hazard environments
        # But instruction trials don't cause strong forgetting
        if not self.instruction_phase and self.game_hazard_rate > 2:
            # Use a more flexible forgetting mechanism with separate decay for different option types
            # Options with low confidence in instruction phase get stronger forgetting
            for i in range(self.num_options):
                if i != h_choice:
                    # Base forgetting rate
                    base_decay = 0.025 + 0.004 * self.game_hazard_rate
                    
                    # Additional forgetting for options that weren't followed during instruction
                    if hasattr(self, 'last_instruction_choice') and self.last_instruction_choice is not None and i != self.last_instruction_choice:
                        # Options not chosen during instruction decay faster
                        base_decay += 0.008
                    
                    # Apply forgetting with decay rate
                    self.value_estimates[i] *= (1 - base_decay)
        else:
            # Minimal decay for instruction trials or low hazard environments
            for i in range(self.num_options):
                if i != h_choice:
                    if self.instruction_phase:
                        decay_rate = 0.003
                    else:
                        decay_rate = 0.007
                    self.value_estimates[i] *= (1 - decay_rate)
        
        # Update choice counts for exploration bonus
        # Use different update rules for different trial types
        if not forced:
            # Free-choice trials get full weight
            self.choice_counts[h_choice] += 1.0
            self.trials_in_game += 1
            self.last_choice = h_choice  # Track for sticky bias
        elif forced and game >= 4:
            # Update choice counts for forced trials in instruction phase, but with smaller increments
            # This allows instruction trials to influence later free choices without overwhelming the model
            self.choice_counts[h_choice] += 0.25
            
            # Track instruction-following behavior for consistency effects
            # Store the instruction-following choice for later reference
            self.last_instruction_choice = h_choice


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