import ast
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Temperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, values):
        """
        Multiplies the input tensor with a parameter.

        Parameter
        ---------
        values : tensor of any shape.

        Returns
        -------
        tensor of any shape
        Scaled tensor.
        """

        return values * self.beta

class Stickiness(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, choices):
        """
        Returns whether a choice has been selected on the previous trial.

        Parameter
        ---------
        choices : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with ones if choices has been selected on the previous trial.
        """

        num_tasks = choices.shape[0]

        previous_choices_0 = torch.zeros(num_tasks, 1, self.num_options)
        previous_choices_1 = torch.stack([(choices[:, :-1] == a).float() for a in range(self.num_options)], dim=-1)
        previous_choices = torch.cat([previous_choices_0, previous_choices_1], dim=1)

        return previous_choices * self.beta

class InformationBonus(nn.Module):
    def __init__(self, num_options):
        super().__init__()
        self.num_options = num_options
        self.beta = nn.Parameter(0.01 * torch.randn([]))

    def forward(self, choices):
        """
        Returns how often a choice has been chosen up to trial t.

        Parameter
        ---------
        choices : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor containing how often a choice has been chosen up to trial t.
        """

        num_tasks = choices.shape[0]

        cumsum_choices_0 = torch.zeros(num_tasks, 1, self.num_options)
        cumsum_choices_1 = torch.stack([torch.cumsum((choices[:, :-1] == a).float(), dim=1) for a in range(self.num_options)], dim=-1)
        cumsum_choices = torch.cat([cumsum_choices_0, cumsum_choices_1], dim=1)

        return cumsum_choices * self.beta

class TabularRescorlaWagnerPlusMinusValueUpdating(nn.Module):
    def __init__(self, num_options, max_initial_values=100, ignore_index=-100):
        super().__init__()

        self.num_options = num_options
        self.max_initial_values = max_initial_values

        self.alpha_plus = nn.Parameter(0.01 * torch.randn([]))
        self.alpha_minus = nn.Parameter(0.01 * torch.randn([]))
        self.initial_values = nn.Parameter(0.01 * torch.randn([]))

        self.ignore_index = ignore_index

    def forward(self, choices, rewards):
        """
        Performs Rescorla-Wagner updating with separate learning rates for positive and negative prediction errors for the given choices and rewards.

        Parameter
        ---------
        choices : tensor of shape (N, T).
        rewards : tensor of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with estimated values for all options.
        """

        num_tasks = choices.shape[0]
        num_trials = choices.shape[1]

        initial_values = self.max_initial_values * F.tanh(self.initial_values)
        alpha_plus = F.sigmoid(self.alpha_plus)
        alpha_minus = F.sigmoid(self.alpha_minus)

        values = torch.ones(list(choices.shape) + [self.num_options]) * initial_values

        for t in range(num_trials-1):
            # copy over everything
            values[:, t+1, :] = values[:, t, :]
            # compute prediction errors
            prediction_error = rewards[:, t] - values[torch.arange(num_tasks), t, choices[:, t]]
            # zero-out prediction errors for missing trial
            prediction_error[torch.isnan(rewards[:, t])] = 0
            # update values for selected actions
            values[torch.arange(num_tasks), t+1, choices[:, t]] = values[torch.arange(num_tasks), t, choices[:, t]] + (alpha_plus * prediction_error * (prediction_error >= 0).float()) + (alpha_minus * prediction_error * (prediction_error < 0).float())

        return values

class RescorlaWagnerModel(nn.Module):
    def __init__(self, num_options):
        super().__init__()

        self.num_options = num_options
        self.ignore_index = -100

        self.value_updating = TabularRescorlaWagnerPlusMinusValueUpdating(num_options)

        self.information_logits = InformationBonus(num_options)
        self.stickiness_logits = Stickiness(num_options)
        self.value_logits = Temperature()

    def forward(self, data):
        """
        Model with Rescorla-Wagner-based learning, stickiness and information bonus.

        Parameter
        ---------
        data : dictionary with tensors named 'choice' and 'reward' of shape (N, T).

        Returns
        -------
        tensor of shape (N, T, self.num_options)
        Tensor filled with logits for all options.
        """

        values = self.value_updating(data['choice_for_updating'].long(), data['reward'].float())

        information_logits = self.information_logits(data['choice'].long())
        stickiness_logits = self.stickiness_logits(data['choice'].long())
        value_logits = self.value_logits(values)
        return value_logits + stickiness_logits + information_logits
    