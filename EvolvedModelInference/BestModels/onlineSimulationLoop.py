# loop over each row (i.e., trial) in the data
for i in range(len(data)):

    # extract the data from the row
    participant = all_participant[i]
    game = all_game[i]
    horizon = all_horizon[i]
    trial = all_trial[i]
    forced = all_forced[i]
    human_choice = all_human_choice[i]
    reward = all_reward[i]
    hazard_rate = all_hazard_rate[i]

    # Initialize model state per participant
    if participant != prev_participant:
        model.participant_reset()
        prev_participant = participant
        prev_game = None

    # Reset game-specific variables
    if game != prev_game:
        model.game_reset(game, horizon, hazard_rate)
        prev_game = game

    # Predict action probabilities
    logits = model.predict(game, trial, horizon, hazard_rate, forced)

    if forced == 0:
        # store logits for vectorized computations later
        all_logits.append(logits)

    # Update model based on human choice and reward
    model.update(game, trial, horizon, hazard_rate, forced, human_choice, reward)