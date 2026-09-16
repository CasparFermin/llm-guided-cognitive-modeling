from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np

experiments = [
    {'name': 'TwoBandit', 'experiment': 'exp1', 'split': 'Train'},
    {'name': 'TwoBandit', 'experiment': 'exp2', 'split': 'Train'},
    {'name': 'HorizonSomer', 'experiment': 'exp0', 'split': 'Train'},
    {'name': 'HorizonWaltz', 'experiment': 'exp0', 'split': 'Train'},
    {'name': 'DriftingBandit', 'experiment': 'exp0', 'split': 'Train'},
    {'name': 'ChangingBandit', 'experiment': 'exp0', 'split': 'OOD'},
    {'name': 'HorizonSade', 'experiment': 'exp0', 'split': 'OOD'},
    {'name': 'HorizonFeng', 'experiment': 'exp0', 'split': 'OOD'},
    {'name': 'MaggiesFarm', 'experiment': 'exp0', 'split': 'OOD'},
]

# target columns in exact order
TARGET_COLS = [
    'participant', 'game', 'horizon', 'trial', 'forced', 'human_choice', 'reward', 'hazard_rate'
]

exps_with_nans = []

# loop over all data
for exp in experiments:

    # load data
    raw_data = pd.read_csv(f"../RawData/{exp['name']}/full_data_{exp['experiment']}.csv")
    text_data = pd.read_csv(f"../RawData/{exp['name']}/text_data_{exp['experiment']}.csv")

    """
    All experiments should contain the data column:
    participant:    int32   - The participant ID
    game:           int32   - The number of the game, starts at 1
    horizon:        int32   - The number of trials per game, -1 if not known by the participant
    trial:          int32   - The trial number, resets every game and starts at 1
    forced:         int32   - Either 0 or 1, with 1 representing a non-trial, which won't be taken into account during computation of nll
    human_choice:   int32   - The human choice that has to be modelled
    reward:         int32   - The reward as a consequence of the human_choice
    hazard_rate:    float64 - Only relevant in the test-set
    """

    # drop reaction time column
    if 'RT' in raw_data.columns:
        raw_data = raw_data.drop(['RT'], axis=1).copy()

    # remove nans and check the difference
    data = raw_data.dropna().copy()
    len_rdata = len(raw_data)
    len_data =  len(data)

    exps_with_nans.append({
        "name": exp['name'],
        "removed_rows": len_rdata - len_data,
        "total_rows": len_data,
        "nan_fraction": len_rdata / len_data
    })

    # fix trial index
    data['trial'] = data.groupby(['participant', 'task']).cumcount()
    
    # 2. Create clean dataframe with defaults
    clean_data = pd.DataFrame({
        'participant':  data['participant'].astype('int32'),
        'game':         1 + data['task'].astype('int32'),
        'trial':        1 + data['trial'].astype('int32'),
        'human_choice': data['choice'].astype('int32'),
        'reward':       data['reward'].astype('int32'),
        'forced':       np.zeros_like(data['trial']).astype('int32'),      # Default 0
        'hazard_rate':  np.zeros_like(data['trial']).astype('int32'),      # Default 0
        'horizon':      (np.zeros_like(data['trial']) - 1).astype('int32') # Default -1 (if horizon is unknown)
    })

    # experiment-specific overrides
    if exp['name'] == 'ChangingBandit':
        clean_data['hazard_rate'] = data['hazard_rate'] * 10

    elif exp['name'] in ['HorizonSade', 'HorizonSomer', 'HorizonFeng', 'HorizonWaltz']:
        clean_data['horizon'] = data['horizon']
        clean_data['forced'] = data['forced']

    elif exp['name'] == 'MaggiesFarm':
        clean_data['forced'] = data['forced']

    # group by participant and game to find the max trial for that specific block
    if exp['name'] not in ['HorizonSade', 'HorizonSomer', 'HorizonFeng', 'HorizonWaltz', 'DriftingBandit']:

        # add horizon to the other tasks, as nearly all tasks indicate horizon (e.g., 10 trials per game)
        clean_data['horizon'] = clean_data.groupby(['participant', 'game'])['trial'].transform('max')

    # get a list of the participant IDs
    participants = clean_data['participant'].unique()
    n = len(participants)

    # shuffle them
    rng = np.random.RandomState(42)
    rng.shuffle(participants)
    
    if exp['split'] == 'Train':
        # get rounded index point of participants for train and val
        if exp['name'] == 'DriftingBandit':
            train_cut = int(0.6 * n)
            val_cut = int(0.70 * n)
        else:
            train_cut = int(0.7 * n)
            val_cut = int(0.80 * n)

        # cut the unique participant list into proportional parts
        train_parts = participants[:train_cut]
        val_parts = participants[train_cut:val_cut]
        test_parts = participants[val_cut:]

        # extract the data based on the indexes
        train_data = clean_data[clean_data['participant'].isin(train_parts)]
        val_data = clean_data[clean_data['participant'].isin(val_parts)]
        test_data = clean_data[clean_data['participant'].isin(test_parts)]

        # extract the text data based on the indexes
        train_text = text_data[text_data['participant'].isin(train_parts)]
        val_text = text_data[text_data['participant'].isin(val_parts)]
        test_text = text_data[text_data['participant'].isin(test_parts)]

        # save the split text data as processed csv
        train_text.to_csv(f"{exp['name']}/Train_text_{exp['experiment']}.csv", index=False)
        val_text.to_csv(f"{exp['name']}/Val_text_{exp['experiment']}.csv", index=False)
        test_text.to_csv(f"{exp['name']}/Test_text_{exp['experiment']}.csv", index=False)

        # ensure columns are in the exact target order and convert to Numpy for both the datasets
        train_num = train_data[TARGET_COLS].to_numpy()
        np.save(f"{exp['name']}/proc_Train_{exp['experiment']}.npy", train_num)
        val_num = val_data[TARGET_COLS].to_numpy()
        np.save(f"{exp['name']}/proc_Val_{exp['experiment']}.npy", val_num)
        test_num = test_data[TARGET_COLS].to_numpy()
        np.save(f"{exp['name']}/proc_Test_{exp['experiment']}.npy", test_num)

    # if get the different splits for the OOD experiments
    elif  exp['split'] == 'OOD':
        # get train_cut
        train_cut = int(0.8 * n)

        # cut the unique participant list into proportional parts
        train_parts = participants[:train_cut]
        test_parts = participants[train_cut:]

        # extract the data based on the indexes
        train_data = clean_data[clean_data['participant'].isin(train_parts)]
        test_data = clean_data[clean_data['participant'].isin(test_parts)]

        # extract the text data based on the indexes
        train_text = text_data[text_data['participant'].isin(train_parts)]
        test_text = text_data[text_data['participant'].isin(test_parts)]

        # save the split text data as processed csv
        train_text.to_csv(f"{exp['name']}/Train_text_{exp['experiment']}.csv", index=False)
        test_text.to_csv(f"{exp['name']}/Test_text_{exp['experiment']}.csv", index=False)

        # ensure columns are in the exact target order and convert to Numpy for both the datasets
        train_num = train_data[TARGET_COLS].to_numpy()
        np.save(f"{exp['name']}/proc_Train_{exp['experiment']}.npy", train_num)
        test_num = test_data[TARGET_COLS].to_numpy()
        np.save(f"{exp['name']}/proc_Test_{exp['experiment']}.npy", test_num)
    else:
        print("error, neither split:", exp['name'])


# --- Summary Report ---
if exps_with_nans:
    print("\n Experiments containing NaNs:")
    for entry in exps_with_nans:
        print(f"- {entry['name']}: {entry['removed_rows']} rows removed")
else:
    print("\n No NaNs found in any experiments.")

# Result:
"""
 Experiments containing NaNs:
- TwoBandit: 0 rows removed
- TwoBandit: 0 rows removed
- ChangingBandit: 0 rows removed
- DriftingBandit: 4934 rows removed
- HorizonSade: 0 rows removed
- HorizonSomer: 0 rows removed
- HorizonFeng: 0 rows removed
- HorizonWaltz: 0 rows removed
- MaggiesFarm: 0 rows removed
"""
