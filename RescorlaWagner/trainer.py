import torch
import torch.nn.functional as F
import schedulefree
from tqdm import tqdm
import statistics as stat

def pd_to_pth(df, values, keys=['participant', 'game', 'trial']):
    column_names_list = [keys + [value] for value in values]
    wide_arrs = {}
    for column_names in column_names_list:
        arr = df[column_names].values
        dims = [np.unique(arr[:, i], return_inverse=True) for i in range(len(column_names)-1)]
        wide_arr = np.full([len(dims[i][0]) for i in range(len(column_names)-1)], np.nan)
        wide_arr[*[dims[i][1] for i in range(len(column_names)-1)]] = arr[:, -1]
        wide_arrs[column_names[-1]] = torch.from_numpy(wide_arr).reshape(-1, wide_arr.shape[-1])
    return wide_arrs

def preprocess_data(train_df, eval_df):
    """
    Preprocesses data into pytorch format.

    Parameter
    ---------
    train_df : pandas dataframe.
    eval_df : pandas dataframe.

    Returns
    -------
    dict, dict
    Dictionaries with tensors named 'choice' and 'reward' of shape (N, T).
    """
    ignore_index = -100
    if 'forced' in train_df:
        train_data = pd_to_pth(train_df, ['reward', 'choice', 'forced'])
    else:
        train_data = pd_to_pth(train_df, ['reward', 'choice'])

    if 'forced' in train_df:
        eval_data = pd_to_pth(eval_df, ['reward', 'choice', 'forced'])
    else:
        eval_data = pd_to_pth(eval_df, ['reward', 'choice'])

    # deal with nans
    train_data['choice'] = torch.nan_to_num(train_data['choice'], nan=ignore_index).long()
    eval_data['choice'] = torch.nan_to_num(eval_data['choice'], nan=ignore_index).long()

    # store copy used for updating
    train_data['choice_for_updating'] = train_data['choice'].clone().clamp(min=0)
    eval_data['choice_for_updating'] = eval_data['choice'].clone().clamp(min=0)

    # if forced, don't use for loss computation
    forced_mask = (torch.nan_to_num(train_data['forced'], nan=1) == 1)
    train_data['choice'][forced_mask] = ignore_index
    forced_mask = (torch.nan_to_num(eval_data['forced'], nan=1) == 1)
    eval_data['choice'][forced_mask] = ignore_index

    return train_data, eval_data

class Trainer:
    def __init__(self, model, num_iter=500):
        self.model = model
        self.num_iter = num_iter
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=0.1)

    def fit_and_evaluate(self, train_df, eval_df):
        ### PREPROCESS DATA ###
        train_data, eval_data = preprocess_data(train_df, eval_df)

        ### FITTING ###
        self.model.train()
        self.optimizer.train()

        # store losses
        all_losses = []

        for _ in tqdm(range(self.num_iter)):
            self.optimizer.zero_grad()
            logits = self.model(train_data)
            loss = F.cross_entropy(logits.flatten(0, -2), train_data['choice'].flatten().long())
            all_losses.append(loss.item())
            loss.backward()
            #print(loss.item(), flush=True)
            self.optimizer.step()

        ### EVALUATION ###
        self.model.eval()
        self.optimizer.eval()
        
        # extract logits and flatten
        with torch.no_grad():
            eval_logits = self.model(eval_data)
        flat_eval_logits = eval_logits.flatten(0, -2)

        # only return logits of non-padded data
        no_padding_eval_logits = flat_eval_logits[(eval_data['choice'] != self.model.ignore_index).flatten()]

        # return mean train_nll, eval_nll, and all evaluate logits logits
        return all_losses, F.cross_entropy(eval_logits.flatten(0, -2), eval_data['choice'].flatten().long()).item(), no_padding_eval_logits