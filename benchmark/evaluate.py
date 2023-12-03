from src.data.make_dataset import one_hot, preprocess, _preprocess
from tqdm import tqdm

import torch


# Return MSE loss for given data and model
def evaluate(model, data, user_data, movie_data, device=torch.device("cpu")):
    mseLoss = 0.0
    for _row in tqdm(data.iterrows()):
        row = _row[1]
        row = preprocess(row, user_data, movie_data, model.vocab)
        row = one_hot(row.loc[0], model.vocab)
        model_input, target = _preprocess(row.loc[0], device=device)
        model_input, target = model_input.reshape(1, -1), target
        res = model(model_input)
        mseLoss += (res.item() - target.item())**2
    
    mseLoss /= len(data)
    return mseLoss