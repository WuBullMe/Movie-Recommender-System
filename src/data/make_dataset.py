import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Vocabulary:
    """
        Vocabulary of words:
    """
    def __init__(self, name):
        self.name = name
        self.columns = []
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def addColumn(self, data, column_name):
        column = data[column_name].tolist()
        self.columns.append(column_name)
        for row in column:
            self.addWord(column_name + f"_{row}")
            

def one_hot(row, vocab, ret_dataFrame=True):
    columns = []
    for col in row.index.tolist():
        if col in vocab.columns:
            continue
        columns.append(col)
    G = 0
    new_col = len(columns)
    for key, val in vocab.word2index.items():
        columns.append(key)

    n_row = []
    for col in row.index.tolist():
        if col in columns:
            n_row.append(row[col])
    
    for col_ in columns[new_col:]:
        cur = 0
        for n_col in vocab.columns:
            if col_.startswith(n_col):
                val = col_[len(n_col)+1:] 
                if row[n_col] == val:
                    cur = 1
        n_row.append(cur)

    Data = n_row
    if ret_dataFrame:
        Data = pd.DataFrame([n_row], columns=columns)
    
    return Data


def preprocess(row, user_data, movie_data, vocab):
    columns = []
    user_id = row['user_id']
    movie_id = row['item_id']
    rating = row['rating']

    user_row = user_data.loc[user_id - 1] # index start 0
    movie_row = movie_data.loc[movie_id - 1]
    skip_mov = ["movie title", "release date", "video release date", "IMDb URL", "zip_code"]

    assert user_row['user_id'] == user_id, "user id must be the same, but found"
    assert movie_row['movie id'] == movie_id, "movie id must be the same"

    src_row = []
    for col in user_row.index.tolist():
        if col in skip_mov:
            continue
        columns.append(col)
        src_row.append(user_row[col])

    for col in movie_row.index.tolist():
        if col in skip_mov:
            continue
        columns.append(col)
        src_row.append(movie_row[col])
    columns.append('rating')
    src_row.append(rating)
    res = pd.DataFrame([src_row], columns=columns)
    return res


class RecDataset(Dataset):
    def __init__(
        self,
        data,
        vocab,
        s_col=None,
        ignor_col=None,
        device=torch.device("cpu")
    ):
        self.vocab = vocab
        self.device = device
        
        columns = []
        for col in data.columns.tolist():
            if col in vocab.columns:
                continue
            columns.append(col)
        
        for key, val in vocab.word2index.items():
            columns.append(key)

        Data = []
        for row in data.iterrows():
            n_row = one_hot(row[1], self.vocab, ret_dataFrame=False)
            Data.append(n_row)
        
        Data = pd.DataFrame(Data, columns=columns)
        
        self.data = Data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.loc[idx]
        return _preprocess(row, self.device)
        

def _preprocess(row, device):
    src_row = []
    columns = row.index.tolist()
    for col in columns:
        if col == 'rating':
            continue
        src_row.append(row[col])

    trg_row = [row['rating']]

    return torch.tensor(src_row, device=device, dtype=torch.float), torch.tensor(trg_row, device=device, dtype=torch.float)