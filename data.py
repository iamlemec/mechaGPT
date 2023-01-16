# data tools for GPT

import fire
import pickle
import torch
import numpy as np
from pathlib import Path

# convert text to numeric data
def prepare(dataset, verbose=True):
    # use pathlib
    path = Path('datasets') / dataset

    # load text data
    with open(path/'input.txt', 'r') as f:
        data = f.read()

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # encode to integers
    data_toks = [stoi[c] for c in data]

    # export to bin file
    np.array(data_toks, dtype=np.uint16).tofile(path/'tokens.bin')

    # save the meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(path/'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    # output diagnostic info
    if verbose:
        print('length of dataset in characters: ', len(data))
        print('all the unique characters:', ''.join(chars))
        print('vocab size:', vocab_size)
        print(f'data has {len(data_toks)} tokens')

def get_generator(seed):
    return torch.manual_seed(seed) if seed is not None else None

# create sequences and split into training and validation
def load_dataset(dataset, valid_frac=0.1, seed=None):
    # use pathlib
    path = Path('datasets') / dataset

    # use random seed
    gen = get_generator(seed)

    # load data file
    data = np.memmap(path/'tokens.bin', dtype=np.uint16, mode='r')
    data = torch.from_numpy(data.astype(np.int64))

    # get output size
    N = len(data)
    V = int(valid_frac*N)

    # generate split indices
    indices = torch.randperm(N, generator=gen)
    idx_train, idx_valid = indices[:-V], indices[-V:]

    return data, idx_train, idx_valid

# return randomly shuffled sequence data
def get_batch(data, indices, batch_size, block_size, seed=None):
    gen = get_generator(seed)

    # generate random indices
    max_index = len(indices) - block_size
    idx = torch.randint(max_index, (batch_size,), generator=gen)

    # select input and output data
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+1+block_size] for i in idx])

    return x, y

# create interface
if __name__ == '__main__':
    fire.Fire(prepare)
