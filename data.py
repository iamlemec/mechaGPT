# data tools for GPT

import fire
import pickle
from pathlib import Path

import numpy as np
import torch
import tiktoken

# convert text to numeric data
def prepare_char(dataset):
    # use pathlib
    path = Path(dataset)

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
    toks_file = path / 'char_tokens.bin'
    np.array(data_toks, dtype=np.uint16).tofile(toks_file)

    # save the meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(path/'char_meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    # output diagnostic info
    print('length of dataset in characters: ', len(data))
    print('all the unique characters:', ''.join(chars))
    print('vocab size:', vocab_size)
    print(f'data has {len(data_toks)} tokens')

# convert text to gpt2 usable
def prepare_gpt2(dataset, num_proc=8):
    # use pathlib
    path = Path(dataset)

    # load text data
    with open(path/'input.txt', 'r') as f:
        data = f.read()

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    encoder = tiktoken.get_encoding('gpt2')
    data_toks = encoder.encode_ordinary(data)
    data_toks.append(encoder.eot_token)

    # export to bin file
    toks_file = path / 'gpt2_tokens.bin'
    np.array(data_toks, dtype=np.uint16).tofile(toks_file)

    # output diagnostic info
    print('length of dataset in characters: ', len(data))
    # print('all the unique characters:', ''.join(chars))
    # print('vocab size:', vocab_size)
    print(f'data has {len(data_toks)} tokens')

def get_generator(seed):
    return torch.manual_seed(seed) if seed is not None else None

# create sequences and split into training and validation
def load_dataset(datadir, encoding='char', valid_frac=0.1, seed=None):
    # use pathlib
    datadir = Path(datadir)

    # use random seed
    gen = get_generator(seed)

    # load data file
    path = datadir / f'{encoding}_tokens.bin'
    data = np.memmap(path, dtype=np.uint16, mode='r')
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
    fire.Fire({
        'prepare_char': prepare_char,
        'prepare_gpt2': prepare_gpt2,
    })
