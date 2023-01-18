# sample from a GPT model

import fire
import pickle
from pathlib import Path

import numpy as np
import torch
import tiktoken

from util import load_config, init_torch, ensure_checkpoint
from model import GPTConfig, GPT

def sample(config=None, **kwargs):
    # load in default + config
    cfg = load_config(config, **kwargs)

    # get true outdir
    datadir = Path(cfg.datadir) if cfg.datadir is not None else None
    ckpt = ensure_checkpoint(cfg.checkpoint, should_exist=True)

    # initialize torch and get autocast context
    ctx = init_torch(cfg.device, cfg.dtype, cfg.seed)

    # create model object
    checkpoint = torch.load(ckpt, map_location=cfg.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # load trimmed parameters - need to hack out weird prefixes
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # go to eval on device with autocast 
    model.eval()
    model.to(cfg.device)

    # load metadata or use gpt-2 as fallback
    if cfg.encoding == 'char':
        # load in metadata pickle
        meta_path = datadir / 'char_meta.pkl'
        print(f'Loading meta from {meta_path}...')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        # create simple encoder/decoder
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    elif cfg.encoding == 'gpt2':
        # use standard gpt2 encoding
        encoding = tiktoken.get_encoding('gpt2')
        encode = lambda s: encoding.encode(s, allowed_special={'<|endoftext|>'})
        decode = lambda l: encoding.decode(l)

    # encode the beginning of the prompt
    start_ids = encode(cfg.start)
    x = torch.tensor(start_ids, dtype=torch.long, device=cfg.device)[None, ...]

    # run generation
    for k in range(cfg.num_samples):
        with ctx:
            y = model.generate(x, cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
        print(decode(y[0].tolist()))
        print('---------------')

# create interface
if __name__ == '__main__':
    fire.Fire(sample)
