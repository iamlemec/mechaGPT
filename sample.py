# sample from a GPT model

import os
import fire
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import tiktoken

from util import load_config
from model import GPTConfig, GPT

def sample(dataset, config=None, **kwargs):
    # load in default + config
    cfg = load_config(config, **kwargs)

    # get true outdir
    dat_dir = Path('datasets') / dataset
    out_dir = Path('output') / dataset

    # initialize torch config
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # set up autocasting context for cuda
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg.dtype]
    ctx = nullcontext() if cfg.device == 'cpu' else torch.amp.autocast(device_type=cfg.device, dtype=ptdtype)

    # create model object
    checkpoint = torch.load(out_dir/'ckpt.pt', map_location=cfg.device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # load trimmed parameters
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # go to eval on device with autocast 
    model.eval()
    model.to(cfg.device)

    # load metadata or use gpt-2 as fallback
    meta_path = dat_dir / 'meta.pkl'
    if os.path.exists(meta_path):
        print(f'Loading meta from {meta_path}...')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        # create simple encoder/decoder
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print('No meta.pkl found, assuming GPT-2 encodings...')
        enc = tiktoken.get_encoding('gpt2')
        encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
        decode = lambda l: enc.decode(l)

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
