# general utilities

import toml
import torch
from contextlib import nullcontext
from pathlib import Path

# simple addressable dict
class Bundle:
    def __init__(self, *args, **kwargs):
        for d in args + (kwargs,):
            self.__dict__.update(d)

    def __repr__(self):
        return '\n'.join([
            f'{k} = {v}' for k, v in self.__dict__.items()
        ])

    def __getattr__(self, k):
        return self.__dict__.get(k, None)

    def subset(self, keys):
        return Bundle({k: self.__dict__[k] for k in keys})

    def to_dict(self):
        return self.__dict__.copy()

# load config info
def load_config(config=None, **kwargs):
    with open('config/default.toml') as f:
        default = toml.load(f)
    if config is not None:
        with open(config) as f:
            extra = toml.load(f)
    else:
        extra = {}
    return Bundle(default, extra, **kwargs)

# torch setup and context
def init_torch(device, dtype, seed):
    # initialize torch config
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # set up autocasting context for cuda
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    return ctx

# do a couple different path related things
# only checks is path is not None
def ensure_checkpoint(path, make_dir=False, should_exist=False):
    if path is None:
        return None
    path = Path(path)
    if should_exist and not path.exists():
        raise Exception('Checkpoint "{path}" not found')
    if path.is_dir():
        direc, name = path, 'ckpt.pt'
        path = direc / name
    else:
        direc, name = path.parent, path.name
    if make_dir and not direc.exists():
        direc.mkdir(parents=True)
    return path
