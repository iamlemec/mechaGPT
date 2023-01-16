# general utilities

import toml

class Bundle:
    def __init__(self, *args, **kwargs):
        for d in args + (kwargs,):
            self.__dict__.update(d)

    def __repr__(self):
        return '\n'.join([
            f'{k} = {v}' for k, v in self.__dict__.items()
        ])

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
