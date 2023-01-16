# train a GPT model

import os
import time
import math
import fire
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from util import load_config
from data import load_dataset, get_batch
from model import GPTConfig, GPT

def train(dataset, config=None, **kwargs):
    # load in default + config
    cfg = load_config(config, **kwargs)

    # ensure output dir exists
    out_dir = Path('output') / dataset
    os.makedirs(out_dir, exist_ok=True)

    # initialize torch config
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # set up autocasting context for cuda
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[cfg.dtype]
    ctx = nullcontext() if cfg.device == 'cpu' else torch.amp.autocast(device_type=cfg.device, dtype=ptdtype)

    # dataloader that could be make seed'able if wanted
    data, idx_train, idx_valid = load_dataset(dataset, valid_frac=cfg.valid_frac)
    def batch(split):
        idx = idx_train if split == 'train' else idx_valid
        x, y = get_batch(data, idx, cfg.batch_size, cfg.block_size)
        return x.to(cfg.device), y.to(cfg.device)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model config init
    model_args = cfg.subset([
        'n_layer', 'n_head', 'n_embd', 'block_size', 'dropout'
    ]).to_dict()

    # initialize parameters
    if cfg.init_from == 'scratch':
        # init a new model from scratch
        print('Initializing a new model from scratch')
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif cfg.init_from == 'resume':
        print(f'Resuming training from {cfg.out_dir}')
        # resume training from a checkpoint.
        checkpoint = torch.load(out_dir/'ckpt.pt', map_location=cfg.device)
        checkpoint_model_args = checkpoint['model_args']
        for k, v in model_args.items():
            assert checkpoint_model_args[k] == v, 'for now'
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif cfg.init_from.startswith('gpt2'):
        print(f'Initializing from OpenAI GPT-2 weights: {cfg.init_from}')
        # initialize from OpenAI GPT-2 weights
        override_args = {'dropout': cfg.dropout}
        model = GPT.from_pretrained(cfg.init_from, override_args)
        # read off and override the GPT sizing model args from the model config
        model_args['n_layer'] = model.config.n_layer
        model_args['n_head'] = model.config.n_head
        model_args['n_embd'] = model.config.n_embd

    # crop down the model block size if desired
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)

    # send to proper device
    model.to(cfg.device)

    # optimizer
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2))
    if cfg.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])

    # compile the model
    if cfg.compile:
        model = torch.compile(model) # requires PyTorch 2.0

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(iter):
        # 1) linear warmup for warmup_iters steps
        if iter < cfg.warmup_iters:
            return cfg.learning_rate * iter / cfg.warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # training loop
    t0 = time.time()
    while True:
        # determine the learning rate for this iteration
        if cfg.decay_lr:
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = cfg.learning_rate

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg.to_dict(),
                    }
                    print(f'saving checkpoint to {out_dir}')
                    torch.save(checkpoint, out_dir/'ckpt.pt')
        if iter_num == 0 and cfg.eval_only:
            break

        X, Y = batch('train')
        with ctx:
            logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0:
            lossf = loss.item() # loss as float
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1

        # termination conditions
        if iter_num > cfg.max_iters:
            break

# create interface
if __name__ == '__main__':
    fire.Fire(train)
