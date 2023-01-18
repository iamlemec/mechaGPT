# train a GPT model

import time
import math
import fire

import numpy as np
import torch

from util import load_config, init_torch, ensure_checkpoint
from data import load_dataset, get_batch
from model import GPTConfig, GPT

def train(config=None, **kwargs):
    # load in default + config
    cfg = load_config(config, **kwargs)

    # ensure output dir exists (ignores None)
    load = ensure_checkpoint(cfg.load, should_exist=True)
    save = ensure_checkpoint(cfg.save, make_dir=True)

    # initialize torch and get autocast context
    ctx = init_torch(cfg.device, cfg.dtype, cfg.seed)

    # dataloader that could be make seed'able if wanted
    data, idx_train, idx_valid = load_dataset(cfg.datadir, encoding=cfg.encoding, valid_frac=cfg.valid_frac)
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

    # initialize parameters somehow
    if load is not None:
        print(f'resuming training from {load}')
        checkpoint = torch.load(load, map_location=cfg.device)

        # ensure model_args line up
        for k, v in model_args.items():
            assert checkpoint['model_args'][k] == v

        # create model
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

        # restore optimization state
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif cfg.init.startswith('gpt2'):
        print(f'initializing from OpenAI GPT-2 weights: {cfg.init_from}')
        override_args = {'dropout': cfg.dropout}
        model = GPT.from_pretrained(cfg.init, override_args)

        # read off and override the GPT sizing model args from the model config
        model_args['n_layer'] = model.config.n_layer
        model_args['n_head'] = model.config.n_head
        model_args['n_embd'] = model.config.n_embd
    elif cfg.init == 'scratch':
        print('initializing a new model from scratch')
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    # crop down the model block size if desired
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)

    # print out model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {n_params/1e6:.2f}M')

    # send to proper device
    model.to(cfg.device)

    # compile the model
    if cfg.compile:
        print('compiling model...')
        model = torch.compile(model) # requires PyTorch 2.0

    # compare training and validation loss
    @torch.no_grad()
    def estimate_loss(split):
        total = 0.0
        model.eval()
        for _ in range(cfg.eval_iters):
            X, Y = batch(split)
            with ctx:
                _, loss = model(X, Y)
            total += loss.item()
        model.train()
        return total/cfg.eval_iters

    # save model checkpoint
    def save_checkpoint(iter_num, valid_loss):
        print(f'saving checkpoint to {save}')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': valid_loss,
            'config': cfg.to_dict(),
        }
        torch.save(checkpoint, save)

    # optimizer
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2))
    if load is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(iter):
        if iter < cfg.warmup_iters:
            # linear warmup for warmup_iters steps
            return cfg.learning_rate*(iter/cfg.warmup_iters)
        else:
            # use cosine decay down to min learning rate
            decay0 = (iter-cfg.warmup_iters)/(cfg.lr_decay_iters-cfg.warmup_iters)
            decay = np.clip(decay0, 0, 1)
            coeff = 0.5*(1.0+math.cos(math.pi*decay))
            return cfg.min_lr + coeff*(cfg.learning_rate-cfg.min_lr)

    # training loop
    while True:
        t0 = time.time()

        # determine the learning rate for this iteration
        if cfg.decay_lr:
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # evalulate validation loss and possibly save checkpoint
        if iter_num % cfg.eval_interval == 0:
            train_loss, valid_loss = estimate_loss('train'), estimate_loss('valid')
            print(f'step {iter_num}: train loss {train_loss:.4f}, val loss {valid_loss:.4f}')
            if iter_num > 0 and (valid_loss < best_val_loss or cfg.always_save_checkpoint):
                save_checkpoint(iter_num, valid_loss)
        if iter_num == 0 and cfg.eval_only:
            break

        # compute training loss
        X, Y = batch('train')
        with ctx:
            _, loss = model(X, Y)

        # accumualte gradients
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # print out training progress
        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            print(f'iter {iter_num}: loss {loss.item():.4f}, time {1000*dt:.2f}ms')

        # termination conditions
        iter_num += 1
        if iter_num > cfg.max_iters:
            break

# create interface
if __name__ == '__main__':
    fire.Fire(train)
