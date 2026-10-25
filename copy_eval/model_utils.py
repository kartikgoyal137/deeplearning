import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint  # MambaLM.grad_checkpoint uses torch.utils.checkpoint

# our pure-PyTorch Mamba lives one directory up (shared with the train/inference notebooks)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mamba_model import MambaConfig, MambaLM, SelectiveSSM


def init_like_mamba_ssm(model: MambaLM, initializer_range: float = 0.02):
    """Re-initialise `model` exactly the way `mamba_ssm.MambaLMHeadModel` initialises itself.

    `MambaLM` (used for the Shakespeare notebooks) draws *every* Linear weight from
    N(0, 0.02), which also overwrites the dt_proj init. `mamba_ssm` does not: it leaves Linear
    weights at the PyTorch default, uses a uniform dt_proj init, rescales out_proj by
    1/sqrt(n_layer) and only uses N(0, 0.02) for the embedding. Initialisation changes how fast
    Mamba learns copy, so the baseline must match the reference implementation.

    Reference: mamba_ssm/models/mixer_seq_simple.py (`_init_weights`) and
    mamba_ssm/modules/mamba_simple.py (`Mamba.__init__`).
    """
    n_layer = model.cfg.n_layer

    # 1. nn.Linear: PyTorch default weight init; biases zeroed unless flagged `_no_reinit`
    #    (the dt_proj bias keeps its inverse-softplus init from SelectiveSSM.__init__)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                nn.init.zeros_(m.bias)

    # 2. dt_proj weight: uniform(+-dt_rank^-0.5)   (dt_init="random", dt_scale=1.0)
    for m in model.modules():
        if isinstance(m, SelectiveSSM):
            std = m.cfg.dt_rank ** -0.5
            nn.init.uniform_(m.dt_proj.weight, -std, std)

    # 3. out_proj: rescaled prenorm residual (n_residuals_per_layer = 1, no MLP)
    for name, p in model.named_parameters():
        if name.endswith("out_proj.weight"):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p /= math.sqrt(n_layer)

    # 4. embedding (tied with lm_head): N(0, 0.02)
    nn.init.normal_(model.embedding.weight, std=initializer_range)


def get_model(args, tokenizer):

    if args.model == "mamba":
        # mamba_ssm pads the vocab to a multiple of 8 (pad_vocab_size_multiple=8)
        vocab_size = len(tokenizer)
        vocab_size += (-vocab_size) % 8
        cfg = MambaConfig(
                vocab_size=vocab_size,
                d_model=args.hidden_size,
                n_layer=args.layers,
                d_state=args.state_dim,
                )
        model = MambaLM(cfg)
        init_like_mamba_ssm(model)
        model.grad_checkpoint = args.grad_checkpoint  # recompute activations: same maths, less memory
    else:
        raise ValueError(f"unknown model '{args.model}'; register new models in get_model")

    return model
