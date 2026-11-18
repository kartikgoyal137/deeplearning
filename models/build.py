"""Builds the three compared models, all initialised like mamba_ssm. Shared by tasks/copy and tasks/cot."""
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint  # MambaLM.grad_checkpoint uses torch.utils.checkpoint

from models import mamba, mamba_var1
from models.mamba import MambaConfig, MambaLM
from models.mamba_var2 import GrowingMambaLM, parse_schedule


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
        if hasattr(m, "dt_proj"):                     # every selective SSM layer (standard, var1, var2)
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


def build_model(name, vocab_size, d_model, n_layer, d_state, state_schedule, schedule, grad_checkpoint, compile):
    """The three models that are compared, all initialised like mamba_ssm .

    mamba      standard Mamba, state size d_state in every layer
    mamba_var1 var1: state size varies with depth (state_schedule: "constant" | "linear", mean N = d_state)
    mamba_var2 var2: state size grows along the sequence (schedule, e.g. "0:8,16:16,32:32,64:64"; last N is the cap)
    """
    kw = dict(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer)
    if name == "mamba":
        model = MambaLM(MambaConfig(d_state=d_state, **kw))
        scan_module = mamba
    elif name == "mamba_var1":
        scan_module = mamba_var1
        model = scan_module.MambaLM(scan_module.MambaConfig(d_state=d_state, state_schedule=state_schedule, **kw))
    elif name == "mamba_var2":
        schedule = parse_schedule(schedule)
        model = GrowingMambaLM(MambaConfig(d_state=max(n for _, n in schedule), **kw), schedule)
        scan_module = None
    else:
        raise ValueError(f"unknown model '{name}'")
    if compile:
        assert scan_module is not None, "--compile is not implemented for mamba_var2 yet"
        scan_module.compile_scan()                    # fused scan: same maths, faster
    init_like_mamba_ssm(model)
    model.grad_checkpoint = grad_checkpoint           # recompute activations: same maths, less memory
    return model

