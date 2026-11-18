import importlib.util
import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint  # MambaLM.grad_checkpoint uses torch.utils.checkpoint

# our pure-PyTorch Mamba (standard, var2) lives one directory up, var1 in ../var1
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba_model import MambaConfig, MambaLM
from mamba_growing import GrowingMambaLM, parse_schedule


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


def load_var1():
    """var1 (depth-varying state, kept untouched in ../var1/) has the same module name as the
    standard model, so it is loaded under another name."""
    spec = importlib.util.spec_from_file_location("mamba_var1", ROOT / "var1" / "mamba_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mamba_var1"] = module
    spec.loader.exec_module(module)
    return module


def build_model(name, vocab_size, d_model, n_layer, d_state, state_schedule, schedule, grad_checkpoint, compile):
    """The three models that are compared, all initialised like mamba_ssm (shared by copy_eval and cot_eval).

    mamba      standard Mamba, state size d_state in every layer
    mamba_var1 var1: state size varies with depth (state_schedule: "constant" | "linear", mean N = d_state)
    mamba_var2 var2: state size grows along the sequence (schedule, e.g. "0:8,16:16,32:32,64:64"; last N is the cap)
    """
    kw = dict(vocab_size=vocab_size, d_model=d_model, n_layer=n_layer)
    if name == "mamba":
        model = MambaLM(MambaConfig(d_state=d_state, **kw))
        scan_module = sys.modules["mamba_model"]
    elif name == "mamba_var1":
        scan_module = load_var1()
        model = scan_module.MambaLM(scan_module.MambaConfig(d_state=d_state, state_schedule=state_schedule, **kw))
    elif name == "mamba_var2":
        schedule = parse_schedule(schedule)
        model = GrowingMambaLM(MambaConfig(d_state=max(n for _, n in schedule), **kw), schedule)
        scan_module = None
    else:
        raise ValueError(f"unknown model '{name}'")
    if compile:
        assert scan_module is not None, "--compile is not implemented for mamba_var2 yet"
        torch._dynamo.config.cache_size_limit = 32    # --same_len gives a few context lengths; compile each one
        scan_module.compile_scan()                    # fused scan: same maths, faster
    init_like_mamba_ssm(model)
    model.grad_checkpoint = grad_checkpoint           # recompute activations: same maths, less memory
    return model


def get_model(args, tokenizer):
    # mamba_ssm pads the vocab to a multiple of 8 (pad_vocab_size_multiple=8)
    vocab_size = len(tokenizer)
    vocab_size += (-vocab_size) % 8
    return build_model(args.model, vocab_size, args.hidden_size, args.layers, args.state_dim,
                       args.state_schedule, args.schedule, args.grad_checkpoint, args.compile)


def describe(model):
    """One line saying which model was really built (the standard model and var1 are both files called mamba_model.py)."""
    ssm = model.layers[0].mixer.ssm
    if hasattr(model, "schedule"):
        sizes = f"state-size schedule {model.schedule}"
    else:
        sizes = f"state size per layer {[getattr(l.mixer.ssm, 'd_state', model.cfg.d_state) for l in model.layers]}"
    return f"{type(model).__module__}.{type(model).__name__} with {type(ssm).__name__}, {sizes}, {sum(p.numel() for p in model.parameters()):,} parameters"


def optimizer_groups(model, weight_decay):
    """AdamW parameter groups as in mamba_ssm training: no weight decay on biases, norm weights, A_log and D.
    (Decaying A_log pulls every forgetting speed towards A = -1 and erases the multi-timescale init.)"""
    flagged = lambda p: p.ndim < 2 or getattr(p, "_no_weight_decay", False)
    return [{"params": [p for p in model.parameters() if not flagged(p)], "weight_decay": weight_decay},
            {"params": [p for p in model.parameters() if flagged(p)], "weight_decay": 0.0}]


# settings that decide the model, per --model (a var2 checkpoint has the same shapes as a plain model with N = its cap)
MODEL_SETTINGS = {"mamba": ["state_dim"], "mamba_var1": ["state_dim", "state_schedule"], "mamba_var2": ["schedule"]}


def save_with_args(model, args, path):
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, path)


def load_with_args(model, args, path, size_keys):
    """Load a checkpoint after checking it was trained with the same model settings. size_keys: the width / depth / vocab argument names."""
    ckpt = torch.load(path, map_location="cpu")
    for key in ["model"] + list(size_keys) + MODEL_SETTINGS[args.model]:
        assert ckpt["args"].get(key) == getattr(args, key), \
            f"checkpoint was trained with {key}={ckpt['args'].get(key)!r}, but {key}={getattr(args, key)!r} was given"
    model.load_state_dict(ckpt["state_dict"])
