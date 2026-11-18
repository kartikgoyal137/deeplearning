import pathlib
import sys

import torch.nn as nn

# the models (standard Mamba, var1, var2) and the mamba_ssm-matching init live in <repo>/models
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from models.build import build_model


class Mamba(nn.Module):
    """Same interface as upstream's GPT: token ids (B, L) -> logits (B, L, vocab)."""

    def __init__(self, args):
        super().__init__()
        self.lm = build_model(args.model, args.vocab, args.dmodel, args.num_layer, args.state_dim,
                              args.state_schedule, args.schedule, args.grad_checkpoint, args.compile)

    def forward(self, idx):
        return self.lm(idx)[0]    # the models return (logits, loss)


def get_model(args):
    return Mamba(args)
