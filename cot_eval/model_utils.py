import pathlib
import sys

import torch.nn as nn

# our pure-PyTorch Mamba and the mamba_ssm-matching init (see ../copy_eval/README.md)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mamba_model import MambaConfig, MambaLM
from copy_eval.model_utils import init_like_mamba_ssm


class Mamba(nn.Module):
    """Same interface as upstream's GPT: token ids (B, L) -> logits (B, L, vocab)."""

    def __init__(self, args):
        super().__init__()
        cfg = MambaConfig(vocab_size=args.vocab, d_model=args.dmodel, n_layer=args.num_layer)
        self.lm = MambaLM(cfg)
        init_like_mamba_ssm(self.lm)

    def forward(self, idx):
        return self.lm(idx)[0]    # MambaLM returns (logits, loss)


def get_model(args):
    return Mamba(args)
