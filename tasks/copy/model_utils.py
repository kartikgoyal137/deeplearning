import pathlib
import sys

# the models live in <repo>/models
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from models.build import build_model, init_like_mamba_ssm  # noqa: E402,F401  (re-exported for tests)


def get_model(args, tokenizer):
    # mamba_ssm pads the vocab to a multiple of 8 (pad_vocab_size_multiple=8)
    vocab_size = len(tokenizer)
    vocab_size += (-vocab_size) % 8
    return build_model(args.model, vocab_size, args.hidden_size, args.layers, args.state_dim,
                       args.state_schedule, args.schedule, args.grad_checkpoint, args.compile)
