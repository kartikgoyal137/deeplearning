"""Sanity checks for the copy-task harness. Run:  python sanity_checks.py

1. data      - example format `$abc|abc.` and that the loss mask covers exactly `answer + '.'`
2. metrics   - get_score / evaluation() give 1.0 for a perfect predictor and ~0 for a random one
3. model     - init matches mamba_ssm, vocab padding, chunked scan == literal recurrence at a
               length that is not a multiple of the chunk size, and the model is causal
               (the reference evaluates with one teacher-forced pass, so a leak would inflate accuracy)
"""
import contextlib
import io
import math
import re
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

from data_utils import get_tokenizer, get_train_dataset, get_eval_dataset
from model_utils import get_model, init_like_mamba_ssm
from test_utils import evaluation, get_score
from mamba_model import (MambaConfig, MambaLM, SelectiveSSM,
                         selective_scan_chunked, selective_scan_seq)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

args = Namespace(
    train_task="copy", eval_task="copy", vocab_size=26, n_gram=0, length_answer=0,
    model="mamba", hidden_size=64, layers=2, state_dim=16, grad_checkpoint=False,
    train_batch_size=4, eval_batch_size=8, eval_num_batches=3,
    min_train_len=5, max_train_len=20, min_eval_len=10, max_eval_len=20,
    context_len=220, eval_context_len=220,
)
np.random.seed(0); torch.manual_seed(0)
tokenizer, TO_TOKEN, TO_CHAR = get_tokenizer(args)


def ok(msg):
    print(f"  [ok] {msg}")


# ---------------------------------------------------------------------------------- 1. data
print("1. data")
assert len(tokenizer) == 30
ev = get_eval_dataset(args, tokenizer, TO_TOKEN, target_min_len=12, target_max_len=12)
b = ev[0]
for s, ids, m in zip(b["input"], b["input_ids"], b["mask"]):
    assert re.fullmatch(r"\$([a-z]{12})\|\1\.", s), s
    assert tokenizer.decode(ids) == s + "*" * (220 - len(s))
    assert tokenizer.decode(ids[m == 1]) == s.split("|")[1], (s, tokenizer.decode(ids[m == 1]))
ok("eval batch: `$str|str.`, padded with `*`, mask selects exactly `answer.`")

tr = get_train_dataset(args, tokenizer)[0]
for s, m in zip(tr["input"], tr["mask"]):
    assert len(s) == 220 and len(m) == 220
    expected, pos = [], 0
    for mt in re.finditer(r"\$([a-z]+)\|([a-z]+)\.", s):
        if mt.end() > 220:
            break
        n = len(mt.group(1))
        expected += [0] * (n + 2) + [1] * (n + 1)
        pos = mt.end()
    assert m[:pos].tolist() == expected, "mask mismatch"
    assert m[pos:].sum() == 0                       # truncated tail is masked out
ok("train batch: packed to 220 tokens; mask = 1 only on each answer + EOS, truncated tail masked")


# -------------------------------------------------------------------------------- 2. metrics
print("2. metrics")
x = get_eval_dataset(args, tokenizer, TO_TOKEN, 15, 15)[0]["input_ids"]
perfect = torch.roll(x, -1, dims=1)                 # pred[t] = token t+1  (teacher forced)
for i in range(len(x)):
    assert get_score(args, tokenizer, x, perfect, i) == (1, 1.0)
ok("perfect prediction -> str_acc 1, char_acc 1.0")

wrong = perfect.clone()
j = tokenizer.decode(x[0]).index("|") + 3           # corrupt one answer character
wrong[0, j] = (wrong[0, j] + 1) % 26
s_acc, c_acc = get_score(args, tokenizer, x, wrong, 0)
assert s_acc == 0 and abs(c_acc - 14 / 15) < 1e-9, (s_acc, c_acc)
ok("one wrong character -> str_acc 0, char_acc 14/15")


class Oracle(nn.Module):                            # next-token oracle, returns (logits,) like MambaLM
    def __init__(self, V):
        super().__init__(); self.V = V; self.p = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        lg = torch.zeros(*x.shape, self.V, device=x.device)
        lg.scatter_(2, torch.roll(x, -1, 1).unsqueeze(-1), 1.0)
        return (lg,)


class RandomModel(Oracle):
    def forward(self, x):
        return (torch.randn(*x.shape, 26, device=x.device),)


def run_eval(m):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return evaluation(args, m, tokenizer, TO_TOKEN)

for name, m, lo, hi in [("oracle", Oracle(len(tokenizer)).to(DEVICE), 1.0, 1.0),
                        ("random", RandomModel(26).to(DEVICE), 0.0, 0.1)]:
    mean, std, char = run_eval(m)
    assert len(mean) == args.max_eval_len - args.min_eval_len        # lengths 10..19 (upstream arange)
    assert all(lo <= v <= hi for v in mean), (name, mean)
    ok(f"evaluation() with {name} model: str_acc in [{lo}, {hi}] for all {len(mean)} lengths "
       f"(char_acc {min(char):.2f}..{max(char):.2f})")


# ---------------------------------------------------------------------------------- 3. model
print("3. model")
model = get_model(args, tokenizer).to(DEVICE)
cfg = model.cfg
assert cfg.vocab_size == 32 and model.embedding.weight.shape[0] == 32
assert model.lm_head.weight is model.embedding.weight
ok("vocab padded 30 -> 32 (mamba_ssm pad_vocab_size_multiple=8), embeddings tied")

ssm = model.layers[0].mixer.ssm
r = cfg.dt_rank
assert ssm.dt_proj.weight.abs().max() <= r ** -0.5 + 1e-6 and ssm.dt_proj.weight.std() > 0.3 * r ** -0.5
dt = torch.nn.functional.softplus(ssm.dt_proj.bias)
assert dt.min() >= 1e-4 * 0.99 and dt.max() <= 0.11
ok(f"dt_proj.weight ~ U(+-{r ** -0.5:.3f}) (not N(0, 0.02)); softplus(bias) in [{dt.min():.1e}, {dt.max():.1e}]")

op = model.layers[0].mixer.out_proj.weight
bound = 1 / math.sqrt(op.shape[1]) / math.sqrt(cfg.n_layer)
assert op.abs().max() <= bound + 1e-6
assert abs(model.embedding.weight.std().item() - 0.02) < 0.004
assert (-torch.exp(ssm.A_log) < 0).all() and torch.allclose(ssm.D, torch.ones_like(ssm.D))
ok(f"out_proj ~ kaiming_uniform / sqrt(n_layer) (|w| <= {bound:.4f}); embedding std 0.02; A<0, D=1")

L = 219                                              # 219 = 27 * 8 + 3, not a multiple of scan_chunk
g = torch.Generator(device=DEVICE).manual_seed(0)
mk = lambda *s: torch.randn(*s, device=DEVICE, generator=g)
u, dl, Bm, Cm = mk(2, L, 32), torch.nn.functional.softplus(mk(2, L, 32)), mk(2, L, 16), mk(2, L, 16)
A = -torch.exp(mk(32, 16)); Dd = mk(32); z = mk(2, L, 32)
err = (selective_scan_chunked(u, dl, A, Bm, Cm, Dd, z, chunk=8) - selective_scan_seq(u, dl, A, Bm, Cm, Dd, z)).abs().max()
assert err < 1e-3, err
ok(f"chunked scan == literal recurrence at L={L} (max err {err:.1e})")

model.eval()
idx = torch.randint(0, 26, (2, L), device=DEVICE)
with torch.no_grad():
    base = model(idx)[0]
    idx2 = idx.clone(); idx2[:, 100] = (idx2[:, 100] + 1) % 26
    pert = model(idx2)[0]
assert torch.equal(base[:, :100], pert[:, :100]) and not torch.equal(base[:, 100:], pert[:, 100:])
ok("causal: changing token 100 leaves logits at positions < 100 untouched, changes later ones")

print("\nall checks passed")
