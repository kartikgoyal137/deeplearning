# Copy-task evaluation

Train and evaluate Mamba variants on the copy task of Jelassi et al. 2024 ("Repeat After Me"), the
same task used by `cot.pdf` (Ren et al.). The code is vendored from
[sjelassi/transformers_ssm_copy](https://github.com/sjelassi/transformers_ssm_copy) (`synthetic_exps/`,
MIT, see `LICENSE`) and changed as little as possible, so results stay comparable to the paper.

```
python main.py --train_task copy --eval_task copy --model mamba      # reference defaults
python sanity_checks.py                                              # harness self-tests
```

Dependencies: `torch numpy tqdm transformers accelerate` (no `mamba-ssm` / `causal-conv1d` needed).

## Task (unchanged from upstream)
- Vocabulary: letters `a..` (`--vocab_size` <= 26) + `$` BOS, `|` separator, `.` EOS, `*` pad.
- Example: `$abcde|abcde.`; the loss is computed on the answer and EOS only.
- Training: string lengths uniform in `[min_train_len, max_train_len]`, packed into `context_len` tokens.
- Evaluation: teacher-forced single forward pass + argmax. `str_acc` = exact-match of the whole
  string, `char_acc` = fraction of correct characters. Both are reported per length.
- Optimiser: AdamW (wd 0.1), linear schedule with 100 warm-up steps, grad-clip 1.0, fp32.

## Differences from upstream
| File | Change | Why |
|---|---|---|
| `data_utils.py`, `LICENSE` | none | |
| `train_utils.py` | `'cuda'` -> `accelerator.device`; take `logits[0]` for any tuple output | no hard-coded GPU; lets new models plug in |
| `test_utils.py` | `'cuda'` -> the model's device; same tuple handling | same |
| `model_utils.py` | `mamba` is our `MambaLM` (`../mamba_model.py`) instead of `mamba_ssm.MambaLMHeadModel`; vocab padded to a multiple of 8; `init_like_mamba_ssm` | `mamba_ssm` needs custom CUDA kernels (see below) |
| `main.py` | only `mamba` in `--model`; unused imports removed (wandb, datasets, ...); added `--seed` and `--grad_checkpoint` | the other models' code is not vendored; reproducibility |

## Is our Mamba the same baseline as `mamba_ssm`?
Forward maths: yes. Same S6 recurrence (ZOH `A`, `B_bar ~ dt*B`), `E=2`, `d_conv=4`, `dt_rank=ceil(D/16)`,
RMSNorm (eps 1e-5), pre-norm residual blocks, tied embeddings, no bias in projections. The chunked scan
is tested against the literal recurrence. The custom CUDA kernels only make the same computation faster.

**Initialisation was different**, and it affects how fast Mamba learns copy. `MambaLM` draws every Linear weight
from N(0, 0.02) (which also overwrote the `dt_proj` init); `mamba_ssm` keeps PyTorch's default
Linear init, uses `U(+-dt_rank^-0.5)` for `dt_proj`, divides `out_proj` by `sqrt(n_layer)` and uses
N(0, 0.02) for the embedding only. `model_utils.init_like_mamba_ssm` re-initialises `MambaLM`
the `mamba_ssm` way, and `sanity_checks.py` verifies it. `mamba_model.py` itself is untouched.
When you add the growing-state variant, build it with the same init so only the state mechanism differs.

Not replicated: the fused add+norm kernel and bf16/fp32 residual mixed-precision details (we run fp32).

## Known quirks of the reference code (kept as is)
- `evaluation()` loops over `np.arange(min_eval_len, max_eval_len)`, so **`max_eval_len` is excluded**. To
  evaluate a single length N use `--min_eval_len N --max_eval_len N+1`.
- `--eval_context_len` is parsed but unused; the eval set uses `--context_len`.
- `train()` runs `steps + 1` updates and ignores `--epochs` for the loop (only the LR schedule uses it).

## Adding a model
Register it in `model_utils.get_model` and add its name to `--model` choices in `main.py`. It must map
`ids (B, L) -> logits (B, L, V)` (or a tuple whose first element is the logits).
