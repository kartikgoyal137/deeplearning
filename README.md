# Mamba, two state-size variations, and copy / CoT evaluation

From-scratch, pure-PyTorch Mamba ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) and variations of its state size,
evaluated on the copy task and a chain-of-thought task as in `cot.pdf` (Ren et al., "Exploring the Limitations of Mamba in
COPY and CoT Reasoning").

| Path | What |
|---|---|
| `mamba_model.py`, `01_mamba_train.ipynb`, `02_mamba_inference.ipynb` | **Standard Mamba** (fixed state size N in every layer) and its Shakespeare train / inference notebooks |
| `var1/` | **Variation 1** (original, untouched): N varies with depth under a fixed total state budget (`state_schedule`) |
| `mamba_growing.py`, `test_growing.py` | **Variation 2**: N grows along the sequence on a schedule, capped at N_max; `test_growing.py` checks it |
| `copy_eval/` | Copy-task train / eval harness (from [sjelassi/transformers_ssm_copy](https://github.com/sjelassi/transformers_ssm_copy)) |
| `cot_eval/` | CoT / LIS train / eval harness (from [guyuntian/CoT_benchmark](https://github.com/guyuntian/CoT_benchmark)) |
| `docs/UNDERSTANDING.md`, `docs/understanding.html` | **Start here if Mamba is new to you:** a no-background guide to Mamba, var1, var2 and the evaluation code (the `.html` version is interactive; open it in a browser) |
| `mamba.pdf`, `cot.pdf` | The papers |

Both harnesses take `--model mamba | mamba_var1 | mamba_var2`, so the three models are trained and tested under identical
settings (see `copy_eval/README.md` and `cot_eval/README.md` for options and commands).

## Status
- Standard Mamba, var1, var2 and both harnesses are implemented and tested (`copy_eval/sanity_checks.py`, `test_growing.py`).
- No baseline or variation results yet: the long training runs have not been done.
- var2: each segment of constant N only runs its own N dimensions in the scan (cost benchmarks against constant N are
  still to be done); no recurrent `step()` / `generate()` yet, and `--compile` is not supported for it.
