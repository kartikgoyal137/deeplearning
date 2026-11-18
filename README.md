# Mamba, two state-size variations, and copy / CoT evaluation

From-scratch, pure-PyTorch Mamba ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) and variations of its state size,
evaluated on the copy task and a chain-of-thought task as in `papers/cot.pdf` (Ren et al., "Exploring the Limitations of
Mamba in COPY and CoT Reasoning").

3 models x 2 tasks:

| `--model` | File | State size N |
|---|---|---|
| `mamba` | `models/mamba.py` | standard: the same N in every layer and at every token |
| `mamba_var1` | `models/mamba_var1.py` | var1: N varies with depth under a fixed total budget (`--state_schedule constant\|linear`) |
| `mamba_var2` | `models/mamba_var2.py` | var2: N grows along the sequence on a schedule, capped at N_max (`--schedule "0:8,16:16,32:32,64:64"`) |

| Task | Harness | Upstream |
|---|---|---|
| COPY | `tasks/copy/` (`main.py`: train + eval) | [sjelassi/transformers_ssm_copy](https://github.com/sjelassi/transformers_ssm_copy) |
| CoT (LIS) | `tasks/cot/` (`LIS/data.py`, `train.py`, `test.py`) | [guyuntian/CoT_benchmark](https://github.com/guyuntian/CoT_benchmark) |

Both harnesses build the models through `models/build.py`, initialised like `mamba_ssm`, so the three models are trained
and tested under identical settings. Options and differences from upstream: `tasks/copy/README.md`, `tasks/cot/README.md`.

## Layout
```
models/          mamba.py, mamba_var1.py, mamba_var2.py, build.py (build_model + mamba_ssm init)
tasks/copy/      copy-task harness
tasks/cot/       LIS (CoT / direct) harness
tests/           test_var2.py, test_copy_harness.py (no training)
notebooks/       mamba/, mamba_var1/: Tiny Shakespeare train / inference notebooks (self-contained:
                 their %%writefile cell writes a local mamba_model.py, identical to models/)
papers/          mamba.pdf, cot.pdf
modal_app.py     training on Modal (see below)
requirements.txt pinned runtime deps (same in the local venv and the Modal image)
```

## Local environment
Uses the venv at `../kaggle/venv` (Python 3.14, `torch`, `transformers`, `accelerate`, `tensorboard`, `modal`, ...):
```
source ../kaggle/venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt      # already installed
python tests/test_var2.py && python tests/test_copy_harness.py
```

## Training on Modal
`modal_app.py` runs the harnesses unchanged on a Modal GPU (default `L40S`, override with `MAMBA_GPU=A100-80GB`). All
outputs go to the Modal volume `mamba-runs`. Full usage is in the docstring at the top of `modal_app.py`.
```
modal setup                                                       # once: log in
modal run modal_app.py::tests                                     # test suites on a GPU
modal run modal_app.py::cot_data --length 50                      # LIS data, once
modal run --detach modal_app.py::copy --models mamba,mamba_var1,mamba_var2
modal run --detach modal_app.py::cot  --models mamba,mamba_var1,mamba_var2 --chain
modal volume get mamba-runs cot/<run> runs/                       # fetch logs / checkpoints
```
`--extra "..."` passes options through to the harness (e.g. `--extra "--steps 4000 --save_every 1000"`). Copy runs use
`--no_pack` for every model by default (required for var2, so the comparison is fair). A Modal function runs for at most
24 h and neither harness resumes from a checkpoint, so size runs to fit.

## Status
- Standard Mamba, var1, var2 and both harnesses are implemented and tested.
- No baseline or variation results yet: the long training runs have not been done.
- var2: each segment of constant N only runs its own N dimensions in the scan (cost benchmarks against constant N are
  still to be done); no recurrent `step()` / `generate()` yet, and `--compile` is not supported for it.
