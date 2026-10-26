# CoT evaluation (Longest Increasing Subsequence)

Train and test Mamba variants on the LIS task with and without chain-of-thought (CoT), the
setup `cot.pdf` (Ren et al.) takes from Feng et al. 2024, "Towards Revealing the Mystery behind
Chain of Thought". The code is vendored from [guyuntian/CoT_benchmark](https://github.com/guyuntian/CoT_benchmark)
(MIT, see `LICENSE`); only the `LIS` task is included. Only the model is replaced.

## Task
Each example is an integer sequence of length L.
- **Direct:** `x1 ... xL <sep> answer` - the model must output the LIS length straight away.
- **CoT (`--chain`):** `x1 ... xL <sep> c1 ... cL <sep> answer` - `ci` is the LIS ending at position `i`,
  so the answer is built step by step.

Loss and accuracy use only the tokens after the first `<sep>`. Test accuracy is teacher-forced: a sequence
counts as correct only if **every** token after the first `<sep>` (steps and answer) is predicted correctly.

## Usage (one GPU, run from this folder)
```
python LIS/data.py --file Data --length 50 --train_size 1e6 --test_size 1e5 --number_range 250

torchrun --standalone --nproc_per_node=1 train.py --file Data --folder LIS --output_dir out \
  --maxlen 105 --maxdata 105 --vocab 253 --num_range 250 --weight_decay 0.01 --learning_rate 1e-4 \
  --batch_size 16 --epoch 100 --warmup 5 --dmodel 256 --num_layer 6 --chain

torchrun --standalone --nproc_per_node=1 test.py --file Data --folder LIS --maxlen 105 --maxdata 105 \
  --vocab 253 --num_range 250 --batch_size 16 --dmodel 256 --num_layer 6 --chain --model_path out/epoch_100.pt
```
For the direct version drop `--chain` and use `--maxlen`/`--maxdata` = L + 4. `MAXLEN` / `VOCAB` values per task
length are in upstream's `assets/TRAIN.md` (LIS: len50 -> 54 direct / 105 CoT, vocab 253). Needs `torch`,
`tensorboard`, `transformers`, `tqdm`, `numpy`.

## Differences from upstream
| File | Change | Why |
|---|---|---|
| `train.py`, `test.py` | `GPT(args)` -> `get_model(args)` | use Mamba instead of the minGPT transformer |
| `model_utils.py` (new) | `Mamba` wrapper around our `MambaLM`, initialised like `mamba_ssm` | see `../copy_eval/README.md` for why the init matters |
| `LIS/dataloader.py` | `num_workers=10` -> `0` | the data is already in memory; Python 3.14's default worker start method re-imports `train.py` and crashes |
| `model.py` | removed | transformer baseline is not part of this harness |
| everything else | unchanged | |

## Notes for runs
- Mamba has no dropout, so `--drop` and `--head` are ignored. The paper uses roughly 2 Mamba layers per transformer
  layer at the same `d` (d=256 -> `--num_layer 6`).
- **Memory:** our pure-PyTorch scan uses far more GPU memory than the fused CUDA kernel. At d=256, 6 layers,
  length 104, a batch of 16 needs ~3.1 GiB and 32 does not fit in 6 GB. Upstream's batch is 128 per GPU x 4 GPUs = 512.
- Upstream's `arithmetic` task evaluates by greedy generation and needs a `generate()` method, so it is not included.
