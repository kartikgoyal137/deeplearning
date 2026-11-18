"""Train / evaluate the three models (mamba, mamba_var1, mamba_var2) on COPY and CoT (LIS) on Modal.

One-time setup (from the repo root, with ../kaggle/venv active):
    modal setup                                   # log in, writes ~/.modal.toml

Commands (add `--detach` after `modal run` for long jobs, so closing the terminal does not stop them):
    modal run modal_app.py::tests                                        # both test suites on a GPU
    modal run modal_app.py::cot_data --length 50                         # generate LIS data once (CPU)
    modal run modal_app.py::copy --models mamba                          # copy task
    modal run modal_app.py::copy --models mamba,mamba_var1,mamba_var2    # all three in parallel
    modal run modal_app.py::cot --models mamba --chain                   # LIS with CoT (drop --chain = direct)
    modal run modal_app.py::cot --models mamba_var2 --chain --extra "--batch_size 64 --epoch 50"

`--extra` is appended verbatim to the harness command line (argparse keeps the last value, so it overrides
the defaults below). Harness options: tasks/copy/README.md, tasks/cot/README.md.

Everything a run writes (logs, checkpoints, tensorboard, eval output) goes to the Modal volume `mamba-runs`:
    /runs/copy/<run>/   train.log, checkpoints/step_*.pt, output_dir/.../model.pt
    /runs/cot/<run>/    train.log, test.log, epoch_*.pt, tensorboard events
    /runs/data/lis/len<L>_range<R>/{chain,decoder}/{train,test}_data.txt
Fetch results:  modal volume ls mamba-runs copy   |   modal volume get mamba-runs copy/<run> runs/

GPU type / timeout are read when the app is launched:  MAMBA_GPU=A100-80GB modal run modal_app.py::cot ...
"""
import os
import pathlib
import shlex
import subprocess
import sys
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent
REMOTE = "/root/project"                     # where the repo code lives in the container
RUNS = "/runs"                               # volume mount point
MODELS = ("mamba", "mamba_var1", "mamba_var2")

GPU = os.environ.get("MAMBA_GPU", "L40S")
TIMEOUT = int(float(os.environ.get("MAMBA_TIMEOUT_H", "24")) * 3600)   # Modal's max is 24 h

requirements = [l.split("#")[0].strip() for l in (ROOT / "requirements.txt").read_text().splitlines()]
image = (
    modal.Image.debian_slim(python_version="3.14")          # same Python as the local venv
    .uv_pip_install(*[r for r in requirements if r])
    .env({"PYTHONUNBUFFERED": "1"})
)
ignore = ["**/__pycache__", "**/output_dir", "**/checkpoints", "**/Data", "**/*.pt"]
for sub in ("models", "tasks", "tests"):                    # code only; notebooks and papers stay local
    image = image.add_local_dir(ROOT / sub, f"{REMOTE}/{sub}", ignore=ignore)

app = modal.App("mamba-state-size", image=image)
volume = modal.Volume.from_name("mamba-runs", create_if_missing=True)


# ------------------------------------------------------------------------------------------- helpers
def _run(cmd, cwd, log_name="train.log"):
    """Run `cmd` in `cwd`, streaming its output to the Modal logs and to cwd/log_name."""
    cwd = pathlib.Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    print("$", shlex.join(cmd), f"   (cwd {cwd})", flush=True)
    with open(cwd / log_name, "a") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        proc.wait()
    volume.commit()
    if proc.returncode:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {shlex.join(cmd)}")


def _lis_dir(length, number_range):
    return f"{RUNS}/data/lis/len{length}_range{number_range}"


def _models(models: str):
    names = [m.strip() for m in models.split(",") if m.strip()]
    bad = [m for m in names if m not in MODELS]
    if bad or not names:
        raise SystemExit(f"--models must be a comma list of {MODELS}, got {models!r}")
    return names


def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")


# ------------------------------------------------------------------------------------ remote functions
@app.function(gpu=GPU, timeout=30 * 60)
def run_tests():
    for test in ("tests/test_var2.py", "tests/test_copy_harness.py"):
        subprocess.run([sys.executable, test], cwd=REMOTE, check=True)


@app.function(cpu=2, timeout=6 * 3600, volumes={RUNS: volume})
def make_lis_data(length: int, train_size: float, test_size: float, number_range: int):
    out = _lis_dir(length, number_range)
    if pathlib.Path(out, "chain", "test_data.txt").exists():
        print(f"{out} already exists, not regenerating")
        return
    _run([sys.executable, f"{REMOTE}/tasks/cot/LIS/data.py", "--file", out, "--length", str(length),
          "--train_size", str(train_size), "--test_size", str(test_size), "--number_range", str(number_range)],
         cwd=out, log_name="data.log")


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={RUNS: volume})
def train_copy(model: str, run: str, args: list[str]):
    cmd = [sys.executable, f"{REMOTE}/tasks/copy/main.py", "--train_task", "copy", "--eval_task", "copy",
           "--model", model, "--ckpt_dir", "checkpoints", *args]
    _run(cmd, cwd=f"{RUNS}/copy/{run}")      # train + eval in one go (main.py); output_dir/ lands in the run dir


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={RUNS: volume})
def train_cot(model: str, run: str, data: str, args: list[str]):
    volume.reload()                           # see data generated after this container started
    if not pathlib.Path(data, "chain", "train_data.txt").exists():
        raise RuntimeError(f"no LIS data at {data}: run `modal run modal_app.py::cot_data` first")
    out = f"{RUNS}/cot/{run}"
    common = ["--file", data, "--folder", f"{REMOTE}/tasks/cot/LIS", "--model", model, *args]
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
    _run([*torchrun, f"{REMOTE}/tasks/cot/train.py", "--output_dir", out, "--write2file", *common], cwd=out)

    last = max(pathlib.Path(out).glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    _run([*torchrun, f"{REMOTE}/tasks/cot/test.py", "--model_path", str(last), *common], cwd=out, log_name="test.log")


# ----------------------------------------------------------------------------------- local entrypoints
@app.local_entrypoint()
def tests():
    run_tests.remote()


@app.local_entrypoint()
def cot_data(length: int = 50, train_size: float = 1e6, test_size: float = 1e5, number_range: int = 250):
    make_lis_data.remote(length, train_size, test_size, number_range)


@app.local_entrypoint()
def copy(models: str = "mamba", extra: str = "", pack: bool = False, tag: str = ""):
    """Copy task. One example per context (--no_pack) unless --pack: var2's state size depends on the
    position in the context, so a fair comparison runs every model unpacked (tasks/copy/README.md)."""
    names = _models(models)
    if pack and "mamba_var2" in names:
        raise SystemExit("--pack is not valid for mamba_var2")
    args = ([] if pack else ["--no_pack"]) + shlex.split(extra)
    stamp = _stamp()
    calls = [train_copy.spawn(m, f"{tag or 'copy'}_{m}_{stamp}", args) for m in names]
    for m, c in zip(names, calls):
        c.get()
        print(f"done: {m}  ->  volume mamba-runs: copy/{tag or 'copy'}_{m}_{stamp}")


@app.local_entrypoint()
def cot(models: str = "mamba", chain: bool = False, length: int = 50, number_range: int = 250,
        extra: str = "", tag: str = ""):
    """LIS with (--chain) or without CoT. Defaults are the command in tasks/cot/README.md."""
    names = _models(models)
    maxlen = 2 * length + 5 if chain else length + 4      # len 50: 105 CoT / 54 direct (upstream TRAIN.md)
    args = ["--maxlen", str(maxlen), "--maxdata", str(maxlen), "--vocab", str(number_range + 3),
            "--num_range", str(number_range), "--weight_decay", "0.01", "--learning_rate", "1e-4",
            "--batch_size", "16", "--epoch", "100", "--warmup", "5", "--dmodel", "256", "--num_layer", "6",
            *(["--chain"] if chain else []), *shlex.split(extra)]
    mode, stamp = "cot" if chain else "direct", _stamp()
    data = _lis_dir(length, number_range)
    runs = {m: f"{tag or 'lis'}{length}_{mode}_{m}_{stamp}" for m in names}
    calls = [train_cot.spawn(m, runs[m], data, args) for m in names]
    for m, c in zip(names, calls):
        c.get()
        print(f"done: {m}  ->  volume mamba-runs: cot/{runs[m]} (test accuracy in test.log)")
