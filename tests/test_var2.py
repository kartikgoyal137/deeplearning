"""Correctness tests for var2 (models/mamba_var2.py). Run:  python tests/test_var2.py   (no training)

1. one-stage schedule == the standard SelectiveSSM (same weights)
2. staged scan == a full-N_max literal recurrence where B and C of not-yet-active dims are zero
   (switch points deliberately not multiples of the scan chunk, L not a multiple either)
3. causal: changing token t does not change outputs before t
4. every parameter gets a finite gradient
"""
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.mamba import MambaConfig, SelectiveSSM, selective_scan_seq
from models.mamba_var2 import GrowingMambaLM, GrowingSSM, parse_schedule

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ok(msg):
    print(f"  [ok] {msg}")


def cfg_for(n_max, n_layer=2):
    return MambaConfig(vocab_size=30, d_model=32, n_layer=n_layer, d_state=n_max)


# 1. one stage == standard
cfg = cfg_for(16)
base, grow = SelectiveSSM(cfg).to(DEVICE), GrowingSSM(cfg, [(0, 16)]).to(DEVICE)
grow.load_state_dict(base.state_dict())
x, z = torch.randn(2, 50, cfg.d_inner, device=DEVICE), torch.randn(2, 50, cfg.d_inner, device=DEVICE)
err = (base(x, z) - grow(x, z)).abs().max().item()
assert err < 1e-4, err
ok(f"one-stage schedule == standard SelectiveSSM (max err {err:.1e})")

# 2. staged == masked full-N_max reference
schedule = parse_schedule("0:4,10:8,25:16")
cfg = cfg_for(16)
ssm = GrowingSSM(cfg, schedule).to(DEVICE)
L = 61
x, z = torch.randn(2, L, cfg.d_inner, device=DEVICE), torch.randn(2, L, cfg.d_inner, device=DEVICE)
n_at = torch.tensor([max(n for s, n in schedule if s <= t) for t in range(L)], device=DEVICE)   # N(t)
active = (torch.arange(cfg.d_state, device=DEVICE)[None, :] < n_at[:, None]).float()               # (L, N_max)
dt, B_, C_ = torch.split(ssm.x_proj(x), [cfg.dt_rank, cfg.d_state, cfg.d_state], -1)
ref = selective_scan_seq(x, F.softplus(ssm.dt_proj(dt)), -torch.exp(ssm.A_log.float()),
                         B_ * active, C_ * active, ssm.D, z)
err = (ssm(x, z) - ref).abs().max().item()
assert err < 1e-3, err
ok(f"schedule {schedule}: staged scan == masked full-N reference (L={L}, max err {err:.1e})")

# 3. causality
model = GrowingMambaLM(cfg, schedule).to(DEVICE).eval()
idx = torch.randint(0, 30, (2, L), device=DEVICE)
idx2 = idx.clone(); idx2[:, 30] = (idx2[:, 30] + 1) % 30
with torch.no_grad():
    a, b = model(idx)[0], model(idx2)[0]
assert torch.equal(a[:, :30], b[:, :30]) and not torch.equal(a[:, 30:], b[:, 30:])
ok("causal: changing token 30 leaves positions < 30 unchanged (across a switch at 25)")

# 4. gradients
model.train()
model(idx, idx)[1].backward()
bad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
assert not bad, bad
late = model.layers[0].mixer.ssm.x_proj.weight.grad[cfg.dt_rank + 8:cfg.dt_rank + 16]   # B rows of dims 8..15
assert late.abs().sum() > 0
ok("all parameters have finite gradients; late dims (activated at token 25) get gradient")

print("\nall checks passed")
