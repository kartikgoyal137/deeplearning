"""
var2: Mamba whose state size N grows along the sequence (capped at N_max).

The standard Mamba (models/mamba.py) uses the same state size N at every token. Here N follows a
schedule, e.g. [(0, 8), (16, 16), (32, 32), (64, 64)] = "N = 8 from token 0, 16 from token 16,
32 from token 32, 64 from token 64 on". The last N is the cap, so the state memory is bounded.

How a switch works: the new state dimensions are appended and start at zero, the old dimensions
are kept exactly as they are (nothing is projected or lost). Parameters (A, and the B/C rows of
x_proj) exist for N_max dimensions; a segment of constant N only uses the first N of them.
Cost therefore follows the schedule: each segment's scan and x_proj only touch N dimensions.

Same schedule in every layer. Training/evaluation path only (`step`/`generate` are not implemented).
"""

import torch
import torch.nn.functional as F

from models.mamba import MambaConfig, MambaLM, SelectiveSSM


def parse_schedule(text: str):
    """"0:8,16:16,32:32" -> [(0, 8), (16, 16), (32, 32)]"""
    return [tuple(int(v) for v in part.split(":")) for part in text.split(",")]


def scan_with_state(u, delta, A, B_, C_, D_, z, h, chunk: int = 8):
    """`selective_scan_chunked` from models/mamba.py, copied, with one change: it starts from the
    state `h` (B, D, N) instead of zeros and also returns the final state."""
    b, l, d = u.shape
    u, delta, B_, C_, A, h = u.float(), delta.float(), B_.float(), C_.float(), A.float(), h.float()

    dA_log = delta.unsqueeze(-1) * A                                # (B,L,D,N), <= 0
    dBu = delta.unsqueeze(-1) * B_.unsqueeze(2) * u.unsqueeze(-1)    # (B,L,D,N)

    ys = []
    for s in range(0, l, chunk):
        e = min(s + chunk, l)
        a = torch.cumsum(dA_log[:, s:e], dim=1)
        acc = dBu[:, s:e]
        step = 1
        while step < e - s:
            a_sh = F.pad(a[:, :-step], (0, 0, 0, 0, step, 0))
            acc_sh = F.pad(acc[:, :-step], (0, 0, 0, 0, step, 0))
            acc = torch.exp(a - a_sh) * acc_sh + acc
            step *= 2
        hs = torch.exp(a) * h.unsqueeze(1) + acc
        ys.append(torch.einsum("bcdn,bcn->bcd", hs, C_[:, s:e]))
        h = hs[:, -1]
    y = torch.cat(ys, dim=1) + u * D_.float()
    return y * F.silu(z.float()), h


class GrowingSSM(SelectiveSSM):
    """SelectiveSSM whose N follows `schedule`. cfg.d_state must be N_max (= the largest N)."""

    def __init__(self, cfg: MambaConfig, schedule):
        super().__init__(cfg)
        assert schedule[0][0] == 0 and schedule[-1][1] == cfg.d_state
        assert all(a < b for (a, _), (b, _) in zip(schedule, schedule[1:])), "starts must increase"
        assert all(n1 <= n2 for (_, n1), (_, n2) in zip(schedule, schedule[1:])), "N must not shrink"
        self.schedule = schedule

    def forward(self, x, z):
        cfg, r, n_max = self.cfg, self.cfg.dt_rank, self.cfg.d_state
        A = -torch.exp(self.A_log.float())                               # (D, N_max)
        L = x.shape[1]
        h, ys = None, []
        ends = [start for start, _ in self.schedule[1:]] + [L]
        for (start, n), end in zip(self.schedule, ends):
            end = min(end, L)
            if start >= end:
                break
            # x_proj rows of this segment: dt, B[:n], C[:n]   (x_proj output is [dt | B (N_max) | C (N_max)])
            w = self.x_proj.weight
            w = torch.cat([w[:r], w[r:r + n], w[r + n_max:r + n_max + n]])
            dt, B_, C_ = torch.split(F.linear(x[:, start:end], w), [r, n, n], -1)
            dt = F.softplus(self.dt_proj(dt))
            # grow the state: keep the old dims, new dims start at zero
            h = x.new_zeros(x.shape[0], x.shape[2], n) if h is None else F.pad(h, (0, n - h.shape[-1]))
            y, h = scan_with_state(x[:, start:end], dt, A[:, :n], B_, C_, self.D, z[:, start:end], h, cfg.scan_chunk)
            ys.append(y)
        return torch.cat(ys, dim=1)

    def step(self, x_t, ssm_state):
        raise NotImplementedError("var2 has no recurrent step yet")


class GrowingMambaLM(MambaLM):
    """MambaLM with every layer's SSM replaced by a GrowingSSM. Build cfg with d_state = N_max."""

    def __init__(self, cfg: MambaConfig, schedule):
        super().__init__(cfg)
        for layer in self.layers:
            layer.mixer.ssm = GrowingSSM(cfg, schedule)
        self.schedule = schedule
