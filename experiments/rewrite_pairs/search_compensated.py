"""Phase 3: compensated sweeps D*(v) = min_θ' D(θ, θ'|driver=v) for rotation and the other drivers,
plus a CMA-ES search over (axis, φ, Δθ) rewarding high barriers."""
from __future__ import annotations

import math

import numpy as np
import torch as th

from . import geometry as G
from . import params as P
from .search_group import synthetic_samples
from .search_sweep import LOCAL_AXES
from .verify import evaluate_pairs

DRIVERS = {  # index, grid range (param space)
    "bulge": (9, (-1.95, 1.95)),
    "taper": (8, (0.05, 1.95)),
    "onion": (10, (0.0, 0.95)),
    "dilate": (7, (0.0, 0.45)),
}
ROT_RESTARTS = ["sizeperm_yxz", "sizeperm_xzy", "sizeperm_zyx", "sizeperm_yzx", "sizeperm_zxy", "bulge_flip"]
DRIVER_RESTARTS = ["taper_flip_x", "taper_flip_y", "bulge_flip", "r180[00+]+bulge_flip", "r90[00+]", "r90[+00]", "r90[0+0]"]


def compensate(thA, thB0, free: th.Tensor, cfg, steps=None, lr=None, chunk=256):
    """Adam over raw(θ_B)[free] minimising D(θ_A, θ_B) on fixed points (samples of A ∪ B0).
    Returns (θ_B*, D*) with D* re-measured on fresh M_opt samples."""
    steps = steps or cfg.inner_steps
    lr = lr or cfg.inner_lr
    N = thA.shape[0]
    outs = []
    sA = G.sample_points(thA, cfg.M_opt, cfg.seed)
    sB = G.sample_points(thB0, cfg.M_opt, cfg.seed)
    fmask = free.to(thA.device).float()
    for sl in G._chunks(N, chunk):
        pts = th.cat([sA.pts[sl], sB.pts[sl]], 1)
        v0 = P.param_to_var(thB0[sl])
        dv = th.zeros_like(v0, requires_grad=True)
        opt = th.optim.Adam([dv], lr=lr)
        a = thA[sl]
        for _ in range(steps):
            opt.zero_grad()
            b = P.var_to_param(v0 + dv * fmask)
            G.D_on_points(a, b, pts).sum().backward()
            opt.step()
        outs.append(P.var_to_param(v0 + dv.detach() * fmask))
    thB = th.cat(outs, 0)
    D, _ = G.D_and_iou(thA, thB, G.sample_points(thA, cfg.M_opt, cfg.seed + 3), G.sample_points(thB, cfg.M_opt, cfg.seed + 3), cfg.chunk)
    return thB, D


def _runs(mask: np.ndarray, circular: bool):
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    if circular and len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(mask) - 1:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    return runs


def rotation_driver(th_, sid, axis_name, cfg, reg):
    dev = th_.device
    phis = th.deg2rad(th.arange(0, 360, cfg.comp_step_deg)).to(dev)
    F = len(phis)
    ax = th.tensor(LOCAL_AXES[axis_name], device=dev).expand(F, 3)
    base = th_.expand(F, 14).clone()
    plain = P.rotate_local(base, P.axis_rot(ax, phis))
    S, _ = G.D_and_iou(base, plain, G.sample_points(base, cfg.M_opt, cfg.seed), G.sample_points(plain, cfg.M_opt, cfg.seed), cfg.chunk)
    restarts = [th_] + [reg[t].apply(th_) for t in ROT_RESTARTS if t in reg.t]
    R = len(restarts)
    inits = []
    for r in restarts:
        x = plain.clone()
        x[:, :11] = r[0, :11]
        inits.append(x)
    init = th.cat(inits, 0)  # (R*F,14)
    free = th.ones(14, dtype=th.bool)
    free[11:14] = False
    thB, D = compensate(base.repeat(R, 1), init, free, cfg)
    D = D.view(R, F)
    best = D.argmin(0)
    Dstar = D.min(0).values
    thB = thB.view(R, F, 14)[best, th.arange(F)]
    return phis, S, Dstar, thB


def param_driver(th_, sid, dname, cfg, reg):
    dev = th_.device
    di, (lo, hi) = DRIVERS[dname]
    vals = th.linspace(lo, hi, cfg.driver_grid, device=dev)
    if dname == "bulge":
        vals = th.where(vals.abs() < P.BULGE_TINY, th.full_like(vals, P.BULGE_TINY), vals)
    F = len(vals)
    base = th_.expand(F, 14).clone()
    plain = base.clone()
    plain[:, di] = vals
    S, _ = G.D_and_iou(base, plain, G.sample_points(base, cfg.M_opt, cfg.seed), G.sample_points(plain, cfg.M_opt, cfg.seed), cfg.chunk)
    restarts = [th_] + [reg[t].apply(th_) for t in DRIVER_RESTARTS if t in reg.t]
    inits = []
    for r in restarts:
        x = r.expand(F, 14).clone()
        x[:, di] = vals
        inits.append(x)
    init = th.cat(inits, 0)
    R = len(restarts)
    free = th.ones(14, dtype=th.bool)
    free[di] = False
    thB, D = compensate(base.repeat(R, 1), init, free, cfg)
    D = D.view(R, F)
    best = D.argmin(0)
    Dstar = D.min(0).values
    thB = thB.view(R, F, 14)[best, th.arange(F)]
    return vals, S, Dstar, thB


def run(cfg, rd, reg, transform_classes, log):
    dev = cfg.device
    theta_all, ids, _ = synthetic_samples(cfg, max(cfg.n_group, cfg.n_comp))
    n = cfg.n_comp
    for k, s0 in enumerate(range(0, n, cfg.shard_size)):
        if rd.shard_done("compensated", k):
            continue
        A, B, metas, curves = [], [], [], {}
        for i in range(s0, min(n, s0 + cfg.shard_size)):
            th_ = theta_all[i:i + 1].to(dev)
            jobs = [("rotation", a) for a in cfg.comp_axes] + [(d, None) for d in DRIVERS]
            for driver, axis_name in jobs:
                if driver == "rotation":
                    grid, S, Ds, thB = rotation_driver(th_, ids[i], axis_name, cfg, reg)
                    far = (th.rad2deg(grid) > 10) & (th.rad2deg(grid) < 350)
                    circ = True
                else:
                    grid, S, Ds, thB = param_driver(th_, ids[i], driver, cfg, reg)
                    di, (lo, hi) = DRIVERS[driver]
                    far = (grid - th_[0, di]).abs() > 0.1 * (hi - lo)
                    circ = False
                mask = ((Ds < cfg.eps_close) & far).cpu().numpy()
                Sn, Dn = S.cpu().numpy(), Ds.cpu().numpy()
                for run_ in _runs(mask, circ):
                    j = run_[int(np.argmin(Dn[run_]))]
                    A.append(th_[0])
                    B.append(thB[j])
                    m = len(metas)
                    gv = float(grid[j])
                    metas.append(dict(phase="compensated", driver=driver, sample_id=ids[i], source="synthetic", prim_class="SF",
                                      transform_id=None, phi=gv + {"x": 1000.0, "y": 2000.0, "z": 3000.0}.get(axis_name, 0.0) + (100.0 * (1 + list(DRIVERS).index(driver)) if driver in DRIVERS else 0.0),
                                      spec=dict(axis=LOCAL_AXES.get(axis_name), angle=gv if driver == "rotation" else None,
                                                driver_value=gv if driver != "rotation" else None, axis_name=axis_name),
                                      features=dict(S_minus_Dstar=float(Sn[j] - Dn[j]), S=float(Sn[j]), Dstar=float(Dn[j]),
                                                    hill_by_curves=bool(Sn[j] - Dn[j] > cfg.delta_hill))))
                    curves[m] = dict(grid=grid.cpu().numpy(), S=Sn, Dstar=Dn)
        rows = evaluate_pairs(th.stack(A), th.stack(B), metas, cfg, rd, reg, transform_classes, extra_curves=curves) if metas else []
        rd.write_shard("compensated", k, rows)
        log(f"phase3 shard {k}: {len(metas)} compensated minima, {sum(r['qualifies'] for r in rows)} pairs")
    run_cmaes(cfg, rd, reg, transform_classes, log)


# ----------------------------------------------------------------------------- CMA-ES
class CMAES:
    """Minimal (μ/μ_w, λ)-CMA-ES (Hansen 2016 defaults)."""

    def __init__(self, x0, sigma, pop, seed):
        self.n = n = len(x0)
        self.m = np.array(x0, float)
        self.sigma = sigma
        self.lam = pop
        self.mu = pop // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.rng = np.random.default_rng(seed)
        self.g = 0

    def ask(self):
        evals, B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(evals, 1e-20))
        self.B = B
        self.z = self.rng.standard_normal((self.lam, self.n))
        self.y = self.z * self.D @ self.B.T
        return self.m + self.sigma * self.y

    def tell(self, f):
        order = np.argsort(f)
        y = self.y[order[: self.mu]]
        z = self.z[order[: self.mu]]
        yw = self.w @ y
        self.m = self.m + self.sigma * yw
        zw = self.w @ z
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ zw)
        self.g += 1
        hs = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < (1.4 + 2 / (self.n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * (y.T * self.w) @ y)
        self.sigma *= math.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-4, 3.0))


def _cma_decode(x, rawA):
    x = th.as_tensor(x, dtype=th.float32, device=rawA.device)
    axis = x[:, 0:3] / x[:, 0:3].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    phi = x[:, 3]
    raw = rawA.expand(x.shape[0], 14).clone()
    raw[:, 3:11] = raw[:, 3:11] + x[:, 4:12]
    thB = P.rotate_local(P.var_to_param(raw), P.axis_rot(axis, phi))
    return thB, axis, phi


def run_cmaes(cfg, rd, reg, transform_classes, log):
    if rd.shard_done("cmaes", 0) or cfg.cmaes_samples <= 0:
        return
    dev = cfg.device
    theta_all, ids, _ = synthetic_samples(cfg, max(cfg.n_group, cfg.cmaes_samples))
    A, B, metas = [], [], []
    for i in range(cfg.cmaes_samples):
        thA = theta_all[i:i + 1].to(dev)
        rawA = P.param_to_var(thA)
        rng = np.random.default_rng(cfg.seed + i)
        ax0 = rng.standard_normal(3)
        es = CMAES(np.concatenate([ax0 / np.linalg.norm(ax0), [math.pi / 2], np.zeros(8)]), 0.5, cfg.cmaes_pop, cfg.seed + i)
        best = (float("inf"), None)
        for _ in range(cfg.cmaes_iters):
            X = es.ask()
            thB, _, _ = _cma_decode(X, rawA)
            a = thA.expand(len(X), 14)
            D, _ = G.D_and_iou(a, thB, G.sample_points(a, cfg.M_opt, cfg.seed), G.sample_points(thB, cfg.M_opt, cfg.seed), cfg.chunk)
            bar = G.straight_barrier(P.param_to_var(a), P.param_to_var(thB), 9, cfg.M_opt // 2, cfg.seed, cfg.chunk)["barrier"]
            f = D + 10 * th.relu(D - cfg.eps_close) - cfg.cmaes_lambda * bar.clamp(max=2 * cfg.delta_hill)
            f = f.cpu().numpy()
            es.tell(f)
            j = int(np.argmin(f))
            if f[j] < best[0]:
                best = (float(f[j]), X[j])
        thB, axis, phi = _cma_decode(best[1][None], rawA)
        A.append(thA[0])
        B.append(thB[0])
        metas.append(dict(phase="cmaes", driver="rotation", sample_id=ids[i], source="synthetic", prim_class="SF",
                          transform_id=None, phi=float(phi[0]),
                          spec=dict(axis=axis[0].tolist(), angle=float(phi[0]), cma_objective=best[0])))
    rows = evaluate_pairs(th.stack(A), th.stack(B), metas, cfg, rd, reg, transform_classes) if A else []
    rd.write_shard("cmaes", 0, rows)
    log(f"phase3 cmaes: {len(A)} searches, {sum(r['qualifies'] for r in rows)} pairs")
