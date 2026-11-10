"""Phase 2: 1-D continuous rotation sweeps S(φ) = D(θ, rot(θ, axis, φ)) and their non-trivial minima."""
from __future__ import annotations

import math

import numpy as np
import torch as th
from scipy.signal import find_peaks

from . import geometry as G
from . import params as P
from .search_group import synthetic_samples
from .verify import evaluate_pairs

LOCAL_AXES = {"x": [1.0, 0, 0], "y": [0, 1.0, 0], "z": [0, 0, 1.0]}


def sample_axes(cfg, sample_idx):
    g = th.Generator().manual_seed(cfg.seed * 1000003 + sample_idx)
    ax = dict(LOCAL_AXES)
    for r in range(cfg.n_random_axes):
        v = th.randn(3, generator=g)
        ax[f"rand{r}"] = (v / v.norm()).tolist()
    return ax


def sweep_curves(theta, axes, phis_rad, cfg):
    """theta (B,14); axes (A,3) -> S (B,A,F) with M_path points."""
    B, A, F = theta.shape[0], axes.shape[0], phis_rad.shape[0]
    ax = axes[None, :, None].expand(B, A, F, 3).reshape(-1, 3)
    ph = phis_rad[None, None].expand(B, A, F).reshape(-1)
    base = theta[:, None, None].expand(B, A, F, 14).reshape(-1, 14).clone()
    rot = P.rotate_local(base, P.axis_rot(ax, ph))
    S, _ = G.D_and_iou(base, rot, G.sample_points(base, cfg.M_path, cfg.seed), G.sample_points(rot, cfg.M_path, cfg.seed), cfg.chunk)
    return S.view(B, A, F), rot.view(B, A, F, 14)


def find_minima(S: np.ndarray, step_deg: float, eps: float, dh: float):
    """Circular curve S over [0, 360). Returns list of (index, barrier_on_short_arc, kind_hint)."""
    F = len(S)
    ext = np.concatenate([S, S, S])
    pk, _ = find_peaks(-ext)
    pk = [p - F for p in pk if F <= p < 2 * F]
    excl = max(2, int(round(5.0 / step_deg)))
    out = []
    for p in pk:
        if p < excl or p > F - excl or S[p] >= eps:
            continue
        arc = S[: p + 1] if p <= F // 2 else np.concatenate([S[p:], S[:1]])
        out.append((p, float(arc.max())))
    return out


def run(cfg, rd, reg, transform_classes, log):
    dev = cfg.device
    theta_all, ids, _ = synthetic_samples(cfg, max(cfg.n_group, cfg.n_sweep))
    n = cfg.n_sweep
    step = cfg.sweep_step_deg
    phis = th.deg2rad(th.arange(0, 360, step)).to(dev)
    for k, s0 in enumerate(range(0, n, cfg.shard_size)):
        if rd.shard_done("sweep", k):
            continue
        rows_a, rows_b, metas, curves = [], [], [], {}
        for i in range(s0, min(n, s0 + cfg.shard_size)):
            axd = sample_axes(cfg, i)
            names = list(axd)
            axes = th.tensor([axd[a] for a in names], device=dev)
            th_ = theta_all[i:i + 1].to(dev)
            S, rot = sweep_curves(th_, axes, phis, cfg)
            Sn = S[0].cpu().numpy()
            for ai, an in enumerate(names):
                mins = find_minima(Sn[ai], step, cfg.eps_close, cfg.delta_hill)
                picked = [m for m in mins if m[1] > cfg.delta_hill]
                if Sn[ai].max() < cfg.eps_close:          # dead rotation: whole curve flat
                    picked = [(int(round(90 / step)), float(Sn[ai].max()))]
                elif not picked and mins:                 # flat-arc minimum -> valley candidate
                    picked = [min(mins, key=lambda m: Sn[ai][m[0]])]
                for p, barr in picked:
                    j = len(metas)
                    rows_a.append(th_[0])
                    rows_b.append(rot[0, ai, p])
                    phi = float(phis[p])
                    metas.append(dict(phase="sweep", driver="rotation", sample_id=ids[i], source="synthetic", prim_class="SF",
                                      transform_id=None, phi=phi + 10 * ai,
                                      spec=dict(axis=axd[an], angle=phi, axis_name=an,
                                                note="rotation about local z == bend direction" if an == "z" else ""),
                                      features=dict(sweep_arc_barrier=barr)))
                    curves[j] = dict(phi=phis.cpu().numpy(), S=Sn[ai])
        if metas:
            rows = evaluate_pairs(th.stack(rows_a), th.stack(rows_b), metas, cfg, rd, reg, transform_classes,
                                  do_topology=True, topo_cap=20, extra_curves=curves)
        else:
            rows = []
        rd.write_shard("sweep", k, rows)
        log(f"phase2 shard {k}: {len(metas)} minima evaluated, {sum(r['qualifies'] for r in rows)} pairs")
