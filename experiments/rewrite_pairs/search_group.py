"""Phase 1: exhaustive discrete group search over the transform registry."""
from __future__ import annotations

import math
import time

import numpy as np
import torch as th
from scipy.stats import qmc

from . import geometry as G
from . import params as P
from .verify import evaluate_pairs


def synthetic_samples(cfg, n=None):
    """Sobol over shape params; identity rotation & zero translation, plus a `rot_subset` fraction with
    random rotations. A 30% subset snaps mechanisms to neutral / sizes to equal so conditional
    identities (which live on measure-zero sets) are represented. Deterministic in cfg.seed."""
    n = n or cfg.n_group
    sob = qmc.Sobol(d=8, scramble=True, seed=cfg.seed)
    u = th.tensor(sob.random(n), dtype=th.float32)
    g = th.Generator().manual_seed(cfg.seed + 1)
    ratios = th.exp(u[:, 0:3] * math.log(25.0) - math.log(5.0))           # [0.2, 5]
    size = 0.8 * ratios / ratios.amax(1, keepdim=True)
    rnd = u[:, 3] * 0.999
    dil = (u[:, 4] ** 2) * 0.25
    tap = 0.05 + 1.9 * u[:, 5]
    blg = (u[:, 6] * 2 - 1) * 1.95
    blg = th.where(blg.abs() < P.BULGE_TINY, th.full_like(blg, P.BULGE_TINY), blg)
    oni = u[:, 7] * 0.95
    snap = th.rand(n, generator=g) < 0.3
    for col, neutral in ((tap, 1.0), (blg, P.BULGE_TINY), (oni, 0.0), (dil, 0.0)):
        m = snap & (th.rand(n, generator=g) < 0.5)
        col[m] = neutral
    eq = snap & (th.rand(n, generator=g) < 0.5)
    size[eq, 1] = size[eq, 0]
    eq3 = eq & (th.rand(n, generator=g) < 0.3)
    size[eq3, 2] = size[eq3, 0]
    theta = th.zeros(n, 14)
    theta[:, 3:6] = size
    theta[:, 6], theta[:, 7], theta[:, 8], theta[:, 9], theta[:, 10] = rnd, dil, tap, blg, oni
    rot = th.rand(n, generator=g) < cfg.rot_subset
    aa = th.randn(n, 3, generator=g)
    ang = th.rand(n, generator=g) * math.pi
    theta[rot, 11:14] = (aa / aa.norm(dim=-1, keepdim=True) * ang[:, None])[rot]
    theta = P.var_to_param(P.param_to_var(theta))  # snap onto the representable set
    ids = [f"s{i:06d}" for i in range(n)]
    return theta, ids, rot


def run(cfg, rd, reg, log):
    dev = cfg.device
    theta_all, ids, rotmask = synthetic_samples(cfg)
    tids = reg.ids()
    T = len(tids)
    n = len(ids)
    stats = {t: dict(n=0, close=0, close_rot=0, n_rot=0, hill=0) for t in tids}
    prev = rd.load_json("phase1_stats.json", {})
    for k, v in prev.get("partial", {}).items():
        if k in stats:
            stats[k] = v
    done_shards = set(prev.get("done", []))
    t0 = time.time()
    for k, s0 in enumerate(range(0, n, cfg.shard_size)):
        if rd.shard_done("group", k) and k in done_shards:
            continue
        sl = slice(s0, min(n, s0 + cfg.shard_size))
        th_ = theta_all[sl].to(dev)
        B = th_.shape[0]
        thA = th_.repeat_interleave(T, 0)
        thB = th.cat([reg[t].apply(th_) for t in tids], 0).view(T, B, 14).transpose(0, 1).reshape(B * T, 14)
        D, _ = G.D_and_iou(thA, thB, G.sample_points(thA, cfg.M_opt, cfg.seed), G.sample_points(thB, cfg.M_opt, cfg.seed), cfg.chunk)
        Dm = D.view(B, T)
        np.savez_compressed(rd.shard_path("group", "dmat", k).replace(".parquet", ".npz"), D=Dm.cpu().numpy(),
                            theta=th_.cpu().numpy(), tids=np.array(tids), ids=np.array(ids[sl]))
        rm = rotmask[sl].to(dev)
        # samples where g(θ) ≈ θ (e.g. a size swap of equal sizes, r -> -r at r = 0) are fixed points,
        # not evidence for an identity: exclude them from the class statistics
        moved = (G.separation(P.param_to_var(thA), P.param_to_var(thB)) > cfg.min_sep).view(B, T)
        for j, t in enumerate(tids):
            c = Dm[:, j] < cfg.eps_close
            mv = moved[:, j]
            stats[t]["n"] += int(mv.sum())
            stats[t]["close"] += int((c & mv).sum())
            stats[t]["n_rot"] += int((rm & mv).sum())
            stats[t]["close_rot"] += int((c & mv & rm).sum())
        near = th.nonzero(Dm < cfg.near_miss_factor * cfg.eps_close)
        metas = []
        for bi, ti in near.tolist():
            metas.append(dict(phase="group", driver="rotation" if reg[tids[ti]].G is not None else tids[ti].split("_")[0],
                              sample_id=ids[s0 + bi], source="synthetic", prim_class="SF", transform_id=tids[ti], phi=float(ti)))
        a = th_[near[:, 0]] if len(near) else th_[:0]
        b = thB.view(B, T, 14)[near[:, 0], near[:, 1]] if len(near) else th_[:0]
        rows = evaluate_pairs(a, b, metas, cfg, rd, reg, do_topology=True, topo_cap=20, curve_cap=3 * 20)
        for r in rows:
            if r["qualifies"] and r["kind"] != "degenerate_valley":
                stats[r["transform_id"]]["hill"] += 1
        samples = [dict(sample_id=ids[s0 + i], source="synthetic", theta_raw=P.param_to_var(th_[i]).cpu(), theta_param=th_[i].cpu())
                   for i in range(B)]
        rd.write_shard("group", k, rows, samples)
        done_shards.add(k)
        rd.save_json("phase1_stats.json", {"partial": stats, "done": sorted(done_shards)})
        el = time.time() - t0
        log(f"phase1 shard {k}: {B} samples x {T} transforms, {len(near)} near, {sum(r['qualifies'] for r in rows)} pairs, "
            f"{B * T / max(el, 1e-9):.0f} (θ,g)/s cumulative")
    classes = {}
    for t, s in stats.items():
        f = s["close"] / max(s["n"], 1)
        fr = s["close_rot"] / max(s["n_rot"], 1) if s["n_rot"] else None
        classes[t] = "exact" if f >= 0.99 else ("conditional" if s["close"] > 0 else "none")
        s["frac_close"], s["frac_close_rot"], s["class"] = f, fr, classes[t]
    rd.save_json("phase1_stats.json", {"partial": stats, "done": sorted(done_shards)})
    rd.save_json("transform_classes.json", classes)
    log(f"phase1: exact={sum(v == 'exact' for v in classes.values())} conditional={sum(v == 'conditional' for v in classes.values())} "
        f"none={sum(v == 'none' for v in classes.values())}")
    return classes
