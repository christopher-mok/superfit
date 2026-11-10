"""Sampling, geometric distance D, IoU, interpolation paths, barriers, descent, string relaxation, topology."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import math
from dataclasses import dataclass

import numpy as np
import torch as th

from . import params as P


# ----------------------------------------------------------------------------- utils
def _chunks(B, chunk):
    for i in range(0, B, chunk):
        yield slice(i, min(B, i + chunk))


def theta_seed(theta: th.Tensor, base_seed: int) -> int:
    """Fixed seed per theta: hash of the rounded parameter vector + run seed."""
    key = np.round(theta.detach().cpu().double().numpy(), 5).tobytes() + str(base_seed).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & 0x7FFFFFFF


def local_sdf(q: th.Tensor, theta: th.Tensor) -> th.Tensor:
    """SDF in the primitive's own local frame (ignores theta's translate/rotation)."""
    size, rnd, dil, tap, blg, oni = theta[:, 3:6], theta[:, 6:7], theta[:, 7:8], theta[:, 8:9], theta[:, 9:10], theta[:, 10:11]
    return P.batched_sf_packed_eval_part_2(q, size, rnd, dil, tap, blg, oni)


def to_world(q: th.Tensor, theta: th.Tensor) -> th.Tensor:
    # evaluator: q = R (p - t)  =>  p = R^T q + t ; with row vectors p = q R + t
    R = P.aa_to_R(theta[:, P.ROT])
    return q @ R + theta[:, None, P.TRANS]


def scale_of(theta: th.Tensor) -> th.Tensor:
    return theta[..., 3:6].amax(-1)


def _probe_box(theta, q_lo, q_hi, u):
    q = q_lo[:, None] + u * (q_hi - q_lo)[:, None]
    with th.no_grad():
        inside = local_sdf(q, theta) < 0
    big = 1e6
    lo = th.where(inside[..., None], q, th.full_like(q, big)).amin(1)
    hi = th.where(inside[..., None], q, th.full_like(q, -big)).amax(1)
    empty = ~inside.any(1)
    lo, hi = th.where(empty[:, None], q_lo, lo), th.where(empty[:, None], q_hi, hi)
    cell = (q_hi - q_lo) / max(u.shape[1] ** (1 / 3) - 1, 1)
    return th.maximum(lo - cell, q_lo), th.minimum(hi + cell, q_hi)


def local_bbox(theta: th.Tensor, probe_u: th.Tensor | None = None, pad: float = 0.25, n_probe: int = 4096):
    """Empirical local-frame bbox of the occupied region (two-stage probe: analytic bound, then the
    stage-1 box), padded by `pad` of its extent per side. probe_u: (B,n,3) uniforms in [0,1)."""
    B = theta.shape[0]
    he = P.bbox_halfextent(theta) * 1.3 + 1e-3
    if probe_u is None:
        probe_u = th.rand(B, n_probe, 3, generator=th.Generator().manual_seed(0)).to(theta)
    h = probe_u.shape[1] // 2
    lo, hi = _probe_box(theta, -he, he, probe_u[:, :h])
    # stage 2: Newton-project probes onto the surface; the extremes of converged points are the bbox
    q = project_to_surface(lo[:, None] + probe_u[:, h:] * (hi - lo)[:, None], theta, steps=6)
    with th.no_grad():
        ok = local_sdf(q, theta).abs() < 2e-3 * scale_of(theta)[:, None]
    big = 1e6
    lo2 = th.where(ok[..., None], q, th.full_like(q, big)).amin(1)
    hi2 = th.where(ok[..., None], q, th.full_like(q, -big)).amax(1)
    good = ok.sum(1) > 16
    lo = th.where(good[:, None], th.maximum(lo2, lo), lo)
    hi = th.where(good[:, None], th.minimum(hi2, hi), hi)
    ext = (hi - lo).clamp_min(1e-3)
    return lo - pad * ext, hi + pad * ext


@dataclass
class Samples:
    pts: th.Tensor        # (B,M,3) world coords; first n_surf near-surface, rest uniform
    n_surf: int


N_PROBE = 4096


def _draws(theta, M, base_seed):
    n_s = M // 2
    n_u = M - n_s
    PU, UU, CU, JN = [], [], [], []
    for b in range(theta.shape[0]):
        g = th.Generator().manual_seed(theta_seed(theta[b], base_seed))
        PU.append(th.rand(N_PROBE, 3, generator=g))
        UU.append(th.rand(n_u, 3, generator=g))
        CU.append(th.rand(2 * n_s, 3, generator=g))
        JN.append(th.randn(n_s, 3, generator=g))
    dev = theta.device
    return [th.stack(x).to(dev) for x in (PU, UU, CU, JN)]


# Sampling is a pure function of (θ, M, seed), and the same θ is sampled again and again (θ_A against
# every transform / path point / restart). Both caches return exactly what a fresh call would.
SAMPLE_CACHE_MAX_BYTES = 2 << 30
BBOX_CACHE_MAX = 500_000
_SAMPLE_CACHE: "OrderedDict[tuple, th.Tensor]" = OrderedDict()
_BBOX_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_cache_bytes = [0]
CACHE_STATS = dict(requests=0, computed=0, bbox_computed=0)


def clear_caches():
    _SAMPLE_CACHE.clear()
    _BBOX_CACHE.clear()
    _cache_bytes[0] = 0


def _bbox_cached(tb, PU, bkeys):
    """Padded local bbox per theta; the probe draws do not depend on M, so one θ sampled at several M
    shares its bbox."""
    need = [j for j, k in enumerate(bkeys) if k not in _BBOX_CACHE]
    if need:
        lo, hi = local_bbox(tb[need], PU[need])
        CACHE_STATS["bbox_computed"] += len(need)
        for n_, j in enumerate(need):
            _BBOX_CACHE[bkeys[j]] = (lo[n_], hi[n_])
    got = [_BBOX_CACHE[k] for k in bkeys]
    while len(_BBOX_CACHE) > BBOX_CACHE_MAX:
        _BBOX_CACHE.popitem(last=False)
    return th.stack([g[0] for g in got]), th.stack([g[1] for g in got])


def _sample_uncached(theta, M, base_seed, bkeys, chunk):
    B = theta.shape[0]
    n_s = M // 2
    out = th.empty(B, M, 3, device=theta.device)
    for sl in _chunks(B, chunk):
        tb = theta[sl]
        PU, UU, CU, JN = _draws(tb, M, base_seed)
        lo, hi = _bbox_cached(tb, PU, bkeys[sl])
        ext = (hi - lo)[:, None]
        uni = lo[:, None] + UU * ext
        cand = project_to_surface(lo[:, None] + CU * ext, tb)
        with th.no_grad():
            f = local_sdf(cand, tb).abs()
        idx = f.topk(n_s, dim=1, largest=False).indices
        surf = th.gather(cand, 1, idx[..., None].expand(-1, -1, 3))
        surf = surf + JN * 0.02 * scale_of(tb)[:, None, None]
        out[sl] = to_world(th.cat([surf, uni], 1), tb)
    return out


def sample_points(theta: th.Tensor, M: int, base_seed: int, chunk: int = 64) -> Samples:
    """50% near-surface (Newton-projected candidates, best |f|, + N(0, 0.02 s) jitter), 50% uniform in
    the 25%-padded bbox. Random draws come from a generator seeded by each theta's values.
    Results are memoised per exact (θ, M, seed); repeats inside one call are computed once."""
    theta = theta.detach()
    B = theta.shape[0]
    dev = str(theta.device)
    arr = np.ascontiguousarray(theta.cpu().numpy())
    raw = [arr[b].tobytes() for b in range(B)]
    keys = [(r, M, base_seed, dev) for r in raw]
    CACHE_STATS["requests"] += B
    local, miss = {}, []
    for b, k in enumerate(keys):
        if k in local:
            continue
        hit = _SAMPLE_CACHE.get(k)
        if hit is not None:
            _SAMPLE_CACHE.move_to_end(k)
            local[k] = hit
        else:
            local[k] = None
            miss.append(b)
    if miss:
        pts = _sample_uncached(theta[miss], M, base_seed, [(raw[b], base_seed, dev) for b in miss], chunk)
        CACHE_STATS["computed"] += len(miss)
        for n_, b in enumerate(miss):
            local[keys[b]] = pts[n_]
            _SAMPLE_CACHE[keys[b]] = pts[n_].clone()
            _cache_bytes[0] += M * 12
        while _cache_bytes[0] > SAMPLE_CACHE_MAX_BYTES and _SAMPLE_CACHE:
            _, old = _SAMPLE_CACHE.popitem(last=False)
            _cache_bytes[0] -= old.numel() * 4
    return Samples(th.stack([local[k] for k in keys]), M // 2)


def project_to_surface(q: th.Tensor, theta: th.Tensor, steps: int = 4) -> th.Tensor:
    q = q.detach()
    theta = theta.detach()
    for _ in range(steps):
        with th.enable_grad():
            q.requires_grad_(True)
            f = local_sdf(q, theta)
            (g,) = th.autograd.grad(f.sum(), q)
        with th.no_grad():
            gn = (g * g).sum(-1, keepdim=True).clamp_min(1e-8)
            q = (q - f[..., None] * g / gn).detach()
    return q


# ----------------------------------------------------------------------------- distances


def D_on_points(thA: th.Tensor, thB: th.Tensor, pts: th.Tensor, s: th.Tensor | None = None) -> th.Tensor:
    """Differentiable mean |f_A - f_B| / s on given points. (B,) """
    fA = P.sdf(pts, thA)
    fB = P.sdf(pts, thB)
    if s is None:
        s = th.maximum(scale_of(thA), scale_of(thB)).detach()
    return (fA - fB).abs().mean(-1) / s


@th.no_grad()
def D_and_iou(thA, thB, sA: Samples, sB: Samples, chunk: int = 64):
    """D over the union of both point sets; 1-IoU on occupancy over the union of uniform points."""
    B = thA.shape[0]
    Ds, ious = th.empty(B, device=thA.device), th.empty(B, device=thA.device)
    for sl in _chunks(B, chunk):
        pts = th.cat([sA.pts[sl], sB.pts[sl]], 1)
        fA, fB = P.sdf(pts, thA[sl]), P.sdf(pts, thB[sl])
        s = th.maximum(scale_of(thA[sl]), scale_of(thB[sl]))
        Ds[sl] = (fA - fB).abs().mean(-1) / s
        nA = sA.pts.shape[1]
        um = th.zeros(pts.shape[1], dtype=th.bool, device=pts.device)
        um[sA.n_surf:nA] = True
        um[nA + sB.n_surf:] = True
        oA, oB = (fA < 0)[:, um], (fB < 0)[:, um]
        inter = (oA & oB).sum(-1).float()
        union = (oA | oB).sum(-1).float().clamp_min(1)
        ious[sl] = inter / union
    return Ds, ious


def D_pair(thA, thB, M, seed, chunk=64):
    sA = sample_points(thA, M, seed)
    sB = sample_points(thB, M, seed)
    return D_and_iou(thA, thB, sA, sB, chunk)


# ----------------------------------------------------------------------------- paths
def path_params(rawA: th.Tensor, rawB: th.Tensor, T: int) -> th.Tensor:
    """SLERP for rotation (shortest arc), linear in raw space for everything else. -> (B,T,14) params."""
    t = th.linspace(0, 1, T, device=rawA.device)[None, :, None]
    raw = (1 - t) * rawA[:, None] + t * rawB[:, None]
    q0 = P.aa_to_quat(rawA[:, P.ROT])[:, None]
    q1 = P.aa_to_quat(rawB[:, P.ROT])[:, None]
    q = P.slerp(q0.expand(-1, T, -1), q1.expand(-1, T, -1), t.expand(rawA.shape[0], T, 1))
    raw = raw.clone()
    raw[..., P.ROT] = P.quat_to_aa(q)
    return P.var_to_param(raw)


@th.no_grad()
def barrier_along(path: th.Tensor, M: int, seed: int, chunk: int = 64, pair_chunk: int = 16):
    """path (B,T,14) params; endpoints path[:,0]=A, path[:,-1]=B.
    Returns dict with curves dA(t)=D(A,θ(t)), dB(t)=D(θ(t),B), barrier, barrier_t, loss_barrier.
    Processed `pair_chunk` pairs at a time to bound memory (B*T*M points otherwise)."""
    B, T, _ = path.shape
    dA = th.empty(B, T, device=path.device)
    dB = th.empty(B, T, device=path.device)
    for sl in _chunks(B, pair_chunk):
        pc = path[sl]
        n = pc.shape[0]
        thA, thB = pc[:, 0], pc[:, -1]
        sA, sB = sample_points(thA, M, seed), sample_points(thB, M, seed)
        flat = pc.reshape(n * T, -1)
        sT = sample_points(flat, M, seed)
        rep = lambda s: Samples(s.pts.repeat_interleave(T, 0), s.n_surf)
        a_, _ = D_and_iou(thA.repeat_interleave(T, 0), flat, rep(sA), sT, chunk)
        b_, _ = D_and_iou(flat, thB.repeat_interleave(T, 0), sT, rep(sB), chunk)
        dA[sl], dB[sl] = a_.view(n, T), b_.view(n, T)
    tg = th.linspace(0, 1, T, device=path.device)
    bar, bi = dA.max(1)
    lbar, _ = dB.max(1)
    return dict(t=tg, dA=dA, dB=dB, barrier=bar, barrier_t=tg[bi], loss_barrier=lbar)


def straight_barrier(rawA, rawB, T, M, seed, chunk=64):
    return barrier_along(path_params(rawA, rawB, T), M, seed, chunk)


def separation(rawX, rawY):
    """Raw-space separation incl. the raw axis-angle difference (what the optimiser has to walk);
    unlike combined_dist it sees representation changes such as the axis-angle wrap."""
    return (rawX - rawY).norm(dim=-1)


def combined_dist(rawX, rawY):
    """Raw-space distance on everything but rotation, plus rotation geodesic (rad)."""
    nr = [i for i in range(14) if i not in (11, 12, 13)]
    return (rawX[..., nr] - rawY[..., nr]).norm(dim=-1), P.rot_geodesic(rawX[..., P.ROT], rawY[..., P.ROT])


def descent_check(rawA, thB, M, seed, steps=500, lr=0.02, eps_close=0.01, chunk=128, fix_mask=None):
    """Adam on D(·, θ_B) from θ_A in raw space on fixed points (A ∪ B samples).
    gd_crosses := the final iterate is in B's basin, i.e. its (raw + rotation) distance to B is
    < 0.5 of the A–B distance (D(A,B) is already < ε_close for a qualifying pair, so 'reaching
    D < ε_close' alone is satisfied trivially at the start; what matters is whether descent can
    move across the hill)."""
    B = rawA.shape[0]
    thA = P.var_to_param(rawA)
    sA, sB = sample_points(thA, M, seed), sample_points(thB, M, seed)
    finals = []
    for sl in _chunks(B, chunk):
        pts = th.cat([sA.pts[sl], sB.pts[sl]], 1)
        s = th.maximum(scale_of(thA[sl]), scale_of(thB[sl])).detach()
        v = rawA[sl].clone().requires_grad_(True)
        opt = th.optim.Adam([v], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = D_on_points(P.var_to_param(v), thB[sl], pts, s).sum()
            loss.backward()
            if fix_mask is not None:
                v.grad[:, fix_mask] = 0
            opt.step()
        finals.append(v.detach())
    vf = th.cat(finals, 0)
    with th.no_grad():
        dfin, _ = D_and_iou(P.var_to_param(vf), thB, sample_points(P.var_to_param(vf), M, seed + 1), sample_points(thB, M, seed + 1), chunk)
    rawB = P.param_to_var(thB)
    nAB, rAB = combined_dist(rawA, rawB)
    nFB, rFB = combined_dist(vf, rawB)
    crosses = (nFB + rFB) < 0.5 * (nAB + rAB)
    return dict(final_raw=vf, gd_final_d=dfin, gd_crosses=crosses)


def string_relax(rawA, rawB, K=9, iters=60, lr=0.05, M=2048, seed=0, chunk=64):
    """Simplified string method on E(θ) = D(θ, θ_B) between fixed endpoints in raw space
    (rotation treated as its raw axis-angle vector). Returns relaxed path (B,K,14) params."""
    path_raw = th.empty(rawA.shape[0], K, 14, device=rawA.device)
    p0 = path_params(rawA, rawB, K)
    path_raw[:] = P.param_to_var(p0)
    path_raw[:, 0], path_raw[:, -1] = rawA, rawB
    thB = P.var_to_param(rawB)
    thA = P.var_to_param(rawA)
    sA, sB = sample_points(thA, M, seed), sample_points(thB, M, seed)
    pts_all = th.cat([sA.pts, sB.pts], 1)
    s = th.maximum(scale_of(thA), scale_of(thB)).detach()
    Bn = rawA.shape[0]
    for _ in range(iters):
        inner = path_raw[:, 1:-1].clone().requires_grad_(True)
        th_in = P.var_to_param(inner.reshape(-1, 14))
        E = D_on_points(th_in, thB.repeat_interleave(K - 2, 0), pts_all.repeat_interleave(K - 2, 0), s.repeat_interleave(K - 2, 0))
        (g,) = th.autograd.grad(E.sum(), inner)
        with th.no_grad():
            tan = path_raw[:, 2:] - path_raw[:, :-2]
            tan = tan / tan.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            g_perp = g - (g * tan).sum(-1, keepdim=True) * tan
            gn = g_perp.norm(dim=-1, keepdim=True)
            path_raw[:, 1:-1] = inner - lr * g_perp / gn.clamp_min(1.0)  # clipped step
            # reparametrize to equal arc length
            seg = (path_raw[:, 1:] - path_raw[:, :-1]).norm(dim=-1)
            cum = th.cat([th.zeros(Bn, 1, device=seg.device), seg.cumsum(1)], 1)
            tgt = th.linspace(0, 1, K, device=seg.device)[None] * cum[:, -1:]
            idx = th.searchsorted(cum.contiguous(), tgt.contiguous()).clamp(1, K - 1)  # (B,K), batched
            c0, c1 = th.gather(cum, 1, idx - 1), th.gather(cum, 1, idx)
            w = ((tgt - c0) / (c1 - c0).clamp_min(1e-9))[..., None]
            ex = idx[..., None].expand(-1, -1, 14)
            new = (1 - w) * th.gather(path_raw, 1, ex - 1) + w * th.gather(path_raw, 1, ex)
            new[:, 0], new[:, -1] = rawA, rawB
            path_raw = new
    return P.var_to_param(path_raw)


def densify(path: th.Tensor, per_seg: int = 4) -> th.Tensor:
    B, K, _ = path.shape
    raw = P.param_to_var(path)
    t = th.linspace(0, 1, per_seg + 1, device=path.device)[:-1]
    segs = [(1 - tt) * raw[:, :-1] + tt * raw[:, 1:] for tt in t]
    out = th.stack(segs, 2).reshape(B, (K - 1) * per_seg, 14)
    out = th.cat([out, raw[:, -1:]], 1)
    return P.var_to_param(out)


# ----------------------------------------------------------------------------- topology
def topology(theta: th.Tensor, res_list=(48, 80), eps_frac=0.01):
    """For each theta: {res: {level: (components, euler_char, cavities, genus)}} in its local frame."""
    from scipy import ndimage
    from skimage.measure import euler_number
    out = []
    for b in range(theta.shape[0]):
        tb = theta[b:b + 1]
        lo, hi = local_bbox(tb)
        s = float(scale_of(tb))
        rec = {}
        for r in res_list:
            ax = [th.linspace(float(lo[0, i]), float(hi[0, i]), r, device=theta.device) for i in range(3)]
            g = th.stack(th.meshgrid(*ax, indexing="ij"), -1).reshape(1, -1, 3)
            with th.no_grad():
                f = local_sdf(g, tb).reshape(r, r, r).cpu().numpy()
            lv = {}
            for name, lvl in (("-eps", -eps_frac * s), ("0", 0.0), ("+eps", eps_frac * s)):
                occ = np.pad(f < lvl, 1)
                lab, ncomp = ndimage.label(occ, structure=np.ones((3, 3, 3)))
                chi = int(euler_number(occ, connectivity=3)) if ncomp else 0
                _, nbg = ndimage.label(~occ)  # 6-connected background (dual of 26-fg)
                cav = max(nbg - 1, 0)
                genus = ncomp + cav - chi
                lv[name] = dict(components=int(ncomp), euler_char=chi, cavities=int(cav), genus=int(genus))
            rec[str(r)] = lv
        out.append(rec)
    return out


def _tkey(d):
    return (d["components"], d["genus"], d["cavities"])


def topology_class(ta: dict, tb: dict) -> str:
    res = list(ta.keys())
    if all(_tkey(ta[r]["0"]) == _tkey(tb[r]["0"]) for r in res):
        return "same"
    lv = ["-eps", "0", "+eps"]
    for la in lv:
        for lb in lv:
            if all(_tkey(ta[r][la]) == _tkey(tb[r][lb]) for r in res):
                return "eps_equivalent"
    return "different"
