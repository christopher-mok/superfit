"""Shared pair evaluation (D, IoU, barriers, descent, nearest transform, topology, classification),
Phase 0 sanity checks, and held-out validation of known pairs."""
from __future__ import annotations

import json
import math

import numpy as np
import torch as th

from . import geometry as G
from . import params as P
from .io import pair_id as make_pair_id

# parameters used for the mechanism-swap label (normalised by range width)
_MECH = [("size_x", 3), ("size_y", 4), ("size_z", 5), ("roundness", 6), ("dilate", 7), ("taper", 8), ("bulge", 9), ("onion", 10)]


def mechanism_swap(thA: th.Tensor, thB: th.Tensor, include_pose=True, k=3):
    w = (P.PMAX - P.PMIN).to(thA)
    out = []
    for a, b in zip(thA, thB):
        sc = {n: float((b[i] - a[i]).abs() / w[i]) for n, i in _MECH}
        signs = {n: ("↑" if float(b[i] - a[i]) > 0 else "↓") for n, i in _MECH}
        if include_pose:
            sc["rotation"] = float(P.rot_geodesic(a[None, P.ROT], b[None, P.ROT])[0] / math.pi)
            sc["translate"] = float((b[P.TRANS] - a[P.TRANS]).norm() / 2.0)
            signs["rotation"] = signs["translate"] = ""
        top = sorted(sc, key=sc.get, reverse=True)[:k]
        out.append([n + signs[n] for n in top if sc[n] > 1e-4])
    return out


def _pdist(x, y):
    n, r = G.combined_dist(P.param_to_var(x), P.param_to_var(y))
    return n + r


@th.no_grad()
def nearest_transform(reg, thA, thB, M, seed, chunk=64, restrict=None):
    """The registry transform that best *explains* the pair: g(θ_A) closest to θ_B in parameter space
    (raw + rotation geodesic). D(A,B) < ε already holds for a pair, so ranking by D would let any
    near-identity transform claim it. Accepted only if it removes >= 75% of the A–B distance.
    Returns (ids, residual D(g(θ_A), θ_B))."""
    ids = restrict or [t for t in reg.ids() if not reg[t].kind.startswith("learned")]
    N = thA.shape[0]
    d0 = _pdist(thA, thB)
    best = d0.clone()
    arg = [None] * N
    gbest = thA.clone()
    for tid in ids:
        gA = reg[tid].apply(thA)
        d = _pdist(gA, thB)
        better = d < best
        for i in th.nonzero(better).flatten().tolist():
            arg[i] = tid
        gbest = th.where(better[:, None], gA, gbest)
        best = th.where(better, d, best)
    ok = best < 0.25 * d0
    resid, _ = G.D_and_iou(gbest, thB, G.sample_points(gbest, M, seed), G.sample_points(thB, M, seed), chunk)
    out = [a if (a is not None and bool(o)) else "none" for a, o in zip(arg, ok)]
    resid = th.where(ok, resid, th.full_like(resid, float("nan")))
    return out, resid


def evaluate_pairs(thA, thB, metas, cfg, rd, reg, transform_classes=None, do_topology=True,
                   topo_cap=None, curve_cap=None, nearest=True, mech_pose=True, extra_curves=None,
                   descent_fix_mask=None):
    """Evaluate candidate pairs. metas: list of dicts with keys phase, driver, sample_id, source,
    prim_class, transform_id (or None), phi, features (dict, optional), delta_iou/delta_loss (optional).
    Returns list of row dicts (near-misses included; `qualifies` marks pairs)."""
    dev = thA.device
    N = thA.shape[0]
    if N == 0:
        return []
    eps, dh = cfg.eps_close, cfg.delta_hill
    D, iou = G.D_and_iou(thA, thB, G.sample_points(thA, cfg.M, cfg.seed), G.sample_points(thB, cfg.M, cfg.seed), cfg.chunk)
    near = D < cfg.near_miss_factor * eps
    # a pair needs two genuinely different parameter vectors (else it is the same point)
    near &= G.separation(P.param_to_var(thA), P.param_to_var(thB)) > cfg.min_sep
    idx = th.nonzero(near).flatten()
    if len(idx) == 0:
        return []
    a, b = thA[idx], thB[idx]
    ra, rb = P.param_to_var(a), P.param_to_var(b)
    bar = G.straight_barrier(ra, rb, cfg.path_T, cfg.M_path, cfg.seed, cfg.chunk)
    Dn = D[idx]
    close = Dn < eps
    hill = close & (bar["barrier"] > dh)
    # descent check on hills only (valleys are flat by construction)
    gd_cross = th.zeros(len(idx), dtype=th.bool, device=dev)
    gd_final = th.full((len(idx),), float("nan"), device=dev)
    hi = th.nonzero(hill).flatten()
    if len(hi):
        dc = G.descent_check(ra[hi], b[hi], cfg.M_opt, cfg.seed, cfg.gd_steps, cfg.gd_lr, eps, fix_mask=descent_fix_mask)
        gd_cross[hi], gd_final[hi] = dc["gd_crosses"], dc["gd_final_d"]
    neb_bar = th.full((len(idx),), float("nan"), device=dev)
    neb_lbar = th.full((len(idx),), float("nan"), device=dev)
    neb_curves = {}
    if cfg.run_neb and len(hi):
        pth = th.cat([G.string_relax(ra[hi][sl], rb[hi][sl], cfg.neb_images, cfg.neb_iters, M=cfg.M_opt, seed=cfg.seed)
                      for sl in G._chunks(len(hi), 32)])
        nb = G.barrier_along(G.densify(pth, 4), cfg.M_path, cfg.seed, cfg.chunk)
        neb_bar[hi], neb_lbar[hi] = nb["barrier"], nb["loss_barrier"]
        for j, h in enumerate(hi.tolist()):
            neb_curves[h] = (nb["t"].cpu().numpy(), nb["dA"][j].cpu().numpy(), nb["dB"][j].cpu().numpy())
    # nearest transform
    tids = [metas[i].get("transform_id") for i in idx.tolist()]
    resid = th.full((len(idx),), float("nan"), device=dev)
    if nearest:
        need = [j for j, t in enumerate(tids) if not t]
        if need:
            nid, nres = nearest_transform(reg, a[need], b[need], cfg.M_opt, cfg.seed, cfg.chunk)
            for k, j in enumerate(need):
                tids[j] = nid[k]
                resid[j] = nres[k]
    known = [j for j, t in enumerate(tids) if t and t != "none" and bool(th.isnan(resid[j])) and metas[int(idx[j])].get("transform_id")]
    if known:
        ga = th.stack([reg[tids[j]].apply(a[j:j + 1])[0] for j in known])
        d, _ = G.D_and_iou(ga, b[known], G.sample_points(ga, cfg.M_opt, cfg.seed), G.sample_points(b[known], cfg.M_opt, cfg.seed), cfg.chunk)
        resid[known] = d
    gdist, rgeo = G.combined_dist(ra, rb)
    mech = mechanism_swap(a, b, include_pose=mech_pose)
    feats = _features(a)
    # topology on qualifying rows (capped)
    topo = {}
    if do_topology:
        cand_topo = th.nonzero(close).flatten().tolist()
        if topo_cap is not None:
            cand_topo = cand_topo[:topo_cap]
        if cand_topo:
            ta = G.topology(a[cand_topo], cfg.topo_res, cfg.topo_eps)
            tb = G.topology(b[cand_topo], cfg.topo_res, cfg.topo_eps)
            for j, x, y in zip(cand_topo, ta, tb):
                topo[j] = (x, y, G.topology_class(x, y))
    rows = []
    ncurve = 0
    for j, i in enumerate(idx.tolist()):
        m = metas[i]
        tid = tids[j] or "none"
        is_close, is_hill = bool(close[j]), bool(hill[j])
        if not is_close:
            kind, q, reason = "approximate", False, "d_geo>=eps_close"
        elif not is_hill:
            kind, q, reason = "degenerate_valley", True, ""
        elif bool(gd_cross[j]):
            kind, q, reason = "approximate", False, "gd_crosses"
        else:
            q, reason = True, ""
            tcls = (transform_classes or {}).get(tid)
            kind = m.get("kind") or (tcls if tcls in ("exact", "conditional") else ("approximate" if tid == "none" else "conditional"))
        pid = make_pair_id(m["phase"], m["sample_id"], tid if m.get("transform_id") else m.get("driver", ""), float(m.get("phi", 0.0)))
        curve_file = None
        if q and (curve_cap is None or ncurve < curve_cap):
            ex = (extra_curves or {}).get(i, {})
            nc = neb_curves.get(j)
            curve_file = rd.save_curve(pid, t=bar["t"].cpu().numpy(), d_from_a=bar["dA"][j].cpu().numpy(),
                                       d_to_b=bar["dB"][j].cpu().numpy(),
                                       neb_t=None if nc is None else nc[0], neb_d_from_a=None if nc is None else nc[1],
                                       neb_d_to_b=None if nc is None else nc[2], **ex)
            ncurve += 1
        tp = topo.get(j)
        f = dict(feats[j])
        f.update(m.get("features", {}))
        spec = dict(m.get("spec", {}))
        if tid != "none" and tid in reg.t:
            t = reg[tid]
            spec.update(dict(axis=spec.get("axis"), angle=spec.get("angle"), size_perm=t.size_perm, bulge_flip=t.bulge_flip,
                             taper_flip=t.taper_flip is not None, axis_class=t.varaxis_alias, rotation_matrix=t.G))
        nb = sorted(set(P.near_bounds(a[j], cfg.near_bound_frac)) | set(P.near_bounds(b[j], cfg.near_bound_frac)))
        rows.append(dict(
            pair_id=pid, phase=m["phase"], kind=kind, driver=m.get("driver", "rotation"), mechanism_swap=mech[j],
            barrier_path="neb" if not math.isnan(float(neb_bar[j])) else "straight",
            topology_a=None if tp is None else tp[0], topology_b=None if tp is None else tp[1],
            topology_class=None if tp is None else tp[2], prim_class=m.get("prim_class", "SF"),
            source=m.get("source", "synthetic"), sample_id=m["sample_id"],
            theta_a_raw=ra[j].cpu(), theta_b_raw=rb[j].cpu(), theta_a=a[j].cpu(), theta_b=b[j].cpu(),
            transform_id=tid, transform_spec=spec, residual_after_transform=float(resid[j]),
            d_geo=float(Dn[j]), iou_ab=float(iou[i]), param_dist_raw=float(gdist[j]), rot_geodesic=float(rgeo[j]),
            barrier=float(bar["barrier"][j]), barrier_t=float(bar["barrier_t"][j]), loss_barrier=float(bar["loss_barrier"][j]),
            barrier_neb=float(neb_bar[j]), loss_barrier_neb=float(neb_lbar[j]),
            gd_crosses=bool(gd_cross[j]), gd_final_d=float(gd_final[j]), near_bounds=nb, features=f,
            delta_iou=m.get("delta_iou"), delta_loss=m.get("delta_loss"), curve_file=curve_file,
            qualifies=q, fail_reason=reason,
        ))
    return rows


def _features(theta):
    from .transforms import features
    f = features(theta)
    return [{k: float(v[i]) for k, v in f.items()} for i in range(theta.shape[0])]


# ----------------------------------------------------------------------------- Phase 0
def phase0(cfg, rd, log):
    """Sanity: layouts, ranges, cube rotation hump, metric stability, VarAxis semantics."""
    from .transforms import build_registry
    dev = cfg.device
    rep = {"checks": {}}
    ok = True
    # ranges
    rep["ranges"] = {P.NAMES[i]: [float(P.PMIN[i]), float(P.PMAX[i])] for i in range(11)}
    raw = th.randn(256, 14, device=dev) * 2
    rt = (P.param_to_var(P.var_to_param(raw)) - raw).abs()[:, :11].max()
    rep["checks"]["raw_roundtrip_max_abs"] = float(rt)
    # 1-3: cube hump
    cube = P.default_theta().to(dev)[None]
    phis = th.arange(0, 361, 5.0, device=dev)
    Rz = P.axis_rot(th.tensor([0.0, 0, 1], device=dev).expand(len(phis), 3), th.deg2rad(phis))
    rot = P.rotate_local(cube.expand(len(phis), -1).clone(), Rz)
    S, iou = G.D_and_iou(cube.expand(len(phis), -1), rot, G.sample_points(cube.expand(len(phis), -1), cfg.M, cfg.seed), G.sample_points(rot, cfg.M, cfg.seed), cfg.chunk)
    d90, d45 = float(S[18]), float(S[9])
    rep["checks"]["cube_D_90"] = d90
    rep["checks"]["cube_D_45"] = d45
    c1, c2 = d90 < cfg.eps_close, d45 > cfg.delta_hill
    rep["checks"]["cube_90_identity_ok"] = c1
    rep["checks"]["cube_45_hump_ok"] = c2
    ok &= c1 and c2
    rd.save_curve("phase0-cube-z", phi_deg=phis.cpu().numpy(), S=S.cpu().numpy(), iou=iou.cpu().numpy())
    # metric stability M vs M_check
    reg = build_registry()
    g = th.Generator().manual_seed(cfg.seed)
    tt = th.stack([P.default_theta(sx=float(0.2 + 0.6 * th.rand(1, generator=g)), sy=float(0.2 + 0.6 * th.rand(1, generator=g)),
                                   sz=0.5, roundness=float(th.rand(1, generator=g)) * 0.8, bulge=float(th.rand(1, generator=g)) * 0.8 + 0.01,
                                   taper=float(0.5 + th.rand(1, generator=g)))
                   for _ in range(8)]).to(dev)
    tb = reg["r90[00+]"].apply(tt)
    d1, _ = G.D_pair(tt, tb, cfg.M, cfg.seed)
    d2, _ = G.D_pair(tt, tb, cfg.M_check, cfg.seed + 7)
    rel = ((d1 - d2).abs() / d2.clamp_min(1e-4)).max()
    rep["checks"]["metric_rel_diff_M_vs_Mcheck"] = float(rel)
    rep["checks"]["metric_stable"] = bool(rel < 0.1)
    # bulge = 0 singularity
    q = th.tensor([[[0.4, 0.0, 0.0]]], device=dev)
    f0 = float(G.local_sdf(q, P.default_theta(bulge=0.0).to(dev)[None]))
    f1 = float(G.local_sdf(q, P.default_theta(bulge=1e-3).to(dev)[None]))
    rep["checks"]["bulge_zero_singular"] = dict(sdf_at_x0p4_bulge0=f0, sdf_at_x0p4_bulge_1em3=f1,
                                                note="sign(0)=0 in map_arc_bulge collapses x; samplers keep |bulge|>=1e-3")
    # VarAxis semantics: class k of batched_varaxis_sf_packed_eval == SF with R'=P R, size'=size[perm]
    from superfit.torch_compute.batched_sf import batched_varaxis_sf_packed_eval
    p17 = th.zeros(4, 17, device=dev)
    p17[:, :11] = tt[:4, :11]
    p17[:, 14:17] = th.randn(4, 3, generator=g).to(dev)
    pts = th.rand(4, 2048, 3, generator=g).to(dev) * 1.2 - 0.6
    errs = {}
    for k, cls in enumerate(["Y", "Z", "X"]):
        lg = th.full((4, 3), -60.0, device=dev)
        lg[:, k] = 60.0
        p17[:, 11:14] = lg
        f_va = batched_varaxis_sf_packed_eval(pts, p17, 1e-3)
        f_sf = P.sdf(pts, P.varaxis_to_sf(p17, cls))
        errs[cls] = float((f_va - f_sf).abs().max())
    rep["checks"]["varaxis_fold_max_err"] = errs
    rep["checks"]["varaxis_fold_ok"] = all(v < 1e-4 for v in errs.values())
    ok &= rep["checks"]["varaxis_fold_ok"]
    rep["varaxis_layouts"] = ("batched_varaxis_sf_packed_eval: logits[11:14], rotation[14:17] (unpacked_params_varsf); "
                              "batched_varaxis_sf_packed_eval_st: rotation[11:14], logits[14:17] (unpacked_params_sf). "
                              "Weight order w0=identity('Y' in reinit), w1=perm[1,2,0]('Z'), w2=perm[2,0,1]('X').")
    rep["squash"] = "algebraic tanh: p=(1-1e-4)*v/sqrt(1+v^2)*scale+offset on params[0:11]; rotation/logits raw"
    rep["ok"] = bool(ok)
    rd.save_json("phase0.json", rep)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(phis.cpu(), S.cpu())
        ax.axhline(cfg.eps_close, ls=":", c="g")
        ax.axhline(cfg.delta_hill, ls=":", c="r")
        ax.set_xlabel("φ (deg) about local z")
        ax.set_ylabel("D(cube, rot(cube))")
        ax.set_title("Phase 0: cube hump")
        fig.tight_layout()
        fig.savefig(rd.path("figures", "phase0_cube_hump.png"), dpi=120)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        log(f"phase0 plot failed: {e}")
    log(f"phase0: D(90°)={d90:.2e} D(45°)={d45:.3f} metric_rel={float(rel):.3f} varaxis_err={errs} ok={ok}")
    return rep


# ----------------------------------------------------------------------------- held-out validation
def known_pairs(dev):
    """Known pairs (NOT used to design/tune the search; only checked after the run).
    Each: (name, query θ, predicate on a bank/partner θ that identifies the 'other mechanism')."""
    ann = P.default_theta(sx=0.6, sy=0.6, sz=0.08, roundness=0.9999, onion=0.75).to(dev)
    # taper flip partner: a frustum with taper > 1
    fr = P.default_theta(sx=0.5, sy=0.5, sz=0.5, roundness=0.6, taper=0.7).to(dev)
    sph = P.default_theta(sx=0.02, sy=0.02, sz=0.02, dilate=0.3).to(dev)
    return [
        ("onion_annulus__bulge_ring", ann, lambda t: (t[..., 9].abs() > 1.4) & (t[..., 10] < 0.3)),
        ("taper_flip", fr, lambda t: t[..., 8] > 1.15),
        ("dilated_core_sphere__rounded_body", sph, lambda t: (t[..., 7] < 0.1) & (t[..., 6] > 0.8)),
    ]
