"""Parameter layouts and raw<->param conversion for SuperFrustum (14) and VarAxisSF (17).

Verified against the checkout (Phase 0 re-checks numerically):
  * SF packed layout (`unpacked_params_sf`): translate[0:3] size[3:6] roundness[6] dilate[7]
    taper[8] bulge[9] onion[10] rotation(axis-angle)[11:14].
  * VarAxisSF (`unpacked_params_varsf`, used by `batched_varaxis_sf_packed_eval`): same first 11,
    logits[11:14], rotation[14:17]. NOTE `batched_varaxis_sf_packed_eval_st` uses the *SF* layout
    with logits at [14:17] (rotation at [11:14]) -- the two VarAxis evaluators disagree.
  * The squash is the *algebraic* tanh p = (1-1e-4) * v/sqrt(1+v^2) * scale + offset applied to
    the first 11 entries (`make_packed_var_to_param`), not torch.tanh. Rotation (and VarAxis
    logits) are passed through unbounded.
"""
from __future__ import annotations

import math

import torch as th

from . import _compat  # noqa: F401  (must precede superfit imports)
from superfit.optim.param_conversion.utils import RANGE_DICT, _EPS_OUT
from superfit.torch_compute.batched_sf import batched_sf_packed_eval, batched_sf_packed_eval_part_2
from superfit.torch_compute.compile_friendly import axis_angle_to_rotation_matrix

NAMES = ["tx", "ty", "tz", "sx", "sy", "sz", "roundness", "dilate", "taper", "bulge", "onion", "rx", "ry", "rz"]
IDX = {n: i for i, n in enumerate(NAMES)}
N_BOUNDED = 11
SHAPE_IDX = list(range(3, 11))  # size, roundness, dilate, taper, bulge, onion
ROT = slice(11, 14)
TRANS = slice(0, 3)

_KEYS = ["translate"] * 3 + ["sp_size"] * 3 + ["sp_roundness", "sp_dilate_3d", "sp_taper", "sp_bulge", "sp_onion_ratio"]


def _range(key):
    if key == "translate":
        import geolipi.symbolic as gls
        return RANGE_DICT[gls.Translate3D]
    return RANGE_DICT[key]


PMIN = th.tensor([_range(k)[0] for k in _KEYS], dtype=th.float32)
PMAX = th.tensor([_range(k)[1] for k in _KEYS], dtype=th.float32)
SCALE = 0.5 * (PMAX - PMIN)
OFFSET = 0.5 * (PMAX + PMIN)
# "Neutral" values where a mechanism switches off.
NEUTRAL = {"bulge": 0.0, "taper": 1.0, "onion": 0.0, "dilate": 0.0}
# bulge == 0 exactly is singular in map_arc_bulge (sign(0) = 0 collapses x); see Phase 0.
BULGE_TINY = 1e-3


def var_to_param(v: th.Tensor) -> th.Tensor:
    vb = v[..., :N_BOUNDED]
    sc, of = SCALE.to(v), OFFSET.to(v)
    p = (1 - _EPS_OUT) * vb / th.sqrt(1 + vb * vb) * sc + of
    p = th.minimum(th.maximum(p, PMIN.to(v)), PMAX.to(v))
    return th.cat([p, v[..., N_BOUNDED:]], -1)


def param_to_var(p: th.Tensor) -> th.Tensor:
    pb = th.minimum(th.maximum(p[..., :N_BOUNDED], PMIN.to(p)), PMAX.to(p))
    pu = (pb - OFFSET.to(p)) / SCALE.to(p) / (1 - _EPS_OUT)
    pu = pu.clamp(-1 + 1e-6, 1 - 1e-6)
    v = pu / th.sqrt((1 - pu * pu).clamp_min(1e-12))
    return th.cat([v, p[..., N_BOUNDED:]], -1)


def near_bounds(p: th.Tensor, frac: float = 0.05) -> list[str]:
    """Names of bounded params within `frac` of a range end (gradient of the squash vanishes there).
    Translation is excluded: its range is the scene box, not a shape bound."""
    out = []
    lo, hi = PMIN.to(p), PMAX.to(p)
    w = hi - lo
    for i in range(3, N_BOUNDED):
        if p[i] < lo[i] + frac * w[i] or p[i] > hi[i] - frac * w[i]:
            out.append(NAMES[i])
    return out


def sdf(coords: th.Tensor, params: th.Tensor) -> th.Tensor:
    """coords (B,M,3), params (B,14) -> (B,M). Thin wrapper over the repo evaluator."""
    return batched_sf_packed_eval(coords, params)


def default_theta(**kw) -> th.Tensor:
    base = dict(tx=0, ty=0, tz=0, sx=0.5, sy=0.5, sz=0.5, roundness=0.0, dilate=0.0, taper=1.0,
                bulge=BULGE_TINY, onion=0.0, rx=0, ry=0, rz=0)
    base.update(kw)
    return th.tensor([float(base[n]) for n in NAMES], dtype=th.float32)


# ----------------------------------------------------------------------------- rotations
def aa_to_R(aa: th.Tensor) -> th.Tensor:
    shp = aa.shape[:-1]
    return axis_angle_to_rotation_matrix(aa.reshape(-1, 3)).reshape(*shp, 3, 3)


def R_to_quat(R: th.Tensor) -> th.Tensor:
    """(...,3,3) -> (...,4) wxyz, robust (Shepperd)."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = th.stack([
        th.stack([1 + t, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], -1),
        th.stack([m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]], -1),
        th.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]], -1),
        th.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1),
    ], -2)  # (...,4,4)
    diag = th.stack([t, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    k = diag.argmax(-1)
    q = th.gather(cands, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True)
    return q * th.where(q[..., :1] < 0, -1.0, 1.0)


def quat_to_aa(q: th.Tensor) -> th.Tensor:
    q = q * th.where(q[..., :1] < 0, -1.0, 1.0)
    w = q[..., 0].clamp(-1, 1)
    v = q[..., 1:]
    s = v.norm(dim=-1)
    ang = 2 * th.atan2(s, w)
    scale = th.where(s > 1e-8, ang / s.clamp_min(1e-12), 2.0 + 0 * s)
    return v * scale[..., None]


def aa_to_quat(aa: th.Tensor) -> th.Tensor:
    ang = aa.norm(dim=-1, keepdim=True)
    axis = aa / ang.clamp_min(1e-12)
    return th.cat([th.cos(ang / 2), axis * th.sin(ang / 2)], -1)


def R_to_aa(R: th.Tensor) -> th.Tensor:
    return quat_to_aa(R_to_quat(R))


def slerp(q0: th.Tensor, q1: th.Tensor, t: th.Tensor) -> th.Tensor:
    """q0,q1 (...,4); t broadcastable (...,1). Shortest arc."""
    d = (q0 * q1).sum(-1, keepdim=True)
    q1 = th.where(d < 0, -q1, q1)
    d = d.abs().clamp(max=1.0)
    om = th.acos(d)
    so = th.sin(om)
    lin = so < 1e-6
    a = th.where(lin, 1 - t, th.sin((1 - t) * om) / so.clamp_min(1e-12))
    b = th.where(lin, t, th.sin(t * om) / so.clamp_min(1e-12))
    q = a * q0 + b * q1
    return q / q.norm(dim=-1, keepdim=True)


def rot_geodesic(aa0: th.Tensor, aa1: th.Tensor) -> th.Tensor:
    q0, q1 = aa_to_quat(aa0), aa_to_quat(aa1)
    d = (q0 * q1).sum(-1).abs().clamp(max=1.0)
    return 2 * th.acos(d)


def axis_rot(axis: th.Tensor, phi: th.Tensor) -> th.Tensor:
    """Rotation matrices for unit axis (3,) or (...,3) and angles (...)."""
    axis = axis / axis.norm(dim=-1, keepdim=True)
    aa = axis * phi[..., None]
    return aa_to_R(aa)


def rotate_local(theta: th.Tensor, Rg: th.Tensor) -> th.Tensor:
    """Apply a local-frame rotation: the evaluator computes q = R (p - t); new q' = Rg q, i.e. R' = Rg R."""
    R = aa_to_R(theta[..., ROT])
    out = theta.clone()
    out[..., ROT] = R_to_aa(Rg @ R)
    return out


# ----------------------------------------------------------------------------- VarAxis
# batched_varaxis_sf_packed_eval: weights w0 -> identity, w1 -> coords/size [1,2,0], w2 -> [2,0,1].
VARAXIS_PERMS = {"Y": [0, 1, 2], "Z": [1, 2, 0], "X": [2, 0, 1]}  # class names as in reinit_params_varaxis_sf


def perm_matrix(perm) -> th.Tensor:
    P = th.zeros(3, 3)
    for j, pj in enumerate(perm):
        P[j, pj] = 1.0  # (P q)_j = q[perm[j]]
    return P


def varaxis_to_sf(p17: th.Tensor, cls: str) -> th.Tensor:
    """Fold a hard VarAxis class into a plain SF: q' = P q, size' = size[perm] -> R' = P R."""
    perm = VARAXIS_PERMS[cls]
    P = perm_matrix(perm).to(p17)
    out = th.cat([p17[..., :11], p17[..., 14:17]], -1).clone()
    out[..., 3:6] = p17[..., 3:6][..., perm]
    R = aa_to_R(p17[..., 14:17])
    out[..., ROT] = R_to_aa(P @ R)
    return out


def bbox_halfextent(theta: th.Tensor) -> th.Tensor:
    """Conservative local-frame half extent (B,3). Accounts for the arc bend of map_arc_bulge:
    half-angle a = |bulge|*pi/2, arc radius R = (sz/2)/sin(a), centre c = (sz/2)/tan(a);
    the bent axis spans x in [c-R, 0] (mirrored by sign) and z up to R (a > pi/2) or sz/2."""
    size = theta[..., 3:6]
    taper = theta[..., 8:9]
    grow = th.maximum(taper, th.ones_like(taper))
    cross = 0.5 * size[..., :2] * grow
    hz = 0.5 * size[..., 2:3]
    a = (theta[..., 9:10].abs() * math.pi / 2).clamp_min(1e-5)
    R = (hz / th.sin(a).clamp_min(1e-3)).clamp(max=20.0)
    xext = (hz * th.tan(0.5 * a).clamp(max=1e3)).clamp(max=20.0)  # = R - c, cancellation-free
    zext = th.where(a > math.pi / 2, R, hz)
    pad = cross.amax(-1, keepdim=True)
    he = th.cat([xext + pad, cross[..., 1:2], zext + pad], -1)
    he = th.maximum(he, th.cat([cross, hz], -1))
    return he + theta[..., 7:8] + 0.02 * size.amax(-1, keepdim=True)
