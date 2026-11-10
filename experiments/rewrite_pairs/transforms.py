"""Transform registry: deterministic, invertible θ -> θ' maps with string ids.

Conventions: the evaluator computes local coords q = R (p - t). A local-frame rotation G
(q' = G q) gives R' = G R, and for a box the matching size is size' = |G| size.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch as th

from . import params as P


@dataclass
class Transform:
    tid: str
    kind: str                         # group | perm | analytic | wrap | learned_affine | learned_mlp
    generators: list[str]
    description: str
    G: list | None = None             # 3x3 local rotation applied last
    size_perm: list | None = None
    bulge_flip: bool = False
    taper_flip: str | None = None     # None | "x" | "y"
    aa_op: str | None = None          # None | "wrap" | "negate"
    varaxis_alias: str | None = None  # VarAxis class change this transform reproduces
    learned: dict | None = None       # {"W": [[...]], "b": [...], "idx": [...]} (raw space)
    fn: Callable | None = field(default=None, repr=False)

    def apply(self, theta: th.Tensor) -> th.Tensor:
        return self.fn(theta)

    def spec(self) -> dict:
        d = dict(generators=self.generators, kind=self.kind, description=self.description,
                 rotation_matrix=self.G, size_perm=self.size_perm,
                 sign_flips=dict(bulge=self.bulge_flip), taper_flip=self.taper_flip,
                 aa_op=self.aa_op, varaxis_alias=self.varaxis_alias)
        if self.learned is not None:
            d["learned"] = self.learned
        return d


# ----------------------------------------------------------------------------- primitive ops
def cube_rotations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if round(np.linalg.det(M)) == 1:
                out.append(M)
    return out


def _rot_name(M: np.ndarray) -> str:
    if np.allclose(M, np.eye(3)):
        return "id"
    ang = math.degrees(math.acos(max(-1, min(1, (np.trace(M) - 1) / 2))))
    w, v = np.linalg.eig(M)
    ax = np.real(v[:, np.argmin(np.abs(w - 1))])
    ax = ax / np.abs(ax).max()
    if ax[np.argmax(np.abs(ax))] < 0:
        ax = -ax
    # sense of rotation about ax
    if ang not in (180.0,):
        Rp = _axis_angle_np(ax / np.linalg.norm(ax), math.radians(ang))
        if not np.allclose(Rp, M, atol=1e-6):
            ax = -ax
    lab = "".join({1: "+", -1: "-", 0: "0"}[int(round(c))] for c in ax)
    return f"r{int(round(ang))}[{lab}]"


def _axis_angle_np(a, ang):
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def op_group(theta, G: th.Tensor):
    out = P.rotate_local(theta, G.to(theta))
    out[..., 3:6] = (theta[..., 3:6][..., None, :] * G.abs().to(theta)).sum(-1)
    return out


def op_size_perm(theta, perm):
    out = theta.clone()
    out[..., 3:6] = theta[..., 3:6][..., perm]
    return out


def op_bulge_flip(theta):
    out = theta.clone()
    out[..., 9] = -theta[..., 9]
    return out


def op_taper_flip(theta, axis="x"):
    """Frustum (base a,b; corner r; top inset δ=(1-taper)·inner) == 180° x/y-rotated frustum with
    base (a-2δ, b-2δ), corner r-δ, outward top offset δ (taper' = 1/taper). Exact iff r >= δ."""
    out = theta.clone()
    a, b = theta[..., 3], theta[..., 4]
    inner = 0.5 * th.minimum(a, b)
    tap = theta[..., 8]
    delta = (1 - tap) * inner
    r = theta[..., 6] * inner
    inner2 = (inner - delta).clamp_min(1e-5)
    out[..., 3] = (a - 2 * delta).clamp_min(2e-5)
    out[..., 4] = (b - 2 * delta).clamp_min(2e-5)
    out[..., 6] = ((r - delta).clamp_min(0) / inner2).clamp(0, float(P.PMAX[6]))
    out[..., 8] = (1.0 / tap.clamp_min(1e-5)).clamp(float(P.PMIN[8]), float(P.PMAX[8]))
    G = th.diag(th.tensor([1.0, -1.0, -1.0])) if axis == "x" else th.diag(th.tensor([-1.0, 1.0, -1.0]))
    out = P.rotate_local(out, G.to(theta))
    if axis == "y":
        out[..., 9] = -out[..., 9]
    return out


def op_aa(theta, mode):
    out = theta.clone()
    r = theta[..., P.ROT]
    ang = r.norm(dim=-1, keepdim=True)
    axis = th.where(ang > 1e-8, r / ang.clamp_min(1e-12), th.tensor([0.0, 0.0, 1.0]).to(r))
    if mode == "wrap":
        out[..., P.ROT] = (ang - 2 * math.pi) * axis
    else:
        out[..., P.ROT] = -r
    return out


# ----------------------------------------------------------------------------- registry
class Registry:
    def __init__(self):
        self.t: dict[str, Transform] = {}

    def add(self, tr: Transform):
        self.t[tr.tid] = tr

    def __getitem__(self, k):
        return self.t[k]

    def __iter__(self):
        return iter(self.t.values())

    def __len__(self):
        return len(self.t)

    def ids(self):
        return list(self.t)

    def to_json(self) -> dict:
        return {k: v.spec() for k, v in self.t.items()}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    def load_learned(self, path):
        with open(path) as f:
            d = json.load(f)
        for tid, spec in d.items():
            if tid in self.t:
                continue
            if spec.get("kind", "").startswith("learned"):
                self.add(make_learned(tid, spec["learned"], spec.get("description", ""), spec.get("generators", [])))
            elif spec.get("kind") == "analytic_rotation":
                self.add(make_axis_rotation(tid, spec["learned"]["axis"], spec["learned"]["angle"], spec.get("description", "")))


def make_learned(tid, learned: dict, description="", generators=()):
    idx = learned["idx"]
    W = th.tensor(learned["W"], dtype=th.float32)
    bvec = th.tensor(learned["b"], dtype=th.float32)

    def fn(theta, W=W, bvec=bvec, idx=idx):
        raw = P.param_to_var(theta)
        out = raw.clone()
        out[..., idx] = raw[..., idx] @ W.to(raw).T + bvec.to(raw)
        return P.var_to_param(out)
    return Transform(tid, "learned_affine", list(generators), description, learned=learned, fn=fn)


def make_axis_rotation(tid, axis, angle, description=""):
    """Pure local rotation about `axis` by `angle` (rules distilled from sweep pairs with no group match)."""
    ax = th.tensor(axis, dtype=th.float32)

    def fn(theta, ax=ax, angle=angle):
        Rg = P.axis_rot(ax.to(theta), th.tensor(float(angle)).to(theta))
        return P.rotate_local(theta, Rg.expand(*theta.shape[:-1], 3, 3))
    return Transform(tid, "analytic_rotation", ["axis_rotation"], description, learned=dict(axis=list(axis), angle=float(angle)), fn=fn)


def _test_thetas(n=4, seed=123):
    g = th.Generator().manual_seed(seed)
    u = th.rand(n, 14, generator=g)
    th_ = th.zeros(n, 14)
    th_[:, 3:6] = 0.2 + 0.6 * u[:, 3:6]
    th_[:, 6] = 0.1 + 0.5 * u[:, 6]
    th_[:, 7] = 0.05 * u[:, 7]
    th_[:, 8] = 0.6 + 0.3 * u[:, 8]
    th_[:, 9] = 0.2 + 0.5 * u[:, 9]
    th_[:, 10] = 0.3 * u[:, 10]
    aa = th.randn(n, 3, generator=g)
    th_[:, 11:14] = aa / aa.norm(dim=-1, keepdim=True) * (0.3 + 2.5 * u[:, 0:1])
    return th_


def build_registry(dedup=True) -> Registry:
    reg = Registry()
    rots = cube_rotations()
    cyc = {"Z": P.perm_matrix(P.VARAXIS_PERMS["Z"]).numpy(), "X": P.perm_matrix(P.VARAXIS_PERMS["X"]).numpy()}
    cands = []
    for M in rots:
        for bf in (False, True):
            for tf in (None, "x", "y"):
                gens, desc = [], []
                name = _rot_name(M)
                if name != "id":
                    gens.append(name)
                    desc.append(f"local rotation {name} + matching size permutation")
                if tf:
                    gens.append(f"taper_flip_{tf}")
                    desc.append(f"taper flip via 180° {tf}-rotation (taper'=1/taper, base inset)")
                if bf:
                    gens.append("bulge_flip")
                    desc.append("bulge sign flip")
                if not gens:
                    continue
                Gt = th.tensor(M, dtype=th.float32)

                def fn(theta, Gt=Gt, bf=bf, tf=tf):
                    x = op_taper_flip(theta, tf) if tf else theta
                    x = op_group(x, Gt)
                    return op_bulge_flip(x) if bf else x
                alias = None
                for k, C in cyc.items():
                    if np.allclose(C, M) and not bf and not tf:
                        alias = k
                kind = "analytic" if tf else "group"
                cands.append(Transform("+".join(gens), kind, gens, "; ".join(desc), G=M.tolist(),
                                       size_perm=np.argmax(np.abs(M), 1).tolist(), bulge_flip=bf,
                                       taper_flip=tf, varaxis_alias=alias, fn=fn))
    for perm in itertools.permutations(range(3)):
        if list(perm) == [0, 1, 2]:
            continue
        lab = "".join("xyz"[i] for i in perm)
        cands.append(Transform(f"sizeperm_{lab}", "perm", [f"sizeperm_{lab}"], f"size permutation -> {lab}, rotation unchanged",
                               size_perm=list(perm), fn=lambda t, perm=list(perm): op_size_perm(t, perm)))
    cands.append(Transform("aa_wrap", "wrap", ["aa_wrap"], "axis-angle r -> (|r|-2π) r̂ (same rotation)", aa_op="wrap",
                           fn=lambda t: op_aa(t, "wrap")))
    cands.append(Transform("aa_negate", "wrap", ["aa_negate"], "axis-angle r -> -r (same rotation iff |r| = π)", aa_op="negate",
                           fn=lambda t: op_aa(t, "negate")))
    if not dedup:
        for c in cands:
            reg.add(c)
        return reg
    # dedup by action on random test thetas (prefer the shortest generator list)
    T = _test_thetas()
    seen = {}
    for c in sorted(cands, key=lambda c: (len(c.generators), c.tid)):
        out = c.apply(T)
        if th.allclose(out, T, atol=1e-5):
            continue
        R = P.aa_to_R(out[:, P.ROT]).reshape(len(T), -1)
        key = th.cat([out[:, :11], R, out[:, P.ROT]], 1)
        dup = None
        for k, v in seen.items():
            if th.allclose(v, key, atol=1e-4):
                dup = k
                break
        if dup is None:
            seen[c.tid] = key
            reg.add(c)
        else:
            reg[dup].generators = sorted(set(reg[dup].generators) | {f"alias:{c.tid}"})
    return reg


def apply_transform(reg: Registry, tid: str, theta: th.Tensor) -> th.Tensor:
    return reg[tid].apply(theta)


# ----------------------------------------------------------------------------- rules
_FEATURE_FNS = {}


def features(theta: th.Tensor) -> dict[str, th.Tensor]:
    """Interpretable precondition features (batched)."""
    size = theta[..., 3:6]
    s = size.amax(-1).clamp_min(1e-6)
    inner = 0.5 * size[..., :2].amin(-1)
    return dict(
        aniso_xy=(size[..., 0] - size[..., 1]).abs() / s,
        aniso_xz=(size[..., 0] - size[..., 2]).abs() / s,
        aniso_yz=(size[..., 1] - size[..., 2]).abs() / s,
        roundness=theta[..., 6],
        dilate_rel=theta[..., 7] / s,
        taper=theta[..., 8],
        taper_dev=(theta[..., 8] - 1).abs(),
        abs_bulge=theta[..., 9].abs(),
        onion=theta[..., 10],
        round_minus_inset=theta[..., 6] - (1 - theta[..., 8]),  # r/inner - δ/inner : taper-flip exactness
        rot_angle=theta[..., P.ROT].norm(dim=-1),
        thin=(size.amin(-1) / s),
        dilate_core=theta[..., 7] / (theta[..., 7] + 0.5 * size.amin(-1)).clamp_min(1e-6),
        inner_rel=inner / s,
    )


def eval_precondition(pre: list, theta: th.Tensor) -> th.Tensor:
    """pre: list of disjuncts, each a list of [feature, op, threshold]. Empty list == always true."""
    if not pre:
        return th.ones(theta.shape[:-1], dtype=th.bool, device=theta.device)
    f = features(theta)
    ok = th.zeros(theta.shape[:-1], dtype=th.bool, device=theta.device)
    for conj in pre:
        c = th.ones_like(ok)
        for name, op, thr in conj:
            c &= (f[name] <= thr) if op == "<=" else (f[name] > thr)
        ok |= c
    return ok


_RULES_CACHE = {}


def load_rules(rules_path):
    if rules_path not in _RULES_CACHE:
        import os
        with open(rules_path) as f:
            rules = {r["rule_id"]: r for r in json.load(f)}
        reg = build_registry()
        tj = os.path.join(os.path.dirname(rules_path), "transforms.json")
        if os.path.exists(tj):
            reg.load_learned(tj)
        _RULES_CACHE[rules_path] = (rules, reg)
    return _RULES_CACHE[rules_path]


def apply_rule(rule_id: str, theta: th.Tensor, rules_path: str, check_precondition=True, refine_fn=None):
    """Execute a rewrite rule. Returns (theta', applicable_mask). For learned maps, `refine_fn(theta')`
    (e.g. a few Adam steps on the caller's loss) is applied if given."""
    rules, reg = load_rules(rules_path)
    r = rules[rule_id]
    squeeze = theta.ndim == 1
    th_ = theta[None] if squeeze else theta
    out = reg[r["transform_id"]].apply(th_)
    ok = eval_precondition(r.get("precondition_struct", []), th_) if check_precondition else th.ones(th_.shape[0], dtype=th.bool)
    if (r.get("map") or "").startswith("learned"):
        out = refine_fn(out) if refine_fn is not None else refine_to_shape(th_, out)
    return (out[0], ok[0]) if squeeze else (out, ok)


def refine_to_shape(theta_src, theta_init, steps=150, M=4096, seed=0):
    """Default execution of a learned map: keep the mapped shape params, choose the pose by aligning
    PCA frames over the 24 ambiguities, then refine all params to minimise D(θ_src, θ')."""
    from types import SimpleNamespace
    from .search_retrieval import Descriptor, initial_pose
    from .search_compensated import compensate
    from . import geometry as G
    dev = theta_src.device
    desc = Descriptor(4, dev)
    loc = lambda t: t * th.tensor([0, 0, 0] + [1] * 8 + [0, 0, 0], device=dev)
    cA, VA, _ = desc.frames(loc(theta_src))
    cB, VB, _ = desc.frames(loc(theta_init))
    RA = P.aa_to_R(theta_src[:, P.ROT])
    best, bestD = None, None
    cfg = SimpleNamespace(M_opt=M, seed=seed, chunk=64, inner_steps=steps, inner_lr=0.03)
    cands = []
    for Gk in desc.Gs:
        Rr, tr = initial_pose(cA, VA, cB, VB, Gk.expand(theta_src.shape[0], 3, 3))
        c = theta_init.clone()
        c[:, P.ROT] = P.R_to_aa(Rr @ RA)
        c[:, P.TRANS] = theta_src[:, P.TRANS] + (RA.transpose(-1, -2) @ tr[..., None])[..., 0]
        cands.append(c)
    C = th.stack(cands, 1)  # (B,24,14)
    B = C.shape[0]
    src = theta_src[:, None].expand(B, 24, 14).reshape(-1, 14)
    pose = th.zeros(14, dtype=th.bool)
    pose[0:3] = True
    pose[11:14] = True
    thP, Dp = compensate(src, C.reshape(-1, 14), pose, cfg, steps=max(steps // 3, 5))
    k = Dp.view(B, 24).argmin(1)
    thP = thP.view(B, 24, 14)[th.arange(B), k]
    out, _ = compensate(theta_src, thP, th.ones(14, dtype=th.bool), cfg, steps=steps)
    return out
