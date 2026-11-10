"""Phase 4: rewrite opportunities on real (released Toys4K) SuperFrustum assemblies.

Assembly = left-folded smooth union of primitives (as `smooth_union_k_way` / the saved programs).
Loss = 0.5·mean((occ(sdf) - target)^2), occ = sigmoid(-tanh(sdf·sf)·sf) (the fitting squash).
Target = Toys4K mesh occupancy when meshes + cubvh are available, else the released assembly's own
occupancy (proxy; recorded in config notes)."""
from __future__ import annotations

import io as _io
import json
import math
import os

import numpy as np
import torch as th

from . import geometry as G
from . import params as P
from .search_compensated import DRIVERS
from .search_retrieval import Descriptor, initial_pose, knn_variants, normalise_scale, shape_dist
from .search_sweep import LOCAL_AXES
from .verify import mechanism_swap

from superfit.torch_compute.compile_friendly import _sdf_smooth_union_pair
from superfit.torch_compute.batched_sf import batched_varaxis_sf_packed_eval


# ----------------------------------------------------------------------------- assembly model
class Assembly:
    def __init__(self, theta: th.Tensor, k: th.Tensor, name: str, category: str = ""):
        self.theta = theta  # (n,14)
        self.k = k          # (n-1,)
        self.name, self.category = name, category

    @property
    def n(self):
        return self.theta.shape[0]


def union_sdf(prim_sdf: th.Tensor, k: th.Tensor) -> th.Tensor:
    """prim_sdf (n,...,N) -> (...,N), left fold."""
    out = prim_sdf[0]
    for i in range(1, prim_sdf.shape[0]):
        out = _sdf_smooth_union_pair(out, prim_sdf[i], k[i - 1].reshape(*([1] * out.ndim)))
    return out


def eval_prims(theta, pts):
    return P.sdf(pts[None].expand(theta.shape[0], -1, -1), theta)


def occ(sdf, sf):
    return th.sigmoid(-th.tanh(sdf * sf) * sf)


class LossCtx:
    """Fast batched loss for 'replace primitive i by candidates' with other primitives fixed."""

    def __init__(self, asm: Assembly, pts, target, sf, n_uniform):
        self.asm, self.pts, self.target, self.sf, self.nu = asm, pts, target, sf, n_uniform
        with th.no_grad():
            self.f = eval_prims(asm.theta, pts)  # (n,N)
        self._prefix = {}

    def prefix(self, i):
        """Union of the fixed primitives before slot i (constant during a replace-optimisation)."""
        if i not in self._prefix:
            with th.no_grad():
                self._prefix[i] = union_sdf(self.f[:i], self.asm.k[: i - 1]) if i > 1 else self.f[0]
        return self._prefix[i]

    def loss_full(self, theta, k=None):
        f = eval_prims(theta, self.pts)
        s = union_sdf(f, self.asm.k if k is None else k)
        return 0.5 * ((occ(s, self.sf) - self.target) ** 2).mean(), s

    def loss_replace(self, i, cand):
        """cand (C,14) -> loss (C,), differentiable wrt cand."""
        C = cand.shape[0]
        fi = P.sdf(self.pts[None].expand(C, -1, -1), cand)
        k = self.asm.k
        if i == 0:
            out = fi
        else:
            pre = self.prefix(i)
            out = _sdf_smooth_union_pair(pre[None].expand(C, -1), fi, k[i - 1].reshape(1, 1))
        for j in range(i + 1, self.asm.n):
            out = _sdf_smooth_union_pair(out, self.f[j][None].expand(C, -1), k[j - 1].reshape(1, 1))
        return 0.5 * ((occ(out, self.sf) - self.target[None]) ** 2).mean(-1)

    def iou(self, sdf):
        o = sdf[..., : self.nu] < 0  # uniform points come first
        t = self.target[: self.nu] > 0.5
        return (o & t).sum(-1).float() / (o | t).sum(-1).float().clamp_min(1)


def optimise_replace(ctx, i, init, steps, lr, free=None, chunk=32):
    """Adam on candidates for primitive i (others fixed), `chunk` candidates at a time."""
    fm = th.ones(14, device=init.device) if free is None else free.float().to(init.device)
    outs, losses = [], []
    for sl in G._chunks(init.shape[0], chunk):
        v0 = P.param_to_var(init[sl])
        dv = th.zeros_like(v0, requires_grad=True)
        opt = th.optim.Adam([dv], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            ctx.loss_replace(i, P.var_to_param(v0 + dv * fm)).sum().backward()
            opt.step()
        out = P.var_to_param(v0 + dv.detach() * fm)
        with th.no_grad():
            losses.append(ctx.loss_replace(i, out))
        outs.append(out)
    return th.cat(outs), th.cat(losses)


def optimise_assembly(ctx, theta, steps, lr, seed, varaxis_prim=None):
    """Whole-assembly Adam for K steps (same seed for every arm). varaxis_prim: index of a primitive to
    run as VarAxisSF with learnable Gumbel axis logits (the VarAxis baseline)."""
    th.manual_seed(seed)
    v = P.param_to_var(theta).clone().requires_grad_(True)
    extra = []
    if varaxis_prim is not None:
        logits = th.zeros(1, 3, device=theta.device, requires_grad=True)
        extra = [logits]
    opt = th.optim.Adam([v] + extra, lr=lr)
    for s in range(steps):
        opt.zero_grad()
        p = P.var_to_param(v)
        f = eval_prims(p, ctx.pts)
        if varaxis_prim is not None:
            temp = 1.0 * (0.1 ** (s / max(steps - 1, 1)))
            p17 = th.cat([p[varaxis_prim, :11], logits[0], p[varaxis_prim, 11:14]])[None]
            fv = batched_varaxis_sf_packed_eval(ctx.pts[None], p17, temp)[0]
            f = th.cat([f[:varaxis_prim], fv[None], f[varaxis_prim + 1:]], 0)
        s_ = union_sdf(f, ctx.asm.k)
        (0.5 * ((occ(s_, ctx.sf) - ctx.target) ** 2).mean()).backward()
        opt.step()
    p = P.var_to_param(v.detach())
    if varaxis_prim is not None:
        cls = ["Y", "Z", "X"][int(extra[0].argmax())]
        p17 = th.cat([p[varaxis_prim, :11], extra[0].detach()[0], p[varaxis_prim, 11:14]])
        p = p.clone()
        p[varaxis_prim] = P.varaxis_to_sf(p17, cls)
    with th.no_grad():
        L, s_ = ctx.loss_full(p)
    return p, float(L), float(ctx.iou(s_))


# ----------------------------------------------------------------------------- loading
class _TorchCPUUnpickler:
    """Same loader as superfit.visualize.video_generators.load_info_dict (torch storages -> CPU)."""

    @staticmethod
    def load(path):
        import _pickle as cPickle

        class U(cPickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: th.load(_io.BytesIO(b), map_location="cpu", weights_only=False)
                return super().find_class(module, name)
        with open(path, "rb") as f:
            return U(f).load()


def select_release_rows(root, n, seed):
    rows = [json.loads(l) for l in open(os.path.join(root, "manifest.jsonl")) if l.strip()]
    rows = [r for r in rows if r["source_dataset"] == "toys4k" and r["method"] == "superfrustum" and r["has_primitive_assembly"]
            and os.path.exists(os.path.join(root, r["primitive_assembly_path"]))]
    by = {}
    for r in rows:
        by.setdefault(r.get("category") or "none", []).append(r)
    rng = np.random.default_rng(seed)
    for v in by.values():
        rng.shuffle(v)
    prio = ["chair", "mug", "cup", "bowl", "vase", "bottle", "pot", "hat", "glass", "helmet", "teapot", "bucket",
            "snake", "spring", "spiral", "pretzel", "horn", "hook", "ring", "sofa"]
    cats = [c for c in prio if c in by] + sorted(c for c in by if c not in prio)
    out = []
    while len(out) < n and any(by[c] for c in cats):
        for c in cats:
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    return out


def _val(e, a):
    lt = getattr(e, "lookup_table", {}) or {}
    v = lt.get(a, a)
    return th.as_tensor(v, dtype=th.float32).flatten()


def extract_assembly(info, name, category):
    """Program -> packed (n,14) params + smooth-union ks, via SuperFit's own reduction
    (fetch_singular_expr_eval with fix_axis_variant=True folds VarAxis classes into the rotation)."""
    import geolipi.symbolic as gls
    from superfit.utils.io import get_best_expr
    from superfit.symbolic.utils import gather_primitives, gather_smooth_union_ops, fetch_singular_expr_eval
    expr = get_best_expr(info).tensor(device="cpu")
    markers = gather_primitives(expr)
    ks = [th.as_tensor(k, dtype=th.float32).flatten()[0] for k in gather_smooth_union_ops(expr)]
    thetas, keep = [], []
    for mi, mk in enumerate(markers):
        e = fetch_singular_expr_eval(mk, relaxed_eval=False, remove_marker=True, fix_axis_variant=True,
                                     use_euler_angle=False, device="cpu")
        if isinstance(e, gls.NullExpression3D):
            continue
        e = e.tensor(device="cpu")
        assert isinstance(e, gls.Translate3D), type(e)
        rot = e.args[0]
        assert isinstance(rot, gls.AxisAngleRotate3D), type(rot)
        prim = rot.args[0]
        t = _val(e, e.args[1])
        aa = _val(rot, rot.args[1])
        pa = [_val(prim, a) for a in prim.args]
        vec = th.cat([t, pa[0], *pa[1:6], aa])
        assert vec.numel() == 14, (type(prim), [x.numel() for x in pa])
        thetas.append(vec)
        keep.append(mi)
    ks_kept = [ks[mi - 1] for mi in keep[1:]] if ks else []
    return Assembly(th.stack(thetas), th.stack(ks_kept) if ks_kept else th.zeros(0), name, category), expr


def check_extraction(asm, expr, pts):
    """Compare our packed evaluation with geolipi's recursive_evaluate of the reduced program."""
    try:
        from geolipi.torch_compute import recursive_evaluate, Sketcher
        from superfit.symbolic.utils import fetch_singular_expr_eval
        e = fetch_singular_expr_eval(expr, relaxed_eval=False, remove_marker=True, fix_axis_variant=True, use_euler_angle=False, device="cpu")
        sk = Sketcher(device="cpu", resolution=32, n_dims=3)
        ref = recursive_evaluate(e.tensor(device="cpu"), sk, coords=pts.cpu()).flatten()
        ours = union_sdf(eval_prims(asm.theta.cpu(), pts.cpu()), asm.k.cpu())
        alt = asm.theta.clone()
        alt[:, 11:14] = -alt[:, 11:14]
        ours_t = union_sdf(eval_prims(alt.cpu(), pts.cpu()), asm.k.cpu())
        e1, e2 = float((ref - ours).abs().max()), float((ref - ours_t).abs().max())
        if e2 < e1:  # opposite rotation convention in the program; adopt it
            asm.theta = alt
        return dict(max_err=min(e1, e2), flipped_rotation=e2 < e1)
    except Exception as ex:  # keep going; record why
        return dict(max_err=None, error=repr(ex))


def synthetic_assemblies(cfg, n):
    """Testing without the release: random 3–4 primitive assemblies where primitive 0 of the 'fit'
    sits in a wrong basin (rotated 90° about z without the size swap), target = the true assembly."""
    g = th.Generator().manual_seed(cfg.seed + 5)
    out = []
    for a in range(n):
        m = 3 + a % 2
        t = th.zeros(m, 14)
        t[:, 0:3] = (th.rand(m, 3, generator=g) - 0.5) * 0.8
        t[:, 3:6] = 0.15 + 0.35 * th.rand(m, 3, generator=g)
        t[0, 3:6] = th.tensor([0.6, 0.15, 0.3])
        t[:, 6] = 0.3 * th.rand(m, generator=g)
        t[:, 8] = 1.0
        t[:, 9] = P.BULGE_TINY
        aa = th.randn(m, 3, generator=g)
        t[:, 11:14] = aa / aa.norm(dim=-1, keepdim=True) * 0.4
        k = th.full((m - 1,), 0.01)
        true = Assembly(t.clone(), k, f"synthetic_{a}", "synthetic")
        fit = t.clone()
        fit[0] = P.rotate_local(fit[0:1], P.axis_rot(th.tensor([0.0, 0, 1]), th.tensor([math.pi / 2])))[0]
        out.append((Assembly(fit, k, f"synthetic_{a}", "synthetic"), true))
    return out


def make_points(cfg, asm_for_surface, n, seed, dev):
    g = th.Generator().manual_seed(seed)
    nu = n // 2
    uni = (th.rand(nu, 3, generator=g) * 2 - 1).to(dev)
    s = G.sample_points(asm_for_surface.theta.to(dev), max(2 * (n - nu) // asm_for_surface.n, 64), seed)
    surf = s.pts[:, : s.n_surf].reshape(-1, 3)
    surf = surf[th.randperm(surf.shape[0], generator=g)[: n - nu].to(dev)]
    return th.cat([uni, surf], 0), nu


def mesh_target(cfg, row, pts):
    """Toys4K mesh occupancy (needs the meshes + cubvh on GPU). Returns None if unavailable."""
    if not cfg.toys4k_mesh_root:
        return None
    try:
        import cubvh
        import trimesh
        cat, oid = row.get("category"), row["object_id"]
        cands = [os.path.join(cfg.toys4k_mesh_root, cat, oid, f) for f in ("mesh.obj", f"{oid}.obj")]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            return None
        mesh = trimesh.load(path, force="mesh")
        # SuperFit normalises meshes to the [-1,1] fitting box; reuse its preprocessing if importable
        try:
            from superfit.utils.mesh_preprocess import normalize_mesh
            mesh = normalize_mesh(mesh)
        except Exception:
            v = mesh.vertices - mesh.bounds.mean(0)
            mesh.vertices = v / np.abs(v).max() * 0.95
        bvh = cubvh.cuBVH(mesh.vertices, mesh.faces)
        d, _, _ = bvh.signed_distance(pts, return_uvw=False, mode="watertight")
        return (d <= 0).float()
    except Exception:
        return None


# ----------------------------------------------------------------------------- per-primitive search
def path_loss(ctx, i, a, b, T=17):
    pth = G.path_params(P.param_to_var(a[None]), P.param_to_var(b[None]), T)[0]
    with th.no_grad():
        return ctx.loss_replace(i, pth)


def primitive_candidates(ctx, i, cfg, bank_ctx):
    """Compensated rotation / driver sweeps on assembly loss + bank retrieval for primitive i.
    Returns list of (theta_B, driver, spec)."""
    dev = ctx.pts.device
    a = ctx.asm.theta[i]
    out = []
    steps = max(cfg.inner_steps // 4, 5)
    with th.no_grad():
        LA = float(ctx.loss_replace(i, a[None])[0])
    # rotation
    phis = th.deg2rad(th.arange(cfg.comp_step_deg, 360, cfg.comp_step_deg)).to(dev)
    for an in cfg.comp_axes:
        ax = th.tensor(LOCAL_AXES[an], device=dev).expand(len(phis), 3)
        plain = P.rotate_local(a[None].expand(len(phis), 14).clone(), P.axis_rot(ax, phis))
        inits = [plain]
        for perm in ([1, 0, 2], [0, 2, 1], [2, 1, 0]):
            x = plain.clone()
            x[:, 3:6] = a[3:6][perm]
            inits.append(x)
        init = th.cat(inits)
        free = th.ones(14, dtype=th.bool)
        free[11:14] = False
        thB, L = optimise_replace(ctx, i, init, steps, cfg.inner_lr, free)
        L = L.view(len(inits), -1)
        best = L.argmin(0)
        Lb = L.min(0).values
        thB = thB.view(len(inits), -1, 14)[best, th.arange(len(phis))]
        j = int(Lb.argmin())
        if float(Lb[j]) < LA:
            out.append((thB[j], "rotation", dict(axis=LOCAL_AXES[an], angle=float(phis[j]))))
    # other drivers
    for dname, (di, (lo, hi)) in DRIVERS.items():
        vals = th.linspace(lo, hi, cfg.driver_grid, device=dev)
        vals = vals[(vals - a[di]).abs() > 0.1 * (hi - lo)]
        if len(vals) == 0:
            continue
        init = a[None].expand(len(vals), 14).clone()
        init[:, di] = vals
        free = th.ones(14, dtype=th.bool)
        free[di] = False
        thB, L = optimise_replace(ctx, i, init, steps, cfg.inner_lr, free)
        j = int(L.argmin())
        if float(L[j]) < LA:
            out.append((thB[j], dname, dict(driver_value=float(vals[j]))))
    # bank retrieval with the primitive's own geometry
    if bank_ctx is not None:
        desc, index, bank, Gs = bank_ctx
        loc = a.clone()
        loc[0:3] = 0
        loc[11:14] = 0
        qn = normalise_scale(loc[None])
        cq = float(qn[0, 3:6].amax() / loc[3:6].amax().clamp_min(1e-6))
        Dq, (cA, VA, _) = desc.compute(qn, seed=cfg.seed)  # same seed as the stored bank frames
        d, nn, var = knn_variants(index, desc, Dq, min(cfg.knn_k, bank.shape[0]))
        cands = [(int(b), int(v)) for b, v in zip(nn[0].tolist(), var[0].tolist()) if b >= 0 and float(shape_dist(qn[0], bank[b])) > cfg.tau_far][:5]
        for b, v in cands:
            _, (cB, VB, _) = desc.compute(bank[b:b + 1], seed=cfg.seed)
            Rr, tr = initial_pose(cA, VA, cB, VB, Gs[v][None])
            RA = P.aa_to_R(a[None, 11:14])[0]
            nb = bank[b].clone()
            nb[3:6] = nb[3:6] / cq
            nb[7] = nb[7] / cq
            nb[11:14] = P.R_to_aa((Rr[0] @ RA)[None])[0]
            nb[0:3] = a[0:3] + RA.T @ (tr[0] / cq)
            thB, L = optimise_replace(ctx, i, nb[None], max(cfg.refine_steps, 5), cfg.inner_lr)
            if float(L[0]) < LA:
                out.append((thB[0], "retrieval", dict(partner=int(b))))
    return out, LA


def overlapping_pairs(ctx, frac=0.01):
    inside = ctx.f < 0
    out = []
    for i in range(ctx.asm.n):
        for j in range(i + 1, ctx.asm.n):
            inter = (inside[i] & inside[j]).sum()
            m = min(int(inside[i].sum()), int(inside[j].sum()))
            if m > 0 and int(inter) > frac * m:
                out.append((i, j))
    return out


def run(cfg, rd, reg, transform_classes, log):
    dev = cfg.device
    jobs = []
    proxy_note = None
    if cfg.toys4k_release_root and os.path.exists(os.path.join(cfg.toys4k_release_root, "manifest.jsonl")):
        for row in select_release_rows(cfg.toys4k_release_root, cfg.n_assemblies, cfg.seed):
            jobs.append(("release", row))
    elif cfg.synthetic_assemblies > 0:
        for fit, true in synthetic_assemblies(cfg, cfg.synthetic_assemblies):
            jobs.append(("synthetic", (fit, true)))
    else:
        log("phase4: no --toys4k_release_root and synthetic_assemblies=0 -> skipped")
        rd.save_json("phase4.json", dict(skipped=True, reason="no release root"))
        return
    bank_ctx = None
    if os.path.exists(rd.path("shape_bank.pt")):
        from .search_retrieval import Index
        sb = th.load(rd.path("shape_bank.pt"))
        desc = Descriptor(sb["res"], dev)
        index = Index.__new__(Index)
        index.mu, index.W, index.Z, index.faiss = sb["mu"].to(dev), sb["W"].to(dev), sb["Z"].to(dev), None
        bank_ctx = (desc, index, sb["bank"].to(dev), th.stack(desc.Gs))
    report = dict(assemblies=[], target_mode={})
    for k, (kind, job) in enumerate(jobs):
        if rd.shard_done("assembly", k):
            continue
        if kind == "release":
            path = os.path.join(cfg.toys4k_release_root, job["primitive_assembly_path"])
            info = _TorchCPUUnpickler.load(path)  # official release file only
            asm, expr = extract_assembly(info, job["object_id"], job.get("category") or "")
            asm.theta, asm.k = asm.theta.to(dev), asm.k.to(dev)
            pts, nu = make_points(cfg, asm, cfg.assembly_points, cfg.seed, dev)
            chk = check_extraction(asm, expr, pts[:2048])
            asm.theta = asm.theta.to(dev)
            tgt = mesh_target(cfg, job, pts)
            mode = "mesh" if tgt is not None else "proxy_self_occupancy"
            if tgt is None:
                with th.no_grad():
                    tgt = (union_sdf(eval_prims(asm.theta, pts), asm.k) < 0).float()
            source_name = f"toys4k:{job['object_id']}"
        else:
            asm, true = job
            asm.theta, asm.k = asm.theta.to(dev), asm.k.to(dev)
            true.theta, true.k = true.theta.to(dev), true.k.to(dev)
            pts, nu = make_points(cfg, true, cfg.assembly_points, cfg.seed, dev)
            with th.no_grad():
                tgt = (union_sdf(eval_prims(true.theta, pts), true.k) < 0).float()
            chk, mode, source_name = dict(max_err=0.0), "synthetic_true_assembly", f"synthetic:{asm.name}"
        report["target_mode"][asm.name] = mode
        ctx = LossCtx(asm, pts, tgt, cfg.assembly_occ_scale, nu)
        L0, s0 = ctx.loss_full(asm.theta)
        rows = []
        opps = []
        for i in range(asm.n):
            cands, LA = primitive_candidates(ctx, i, cfg, bank_ctx)
            for thB, driver, spec in cands:
                lp = path_loss(ctx, i, asm.theta[i], thB)
                LB = float(lp[-1])
                bar = float(lp.max()) - max(LA, LB)
                if bar > cfg.delta_hill * max(LA, 1e-4):  # hill in loss, relative to L_A
                    opps.append(dict(i=i, j=None, thB=thB, driver=driver, spec=spec, LA=LA, LB=LB, curve=lp.cpu().numpy(),
                                     barrier=float(lp.max()) - LA, loss_barrier=float(lp.max()) - LB, tid="none"))
        for i, j in overlapping_pairs(ctx):
            sw = asm.theta.clone()
            sw[[i, j]] = sw[[j, i]]
            Lsw, _ = ctx.loss_full(sw)
            opps.append(dict(i=i, j=j, thB=sw, driver="prim_swap", spec=dict(swap=[i, j]), LA=float(L0), LB=float(Lsw), curve=None,
                             barrier=float("nan"), loss_barrier=float("nan"), tid="prim_swap"))
        # the from-A arm (continued descent) is the same for every opportunity of this assembly, and the
        # VarAxis arm only depends on the primitive: run each once (same seed => same result)
        base_arm = optimise_assembly(ctx, asm.theta, cfg.assembly_K, cfg.assembly_lr, cfg.seed) if opps else None
        varaxis_arm = {}
        for o in opps:
            i = o["i"]
            thetaB = o["thB"] if o["j"] is not None else asm.theta.clone().index_copy_(0, th.tensor([i], device=dev), o["thB"][None])
            pa, La, Ia = base_arm
            pb, Lb, Ib = optimise_assembly(ctx, thetaB, cfg.assembly_K, cfg.assembly_lr, cfg.seed)
            if i not in varaxis_arm:
                varaxis_arm[i] = optimise_assembly(ctx, asm.theta, cfg.assembly_K, cfg.assembly_lr, cfg.seed, varaxis_prim=i)
            pv, Lv, Iv = varaxis_arm[i]
            a_i = asm.theta[i]
            b_i = o["thB"][i] if o["j"] is not None else o["thB"]
            with th.no_grad():
                dgeo, iou = G.D_pair(a_i[None], b_i[None], cfg.M_opt, cfg.seed)
            beats = Lb < La - 1e-6 and Lb < Lv - 1e-6
            nb_ = sorted(set(P.near_bounds(a_i, cfg.near_bound_frac)) | set(P.near_bounds(b_i, cfg.near_bound_frac)))
            ta, tb = G.topology(th.stack([a_i, b_i]), cfg.topo_res, cfg.topo_eps)
            ra_, rb_ = P.param_to_var(a_i[None])[0], P.param_to_var(b_i[None])[0]
            gd, rg = G.combined_dist(ra_[None], rb_[None])
            pid = f"assembly-{asm.name}-{i}-{o['driver']}-{len(rows)}"
            curve = rd.save_curve(pid, assembly_loss_path=o["curve"]) if o["curve"] is not None else None
            src = f"{source_name}:{i}" if o["j"] is None else f"{source_name}:{i},{o['j']}"
            rows.append(dict(
                pair_id=pid, phase="assembly", kind="exact" if o["tid"] == "prim_swap" else "approximate", driver=o["driver"],
                mechanism_swap=mechanism_swap(a_i[None], b_i[None])[0], barrier_path="straight",
                topology_a=ta, topology_b=tb, topology_class=G.topology_class(ta, tb), prim_class="SF", source=src,
                sample_id=f"{asm.name}:{i}", theta_a_raw=ra_.cpu(), theta_b_raw=rb_.cpu(), theta_a=a_i.cpu(), theta_b=b_i.cpu(),
                transform_id=o["tid"], transform_spec=dict(o["spec"]), residual_after_transform=float("nan"),
                d_geo=float(dgeo[0]), iou_ab=float(iou[0]), param_dist_raw=float(gd[0]), rot_geodesic=float(rg[0]),
                barrier=o["barrier"], barrier_t=float("nan"), loss_barrier=o["loss_barrier"], barrier_neb=float("nan"),
                loss_barrier_neb=float("nan"), gd_crosses=bool(La <= o["LB"] + 1e-6), gd_final_d=float("nan"), near_bounds=nb_,
                features=dict(L_A=o["LA"], L_B_init=o["LB"], L_from_A=La, L_from_B=Lb, L_varaxis=Lv, iou_from_A=Ia, iou_from_B=Ib,
                              iou_varaxis=Iv, beats_continued=Lb < La, beats_varaxis=Lb < Lv, category=asm.category, target_mode=mode),
                delta_iou=Ib - Ia, delta_loss=Lb - La, curve_file=curve, qualifies=bool(beats),
                fail_reason="" if beats else "does_not_beat_baselines",
            ))
        samples = [dict(sample_id=f"{asm.name}:{i}", source=f"{source_name}:{i}", theta_raw=P.param_to_var(asm.theta[i:i + 1])[0].cpu(),
                        theta_param=asm.theta[i].cpu()) for i in range(asm.n)]
        rd.write_shard("assembly", k, rows, samples)
        report["assemblies"].append(dict(name=asm.name, category=asm.category, n_prims=asm.n, extraction_check=chk, target_mode=mode,
                                         loss0=float(L0), n_opps=len(opps), n_beat=sum(r["qualifies"] for r in rows)))
        log(f"phase4 {asm.name}: {asm.n} prims, {len(opps)} opportunities, {sum(r['qualifies'] for r in rows)} beat both baselines "
            f"(target={mode}, extraction max_err={chk.get('max_err')})")
    prev = rd.load_json("phase4.json", {}) or {}
    prev.setdefault("assemblies", []).extend(report["assemblies"])
    prev.setdefault("target_mode", {}).update(report["target_mode"])
    rd.save_json("phase4.json", prev)
