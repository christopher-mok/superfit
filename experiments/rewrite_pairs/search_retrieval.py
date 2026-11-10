"""Phase 3b: cross-mechanism retrieval. Shape bank -> pose-invariant descriptor -> kNN -> far-in-raw-space,
unexplained neighbours -> pose alignment + joint refinement -> barrier/descent/topology -> clusters -> learned maps."""
from __future__ import annotations

import itertools
import json
import math

import numpy as np
import torch as th

from . import geometry as G
from . import params as P
from .search_compensated import compensate
from .verify import evaluate_pairs, known_pairs

# bank parameter box (param space): index, lo, hi, neutral band
BOX = {
    "roundness": (6, 0.0, 0.999, None),
    "dilate": (7, 0.0, 0.45, (0.0, 0.01)),
    "taper": (8, 0.05, 1.95, (0.98, 1.02)),
    "bulge": (9, -1.95, 1.95, (-0.02, 0.02)),
    "onion": (10, 0.0, 0.95, (0.0, 0.02)),
}
SIZE_LO, SIZE_HI = 0.03, 1.0


def _uniform(n, g):
    t = th.zeros(n, 14)
    t[:, 3:6] = th.exp(math.log(SIZE_LO) + th.rand(n, 3, generator=g) * (math.log(SIZE_HI) - math.log(SIZE_LO)))
    for name, (i, lo, hi, _) in BOX.items():
        u = th.rand(n, generator=g)
        if name == "dilate":
            u = u ** 2
        t[:, i] = lo + (hi - lo) * u
    return t


def _set_extreme(t, rows, name, g, mode):
    """mode: 'lo' | 'hi' | 'neutral' -- within 10% of the range end, or in the neutral band."""
    if name.startswith("size"):
        ax = {"size_x": 3, "size_y": 4, "size_z": 5}[name]
        f = th.rand(len(rows), generator=g)
        t[rows, ax] = th.where(th.tensor(mode == "hi").expand(len(rows)), 0.9 + 0.1 * f, SIZE_LO + 0.05 * f) * t[rows, 3:6].amax(1).clamp_min(0.3)
        return
    i, lo, hi, neu = BOX[name]
    f = th.rand(len(rows), generator=g)
    w = hi - lo
    if mode == "lo":
        t[rows, i] = lo + 0.1 * w * f
    elif mode == "hi":
        t[rows, i] = hi - 0.1 * w * f
    elif neu is not None:
        t[rows, i] = neu[0] + (neu[1] - neu[0]) * f


def bank_thetas(cfg, n=None):
    """40% uniform over the box, 30% one parameter at a range end or neutral, 30% two extremes at once.
    Every shape is then rescaled (size & dilate) so its tight bbox half-diagonal is 0.5 -- the
    descriptor is scale-normalised, so without this scaled copies would be retrieved as 'pairs'."""
    n = n or cfg.bank_size
    g = th.Generator().manual_seed(cfg.seed + 77)
    t = _uniform(n, g)
    names = list(BOX) + ["size_x", "size_y", "size_z"]
    modes = ["lo", "hi", "neutral"]
    n1, n2 = int(0.4 * n), int(0.7 * n)
    pick = th.randint(len(names), (n,), generator=g)
    mpick = th.randint(3, (n,), generator=g)
    for r in range(n1, n):
        _set_extreme(t, [r], names[pick[r]], g, modes[mpick[r]])
    pick2 = th.randint(len(names), (n,), generator=g)
    mpick2 = th.randint(2, (n,), generator=g)
    for r in range(n2, n):
        nm = names[pick2[r]]
        if nm == names[pick[r]]:
            nm = names[(pick2[r] + 1) % len(names)]
        _set_extreme(t, [r], nm, g, modes[mpick2[r]])
    t[:, 9] = th.where(t[:, 9].abs() < P.BULGE_TINY, th.full_like(t[:, 9], P.BULGE_TINY), t[:, 9])
    t = P.var_to_param(P.param_to_var(t))
    return normalise_scale(t)


def normalise_scale(t, target=0.5, chunk=512):
    out = t.clone()
    for sl in G._chunks(t.shape[0], chunk):
        lo, hi = G.local_bbox(t[sl], pad=0.0)
        c = target / (0.5 * (hi - lo).norm(dim=-1)).clamp_min(1e-4)
        out[sl, 3:6] = (t[sl, 3:6] * c[:, None]).clamp(float(P.PMIN[3]), float(P.PMAX[3]))
        out[sl, 7] = (t[sl, 7] * c).clamp(0.0, float(P.PMAX[7]))
    return out


# ----------------------------------------------------------------------------- descriptor
def cube_group_np():
    from .transforms import cube_rotations
    return cube_rotations()


class Descriptor:
    """Truncated SDF on a res^3 grid in the PCA frame (scaled by the bounding radius). The 24 sign/axis
    ambiguities of the frame are handled at query time by index-permuting the (symmetric) grid."""

    def __init__(self, res, device, trunc=0.2):
        self.res, self.dev, self.trunc = res, device, trunc
        lin = th.linspace(-1, 1, res, device=device)
        self.grid = th.stack(th.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
        self.Gs = [th.tensor(M, dtype=th.float32, device=device) for M in cube_group_np()]
        ijk = th.stack(th.meshgrid(*[th.arange(res, device=device)] * 3, indexing="ij"), -1).reshape(-1, 3)
        perms = []
        for Gk in self.Gs:
            gk = self.grid @ Gk  # rows: (G_k^T g_i)^T
            idx = th.round((gk + 1) / 2 * (res - 1)).long()
            perms.append(idx[:, 0] * res * res + idx[:, 1] * res + idx[:, 2])
        self.perms = th.stack(perms)  # (24, res^3)

    def frames(self, theta, M=2048, seed=0):
        s = G.sample_points(theta * th.tensor([0, 0, 0] + [1] * 8 + [0, 0, 0], device=theta.device), M, seed)
        surf = s.pts[:, : s.n_surf]
        c = surf.mean(1)
        X = surf - c[:, None]
        C = X.transpose(1, 2) @ X / X.shape[1]
        ev, V = th.linalg.eigh(C)
        V = V.flip(-1)  # descending
        # deterministic sign: third moment positive along each axis
        proj = X @ V
        sg = th.sign((proj ** 3).mean(1))
        sg = th.where(sg == 0, th.ones_like(sg), sg)
        V = V * sg[:, None, :]
        det = th.linalg.det(V)
        V[:, :, 2] *= det[:, None]
        r = X.norm(dim=-1).amax(1).clamp_min(1e-4)
        return c, V, r

    @th.no_grad()
    def compute(self, theta, chunk=256, seed=0):
        outs, frames = [], []
        for sl in G._chunks(theta.shape[0], chunk):
            t = theta[sl]
            c, V, r = self.frames(t, seed=seed)
            x = c[:, None] + r[:, None, None] * (self.grid[None] @ V.transpose(1, 2))
            f = G.local_sdf(x, t) / r[:, None]
            outs.append((f.clamp(-self.trunc, self.trunc) / self.trunc).cpu())  # bank descriptors live on CPU
            frames.append((c, V, r))
        c = th.cat([f[0] for f in frames])
        V = th.cat([f[1] for f in frames])
        r = th.cat([f[2] for f in frames])
        return th.cat(outs), (c, V, r)

    def variants(self, d):
        return d.to(self.perms.device)[:, self.perms]  # (B, 24, D)


class Index:
    def __init__(self, bank_desc, pca_dim, device):
        X = bank_desc
        sub = X[th.randperm(X.shape[0], generator=th.Generator().manual_seed(0))[: min(20000, X.shape[0])]].to(device)
        self.mu = sub.mean(0)
        q = min(pca_dim, sub.shape[0] - 1, sub.shape[1])
        _, _, Vt = th.linalg.svd(sub - self.mu, full_matrices=False)
        self.W = Vt[:q].T.contiguous()
        self.Z = th.cat([self.project(X[i:i + 8192].to(device)) for i in range(0, X.shape[0], 8192)]).contiguous()
        self.faiss = None
        try:
            import faiss  # noqa
            idx = faiss.IndexFlatL2(self.Z.shape[1])
            idx.add(self.Z.cpu().numpy().astype(np.float32))
            self.faiss = idx
        except Exception:
            pass

    def project(self, D):
        return (D - self.mu) @ self.W

    def search(self, Q, k, block=20000):
        """Q (n,p) -> (dist (n,k), idx (n,k)) squared L2."""
        if self.faiss is not None:
            d, i = self.faiss.search(Q.cpu().numpy().astype(np.float32), k)
            return th.tensor(d, device=Q.device), th.tensor(i, device=Q.device)
        best_d = th.full((Q.shape[0], k), float("inf"), device=Q.device)
        best_i = th.zeros((Q.shape[0], k), dtype=th.long, device=Q.device)
        for s0 in range(0, self.Z.shape[0], block):
            Zb = self.Z[s0:s0 + block]
            d = th.cdist(Q, Zb).pow(2)
            kk = min(k, Zb.shape[0])
            dv, iv = d.topk(kk, largest=False)
            cd = th.cat([best_d, dv], 1)
            ci = th.cat([best_i, iv + s0], 1)
            o = cd.topk(k, largest=False).indices
            best_d, best_i = th.gather(cd, 1, o), th.gather(ci, 1, o)
        return best_d, best_i


def knn_variants(index, desc, query_desc, k, chunk=128):
    """For each query, k nearest bank items under min over the 24 frame variants.
    Returns (dist, idx, variant) each (n,k) on the index device (idx = -1 where fewer than k found)."""
    out_d, out_i, out_v = [], [], []
    dev = index.Z.device
    for sl in G._chunks(query_desc.shape[0], chunk):
        V = desc.variants(query_desc[sl].to(dev))  # (c,24,D)
        c = V.shape[0]
        Q = index.project(V.reshape(c * 24, -1))
        d, i = index.search(Q, k)
        d, i = d.to(dev).view(c, 24 * k), i.to(dev).view(c, 24 * k)
        var = th.arange(24, device=dev).repeat_interleave(k)[None].expand(c, -1)
        # sort by distance, then stable-sort by id: the first entry of each id run is its best variant
        o = th.argsort(d, 1)
        d, i, var = th.gather(d, 1, o), th.gather(i, 1, o), th.gather(var, 1, o)
        o2 = th.sort(i, dim=1, stable=True).indices
        i2 = th.gather(i, 1, o2)
        first = th.ones_like(i2, dtype=th.bool)
        first[:, 1:] = i2[:, 1:] != i2[:, :-1]
        dup = th.zeros_like(first)
        dup.scatter_(1, o2, ~first)
        d = th.where(dup | (i < 0), th.full_like(d, float("inf")), d)
        kk = min(k, d.shape[1])
        top = d.topk(kk, 1, largest=False).indices
        dd, ii, vv = th.gather(d, 1, top), th.gather(i, 1, top), th.gather(var, 1, top)
        ii = th.where(th.isinf(dd), th.full_like(ii, -1), ii)
        out_d.append(dd), out_i.append(ii), out_v.append(vv)
    return th.cat(out_d), th.cat(out_i), th.cat(out_v)


def shape_dist(a, b):
    w = (P.PMAX - P.PMIN)[3:11].to(a)
    return ((a[..., 3:11] - b[..., 3:11]) / w).norm(dim=-1)


def explained(reg, a, b, tol):
    """Is θ_b's shape (pose ignored) a registry transform of θ_a's shape?"""
    out = th.zeros(a.shape[0], dtype=th.bool, device=a.device)
    for t in reg:
        if t.kind.startswith("learned"):
            continue
        out |= shape_dist(t.apply(a), b) < tol
    return out


def initial_pose(cA, VA, cB, VB, Gk):
    """R' = V_B G_k V_A^T,  t' = c_A - R'^T c_B   (aligns B's PCA frame onto A's)."""
    R = VB @ Gk @ VA.transpose(-1, -2)
    t = cA - (R.transpose(-1, -2) @ cB[..., None])[..., 0]
    return R, t


def refine_pairs(thA, thB0, cfg):
    """Pose-only alignment, then joint refinement of θ_B against D(θ_A, θ_B)."""
    pose = th.zeros(14, dtype=th.bool)
    pose[0:3] = True
    pose[11:14] = True
    thB, _ = compensate(thA, thB0, pose, cfg, steps=max(cfg.refine_steps // 2, 5))
    allf = th.ones(14, dtype=th.bool)
    return compensate(thA, thB, allf, cfg, steps=cfg.refine_steps)


def run(cfg, rd, reg, transform_classes, log):
    dev = cfg.device
    if rd.shard_done("retrieval", 0):
        return
    bank = bank_thetas(cfg).to(dev)
    desc = Descriptor(cfg.desc_res, dev)
    D, (c, V, r) = desc.compute(bank, seed=cfg.seed)
    index = Index(D, cfg.desc_pca_dim, dev)
    log(f"phase3b: bank {bank.shape[0]} shapes, descriptor dim {D.shape[1]} -> {index.Z.shape[1]} (faiss={index.faiss is not None})")
    k = cfg.knn_k
    dist, nn, var = knn_variants(index, desc, D, k + 1)
    dist, nn, var = dist.to(dev), nn.to(dev), var.to(dev)
    # descriptor validation on exact identities
    val = validate_descriptor(cfg, reg, transform_classes, bank, desc, index, D)
    # candidate selection
    ii = th.arange(bank.shape[0], device=dev)[:, None].expand_as(nn)
    ok = (nn >= 0) & (nn != ii) & (nn > ii)
    a_idx, b_idx, v_idx, d_val = ii[ok], nn[ok], var[ok], dist[ok]
    far = shape_dist(bank[a_idx], bank[b_idx]) > cfg.tau_far
    a_idx, b_idx, v_idx, d_val = a_idx[far], b_idx[far], v_idx[far], d_val[far]
    if len(a_idx):
        ex = explained(reg, bank[a_idx], bank[b_idx], 0.1 * cfg.tau_far)
        a_idx, b_idx, v_idx, d_val = a_idx[~ex], b_idx[~ex], v_idx[~ex], d_val[~ex]
    order = th.argsort(d_val)[: cfg.max_refine]
    a_idx, b_idx, v_idx = a_idx[order], b_idx[order], v_idx[order]
    log(f"phase3b: {len(order)} far, unexplained neighbour candidates to refine")
    rows = []
    if len(a_idx):
        Gs = th.stack(desc.Gs)
        R0, t0 = initial_pose(c[a_idx], V[a_idx], c[b_idx], V[b_idx], Gs[v_idx])
        thB0 = bank[b_idx].clone()
        thB0[:, 11:14] = P.R_to_aa(R0)
        thB0[:, 0:3] = t0
        thA = bank[a_idx]
        thB, Dref = refine_pairs(thA, thB0, cfg)
        keep = (shape_dist(thA, thB) > 0.5 * cfg.tau_far) & (Dref < cfg.near_miss_factor * cfg.eps_close)
        keep &= ~explained(reg, thA, thB, 0.1 * cfg.tau_far)
        kk = th.nonzero(keep).flatten()
        metas = [dict(phase="retrieval", driver="retrieval", sample_id=f"b{int(a_idx[j]):07d}", source="synthetic", prim_class="SF",
                      transform_id=None, phi=float(b_idx[j]), spec=dict(partner=f"b{int(b_idx[j]):07d}", frame_variant=int(v_idx[j])),
                      features=dict(desc_dist=float(d_val[order][j]) if len(d_val) else None))
                 for j in kk.tolist()]
        rows = evaluate_pairs(thA[kk], thB[kk], metas, cfg, rd, reg, transform_classes, do_topology=True, mech_pose=False)
    samples = [dict(sample_id=f"b{i:07d}", source="bank", theta_raw=P.param_to_var(bank[i]).cpu(), theta_param=bank[i].cpu())
               for i in sorted(set(a_idx.tolist()) | set(b_idx.tolist()))]
    learned = promote_clusters(rows, reg, rd, log, min_support=2 if cfg.preset == "smoke" else 5)
    held = held_out(cfg, reg, bank, desc, index, rows, log)
    rd.save_json("retrieval_validation.json", dict(descriptor=val, held_out=held, learned=list(learned)))
    th.save(dict(bank=bank.cpu(), mu=index.mu.cpu(), W=index.W.cpu(), Z=index.Z.cpu(), c=c.cpu(), V=V.cpu(), r=r.cpu(),
                 res=cfg.desc_res), rd.path("shape_bank.pt"))
    rd.write_shard("retrieval", 0, rows, samples)
    log(f"phase3b: {sum(r_['qualifies'] for r_ in rows)} pairs; topology classes "
        f"{ {k_: sum(1 for r_ in rows if r_['qualifies'] and r_['topology_class'] == k_) for k_ in ('same', 'eps_equivalent', 'different')} }")


@th.no_grad()
def validate_descriptor(cfg, reg, transform_classes, bank, desc, index, D, n=64):
    """Exact identities (shape-changing ones, e.g. r90 + size swap) must be nearest neighbours."""
    tids = [t for t, c_ in (transform_classes or {}).items() if c_ == "exact" and t in reg.t and reg[t].G is not None]
    tids = tids or ["r90[00+]", "r180[00+]+bulge_flip"]
    sel = th.arange(min(n, bank.shape[0]), device=bank.device)
    res = {}
    for tid in tids[:6]:
        gb = reg[tid].apply(bank[sel])
        Dg, _ = desc.compute(gb, seed=cfg.seed)
        d_q, i_q, _ = knn_variants(index, desc, Dg, 5)
        hit1 = (i_q[:, 0].to(sel.device) == sel).float().mean()
        hit5 = (i_q.to(sel.device) == sel[:, None]).any(1).float().mean()
        res[tid] = dict(recall_at_1=float(hit1), recall_at_5=float(hit5))
    return res


def promote_clusters(rows, reg, rd, log, min_support=5):
    """Cluster qualifying retrieval pairs by mechanism-swap signature and fit raw-space affine maps."""
    from .transforms import make_learned
    groups = {}
    for r_ in rows:
        if r_["qualifies"] and r_["kind"] != "degenerate_valley":
            groups.setdefault(tuple(sorted(r_["mechanism_swap"])), []).append(r_)
    learned = []
    idx = list(range(3, 11))
    for sig, rs in groups.items():
        if len(rs) < min_support:
            continue
        X = th.stack([th.as_tensor(r_["theta_a_raw"])[idx] for r_ in rs]).double()
        Y = th.stack([th.as_tensor(r_["theta_b_raw"])[idx] for r_ in rs]).double()
        Xa = th.cat([X, th.ones(len(X), 1, dtype=X.dtype)], 1)
        lam = 1e-2
        Wb = th.linalg.solve(Xa.T @ Xa + lam * th.eye(Xa.shape[1], dtype=X.dtype), Xa.T @ Y)
        pred = Xa @ Wb
        r2 = 1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum().clamp_min(1e-9)
        tid = "learned:" + "|".join(sig)
        spec = dict(idx=idx, W=Wb[:-1].T.float().tolist(), b=Wb[-1].float().tolist(), r2=float(r2), support=len(rs),
                    note="shape raws only; executing = apply map then refine pose/shape locally")
        reg.add(make_learned(tid, spec, f"learned affine map for mechanism swap {sig}", ["retrieval_cluster"]))
        for r_ in rs:
            r_["transform_id"] = tid
        learned.append(tid)
        log(f"phase3b: learned map {tid} support={len(rs)} R²={float(r2):.3f}")
    reg.save(rd.path("transforms.json"))
    return learned


def held_out(cfg, reg, bank, desc, index, rows, log, k=100):
    """Check known pairs after the run (they were not used to design or tune the search)."""
    out = {}
    dev = bank.device
    for name, q, pred in known_pairs(dev):
        qn = normalise_scale(q[None])
        Dq, (cq, Vq, _) = desc.compute(qn, seed=cfg.seed)
        d, i, v = knn_variants(index, desc, Dq, min(k, bank.shape[0]))
        i = i[0].to(dev)
        hits = [j for j, b in enumerate(i.tolist()) if b >= 0 and bool(pred(bank[b]))]
        rec = dict(rank=None, residual=None, n_partner_like_in_bank=int(pred(bank).sum()))
        if hits:
            j = hits[0]
            b = int(i[j])
            _, (cb, Vb, _) = desc.compute(bank[b:b + 1], seed=cfg.seed)
            R0, t0 = initial_pose(cq, Vq, cb, Vb, desc.Gs[int(v[0, j])][None])
            init = bank[b:b + 1].clone()
            init[:, 11:14], init[:, 0:3] = P.R_to_aa(R0), t0
            _, Dr = refine_pairs(qn, init, cfg)
            rec.update(rank=j + 1, residual=float(Dr[0]), partner=bank[b].tolist())
        found = [r_["pair_id"] for r_ in rows if r_["qualifies"] and (bool(pred(th.as_tensor(r_["theta_a"]))) or bool(pred(th.as_tensor(r_["theta_b"]))))]
        rec["retrieval_pairs_with_partner_mechanism"] = found[:20]
        out[name] = rec
        log(f"held-out {name}: rank={rec['rank']} residual={rec['residual']}")
    return out
