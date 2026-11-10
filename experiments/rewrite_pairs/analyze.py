"""Phase 5 (rule extraction) + figures + summary.md."""
from __future__ import annotations

import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
import torch as th

from . import geometry as G
from . import params as P
from .io import consolidate
from .transforms import features, make_axis_rotation, make_learned


def _feat_names():
    return list(features(th.zeros(1, 14)).keys())


def _fit_precondition(pos: np.ndarray, neg: np.ndarray, depth: int):
    """Shallow tree on interpretable features -> (struct, text, train_acc)."""
    names = _feat_names()
    if len(neg) == 0 or len(pos) == 0:
        return [], "", None
    from sklearn.tree import DecisionTreeClassifier
    X = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    w = np.concatenate([np.full(len(pos), 0.5 / len(pos)), np.full(len(neg), 0.5 / len(neg))])
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0, min_samples_leaf=max(1, len(X) // 100))
    clf.fit(X, y, sample_weight=w)
    t = clf.tree_
    paths = []

    def rec(node, conds):
        if t.children_left[node] == -1:
            if t.value[node][0][1] > t.value[node][0][0]:
                paths.append(conds)
            return
        f, thr = names[t.feature[node]], float(t.threshold[node])
        rec(t.children_left[node], conds + [[f, "<=", round(thr, 4)]])
        rec(t.children_right[node], conds + [[f, ">", round(thr, 4)]])
    rec(0, [])
    txt = " or ".join("(" + " and ".join(f"{a} {o} {b}" for a, o, b in c) + ")" if c else "true" for c in paths)
    acc = float(((clf.predict(X) == y) * w).sum() / w.sum())
    return paths, txt, acc


def _phase1_negatives(rd, tid, eps):
    """Features of Phase 1 samples on which transform `tid` is NOT an identity (D >= eps)."""
    pos, neg = [], []
    for f in sorted(glob.glob(rd.path("shards", "group", "dmat_*.npz"))):
        z = np.load(f, allow_pickle=False)
        tids = list(z["tids"])
        if tid not in tids:
            continue
        j = tids.index(tid)
        th_ = th.tensor(z["theta"])
        F = np.stack([v.numpy() for v in features(th_).values()], 1)
        d = z["D"][:, j]
        pos.append(F[d < eps])
        neg.append(F[d >= eps])
    if not pos:
        return None, None
    return np.concatenate(pos), np.concatenate(neg)


def _rows_features(df):
    names = _feat_names()
    out = []
    for f in df.features:
        d = json.loads(f) if isinstance(f, str) else (f or {})
        out.append([d.get(n, np.nan) for n in names])
    return np.array(out, dtype=float).reshape(len(df), len(names))


def _rule_id(s):
    return re.sub(r"[^A-Za-z0-9_+\-|.]", "_", s)[:80]


def extract_rules(cfg, rd, reg, transform_classes, log):
    df, pairs = consolidate(rd, cfg.dedup_tol)
    if len(pairs) == 0:
        rd.save_json("rules.json", [])
        log("phase5: no pairs")
        return []
    pairs = pairs.copy()
    # finalise kinds for hills with a Phase-1 transform
    hill = pairs.kind != "degenerate_valley"
    for ix in pairs.index[hill]:
        tid = pairs.at[ix, "transform_id"]
        if pairs.at[ix, "phase"] != "assembly" and tid in (transform_classes or {}):
            pairs.at[ix, "kind"] = transform_classes[tid] if transform_classes[tid] in ("exact", "conditional") else "approximate"
    rules = []
    groups = {}
    for ix, r in pairs.iterrows():
        valley = r.kind == "degenerate_valley"
        if r.transform_id and r.transform_id != "none":
            key = r.transform_id
        else:
            key = "mech:" + "|".join(sorted(r.mechanism_swap)) + f"@{r.driver}"
        groups.setdefault(("valley:" if valley else "") + key, []).append(ix)
    names = _feat_names()
    for key, ixs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        g = pairs.loc[ixs]
        valley = key.startswith("valley:")
        tkey = key[len("valley:"):] if valley else key
        tid = tkey if tkey in reg.t else None
        mp = None
        if tid is not None:
            t = reg[tid]
            mp = {"group": "group", "analytic": "analytic", "perm": "group", "wrap": "group"}.get(t.kind, t.kind)
        elif len(g) >= 2 or True:
            # distil an executable map for transform-less groups
            spec = [json.loads(s) if isinstance(s, str) else (s or {}) for s in g.transform_spec]
            axes = [s.get("axis") for s in spec if s.get("axis") is not None and s.get("angle") is not None]
            if g.driver.iloc[0] == "rotation" and len(axes) == len(g):
                ax = np.median(np.array(axes), 0)
                ang = float(np.median([s["angle"] for s in spec]))
                tid = "rot:" + _rule_id(tkey)
                reg.add(make_axis_rotation(tid, (ax / max(np.linalg.norm(ax), 1e-9)).tolist(), ang, f"axis rotation distilled from {tkey}"))
                mp = "analytic"
            else:
                idx = list(range(3, 11))
                X = np.stack(g.theta_a_raw.values)[:, idx].astype(float)
                Y = np.stack(g.theta_b_raw.values)[:, idx].astype(float)
                Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
                Wb = np.linalg.solve(Xa.T @ Xa + 1e-2 * np.eye(Xa.shape[1]), Xa.T @ Y)
                tid = "learned:" + _rule_id(tkey)
                reg.add(make_learned(tid, dict(idx=idx, W=Wb[:-1].T.tolist(), b=Wb[-1].tolist(), support=len(g)),
                                     f"learned affine map for {tkey}", ["rule_extraction"]))
                mp = "learned_affine"
        # precondition
        pos_f = _rows_features(g)
        neg_f = None
        if tid in (transform_classes or {}):
            p1, n1 = _phase1_negatives(rd, tid, cfg.eps_close)
            if p1 is not None and len(p1):
                pos_f, neg_f = p1, n1
        if neg_f is None:
            near = df[(df.driver == g.driver.iloc[0]) & (~df.qualifies.astype(bool))]
            neg_f = _rows_features(near) if len(near) else np.zeros((0, len(names)))
        pos_f = np.nan_to_num(pos_f)
        neg_f = np.nan_to_num(neg_f)
        exact = tid is not None and (transform_classes or {}).get(tid) == "exact"
        struct, txt, acc = ([], "", None) if exact else _fit_precondition(pos_f, neg_f, cfg.tree_depth)
        kinds = g.kind.value_counts()
        topo = g.topology_class.dropna().value_counts().to_dict()
        rules.append(dict(
            rule_id=_rule_id(key), transform_id=tid, kind=str(kinds.index[0]),
            mechanism_swap=sorted({m for ms in g.mechanism_swap for m in ms})[:6],
            topology_class=max(topo, key=topo.get) if topo else None, topology_breakdown=topo,
            map=mp, precondition=txt, precondition_struct=struct, precondition_train_bal_acc=acc,
            support=int(len(g)), phases=g.phase.value_counts().to_dict(),
            mean_residual=float(np.nanmean(g.residual_after_transform.astype(float))) if g.residual_after_transform.notna().any() else None,
            mean_d_geo=float(g.d_geo.mean()), mean_barrier=float(np.nanmean(g.barrier.astype(float))),
            mean_delta_iou=float(np.nanmean(g.delta_iou.astype(float))) if g.delta_iou.notna().any() else None,
            varaxis_covered=bool(tid in reg.t and reg[tid].varaxis_alias is not None) if tid else False,
            degenerate_valley=valley, example_pair_ids=list(g.pair_id.iloc[:5]),
        ))
    rd.save_json("rules.json", rules)
    reg.save(rd.path("transforms.json"))
    log(f"phase5: {len(rules)} rules ({sum(r['degenerate_valley'] for r in rules)} degenerate-valley groups)")
    return rules


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def hump(cfg, theta, axis, title, fn, rd):
    dev = cfg.device
    phis = th.deg2rad(th.arange(0, 361, 2.0)).to(dev)
    base = theta.to(dev)[None].expand(len(phis), 14).clone()
    rot = P.rotate_local(base, P.axis_rot(th.tensor(axis, device=dev).float().expand(len(phis), 3), phis))
    S, _ = G.D_and_iou(base, rot, G.sample_points(base, cfg.M_path, cfg.seed), G.sample_points(rot, cfg.M_path, cfg.seed), cfg.chunk)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot(th.rad2deg(phis).cpu(), S.cpu())
    ax.axhline(cfg.eps_close, ls=":", c="g", lw=0.8)
    ax.axhline(cfg.delta_hill, ls=":", c="r", lw=0.8)
    ax.set_xlabel("φ (deg)")
    ax.set_ylabel("D")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(rd.path("figures", fn), dpi=110)
    plt.close(fig)
    return fn


def curve_plot(rd, row, fn, title):
    z = np.load(rd.path(row.curve_file))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    if "d_from_a" in z:
        ax.plot(z["t"], z["d_from_a"], label="D(A,θ(t)) straight")
        ax.plot(z["t"], z["d_to_b"], label="D(θ(t),B) straight", ls="--")
    if "neb_t" in z:
        ax.plot(z["neb_t"], z["neb_d_from_a"], label="D(A,θ(t)) string")
    if "assembly_loss_path" in z:
        ax.plot(np.linspace(0, 1, len(z["assembly_loss_path"])), z["assembly_loss_path"], label="assembly loss")
    ax.set_xlabel("t")
    ax.legend(fontsize=6)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(rd.path("figures", fn), dpi=110)
    plt.close(fig)
    return fn


def figures(cfg, rd, pairs, rules, log):
    figs = {}
    try:
        figs["cube"] = "phase0_cube_hump.png" if os.path.exists(rd.path("figures", "phase0_cube_hump.png")) else \
            hump(cfg, P.default_theta(), [0, 0, 1], "cube, local z", "hump_cube.png", rd)
        figs["near_square_box"] = hump(cfg, P.default_theta(sx=0.5, sy=0.47, sz=0.35, roundness=0.2), [0, 0, 1],
                                       "near-square box 0.5×0.47×0.35, local z", "hump_near_square_box.png", rd)
        figs["bent_cylinder"] = hump(cfg, P.default_theta(sx=0.25, sy=0.25, sz=0.8, roundness=0.999, bulge=0.6), [0, 0, 1],
                                     "bent cylinder (bulge 0.6), local z = bend direction", "hump_bent_cylinder.png", rd)
        if len(pairs):
            eps_eq = pairs[(pairs.phase == "retrieval") & (pairs.topology_class == "eps_equivalent") & pairs.curve_file.notna()]
            for n_, (_, r) in enumerate(eps_eq.head(2).iterrows()):
                figs[f"eps_equivalent_{n_}"] = curve_plot(rd, r, f"hump_eps_equivalent_{n_}.png", f"retrieval ε-equivalent {r.pair_id}")
            asm = pairs[(pairs.phase == "assembly") & pairs.curve_file.notna()]
            if len(asm):
                r = asm.iloc[0]
                figs["toys4k_primitive"] = curve_plot(rd, r, "hump_assembly_primitive.png", f"assembly {r.source}")
            plt = _plt()
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            for ph, g in pairs.groupby("phase"):
                ax.scatter(g.d_geo, g.barrier, s=6, label=ph, alpha=0.6)
            ax.axvline(cfg.eps_close, ls=":", c="g")
            ax.axhline(cfg.delta_hill, ls=":", c="r")
            ax.set_xlabel("residual D(A,B)")
            ax.set_ylabel("barrier")
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(rd.path("figures", "barrier_vs_residual.png"), dpi=110)
            plt.close(fig)
            figs["barrier_vs_residual"] = "barrier_vs_residual.png"
        if rules:
            top = sorted(rules, key=lambda r: -r["support"])[:20]
            plt = _plt()
            fig, ax = plt.subplots(figsize=(6, 0.25 * len(top) + 1))
            ax.barh([r["rule_id"][:40] for r in top][::-1], [r["support"] for r in top][::-1])
            ax.set_xlabel("support")
            ax.tick_params(labelsize=6)
            fig.tight_layout()
            fig.savefig(rd.path("figures", "rule_coverage.png"), dpi=110)
            plt.close(fig)
            figs["rule_coverage"] = "rule_coverage.png"
    except Exception as e:
        log(f"figures: {e!r}")
    return figs


# ----------------------------------------------------------------------------- summary
def _tbl(df):
    if df is None or len(df) == 0:
        return "_none_\n"
    cols = list(df.columns)
    out = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(_fmt(r[c]) for c in cols) + " |\n"
    return out


def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4g}"
    return str(v)


def summary(cfg, rd, rules, figs, log):
    pairs = pd.read_parquet(rd.path("pairs.parquet")) if os.path.exists(rd.path("pairs.parquet")) else pd.DataFrame()
    cands = pd.read_parquet(rd.path("candidates.parquet")) if os.path.exists(rd.path("candidates.parquet")) else pd.DataFrame()
    conf = rd.load_json("config.json", {})
    L = [f"# Rewrite-pair search — run `{os.path.basename(rd.root)}`\n",
         f"Preset `{cfg.preset}`; ε_close={cfg.eps_close}, δ_hill={cfg.delta_hill}, M={cfg.M}. "
         f"Stubbed modules: {conf.get('stubbed_modules')}.\n"]
    if cfg.notes:
        L.append("**Notes / deviations:**\n" + "".join(f"- {n}\n" for n in cfg.notes))
    p0 = rd.load_json("phase0.json", {})
    if p0:
        L.append("## Phase 0\n```\n" + json.dumps(p0.get("checks", {}), indent=1) + "\n```\n")
    L.append("## Pairs per phase and kind\n")
    if len(pairs):
        L.append(_tbl(pairs.groupby(["phase", "kind"]).size().rename("n").reset_index()))
    else:
        L.append("_no pairs_\n")
    if len(cands):
        L.append("\nCandidates (incl. near-misses) by failure reason:\n" +
                 _tbl(cands.fail_reason.replace("", "qualified").value_counts().rename_axis("reason").reset_index(name="n")))
        nm = cands[(cands.d_geo >= cfg.eps_close) & (cands.d_geo < 2 * cfg.eps_close)]
        L.append(f"\nThreshold calibration: {len(nm)} near-misses with ε ≤ D < 2ε; "
                 f"{int(((cands.barrier > 0.5 * cfg.delta_hill) & (cands.barrier <= cfg.delta_hill)).sum())} with ½δ < barrier ≤ δ. "
                 "Large counts here mean the thresholds sit inside a population and should be revisited (record changes in config.json).\n")
    st = rd.load_json("phase1_stats.json", {})
    if st:
        s = st.get("partial", {})
        dfc = pd.DataFrame([dict(transform=k, cls=v.get("class"), frac_close=v.get("frac_close"), hills=v.get("hill")) for k, v in s.items()])
        dfc = dfc[dfc.cls != "none"].sort_values("frac_close", ascending=False)
        L.append("\n## Phase 1 transform classes (non-none)\n" + _tbl(dfc.head(40)))
    if rules:
        R = pd.DataFrame(rules)
        cols = ["rule_id", "kind", "map", "support", "mean_residual", "mean_barrier", "mean_delta_iou", "topology_class", "varaxis_covered", "precondition"]
        L.append("\n## Top 10 rules by support\n" + _tbl(R[~R.degenerate_valley].sort_values("support", ascending=False)[cols].head(10)))
        if R.mean_delta_iou.notna().any():
            L.append("\n## Top 10 rules by ΔIoU\n" + _tbl(R[R.mean_delta_iou.notna()].sort_values("mean_delta_iou", ascending=False)[cols].head(10)))
        L.append("\n## Degenerate valleys (canonicalise, don't jump)\n" + _tbl(R[R.degenerate_valley].sort_values("support", ascending=False)[["rule_id", "support", "mean_d_geo", "mean_barrier", "mechanism_swap"]].head(15)))
        cov = R[~R.degenerate_valley]
        L.append(f"\n## VarAxis baseline coverage\n{int(cov.varaxis_covered.sum())} of {len(cov)} hill rules are reproduced by a VarAxis "
                 "axis-class switch (cyclic permutation r120[±±±] with size permutation). The rest — 90°/180° rotations with size swaps, "
                 "bulge/taper flips, learned cross-mechanism maps — are not reachable by VarAxis' Gumbel switch.\n")
        L.append("Covered: " + ", ".join(cov[cov.varaxis_covered].rule_id) + "\n" if cov.varaxis_covered.any() else "Covered: none\n")
    if len(pairs) and "retrieval" in set(pairs.phase):
        rp = pairs[pairs.phase == "retrieval"].copy()
        rp["sig"] = rp.mechanism_swap.apply(lambda m: "|".join(sorted(m)))
        L.append("\n## Mechanism-swap clusters (retrieval) × topology class\n" +
                 _tbl(rp.groupby(["sig", "topology_class"]).size().rename("n").reset_index().sort_values("n", ascending=False).head(25)))
    rv = rd.load_json("retrieval_validation.json", {})
    if rv:
        L.append("\n## Held-out validation (known pairs, not used for tuning)\n```\n" + json.dumps(rv.get("held_out", {}), indent=1)[:4000] +
                 "\n```\nDescriptor check on exact identities (recall of g(θ) among θ's neighbours):\n```\n" + json.dumps(rv.get("descriptor", {}), indent=1) + "\n```\n")
    p4 = rd.load_json("phase4.json", {})
    if p4:
        L.append("\n## Phase 4 assemblies\nTarget modes: " + json.dumps(p4.get("target_mode", {})) + "\n")
        if any(v == "proxy_self_occupancy" for v in p4.get("target_mode", {}).values()):
            L.append("**Proxy target**: Toys4K meshes were not available, so the loss target is the released assembly's own occupancy. "
                     "Under the proxy the fit is already optimal, so ΔIoU measures equal-loss alternatives, not improvements.\n")
        if len(pairs) and "assembly" in set(pairs.phase):
            a = pairs[pairs.phase == "assembly"].copy()
            a["category"] = a.features.apply(lambda f: json.loads(f).get("category"))
            L.append("\nOpportunities that beat continued descent and the VarAxis baseline, by category:\n" +
                     _tbl(a.groupby(["category", "driver"]).agg(n=("pair_id", "size"), mean_delta_iou=("delta_iou", "mean"),
                                                                mean_delta_loss=("delta_loss", "mean")).reset_index()))
        L.append("\nExtraction checks: " + json.dumps([dict(name=x["name"], check=x["extraction_check"]) for x in p4.get("assemblies", [])])[:3000] + "\n")
    if figs:
        L.append("\n## Figures\n" + "".join(f"- {k}: ![{k}](figures/{v})\n" for k, v in figs.items()))
    L.append("\n## Open questions\n- Are the ε/δ thresholds well placed? See the calibration line above and candidates.parquet.\n"
             "- Straight-line vs string barriers: pairs where barrier_neb ≪ barrier have a low curved path and may not need a rewrite.\n"
             "- Learned (affine) maps are first-order; check residual after the local refine before relying on them.\n")
    with open(rd.path("summary.md"), "w") as f:
        f.write("\n".join(L))
    log(f"summary written: {rd.path('summary.md')}")


def run(cfg, rd, reg, transform_classes, log):
    rules = extract_rules(cfg, rd, reg, transform_classes, log)
    pairs = pd.read_parquet(rd.path("pairs.parquet"))
    figs = figures(cfg, rd, pairs, rules, log)
    summary(cfg, rd, rules, figs, log)
    return rules
