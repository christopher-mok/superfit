"""Export a shareable bundle from a finished run:

    python -m experiments.rewrite_pairs.export --run outputs/rewrite_pairs/<run_id> --out experiments/rewrite_pairs/results

Writes
  rewrite_rules.json  curated, self-contained rules (transform spelled out inline, simplified preconditions,
                      parameter layout and feature definitions in the header)
  pairs.parquet       every qualifying pair, one row each, with named parameter columns and its rule_id
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch  # noqa: F401  (before pandas/pyarrow: on Windows pyarrow first breaks torch's c10.dll init)
import numpy as np
import pandas as pd

from . import _compat  # noqa: F401  (must precede superfit imports)
from . import params as P

# --------------------------------------------------------------------------- curation thresholds
MIN_SUPPORT = dict(exact=1, conditional=100, canonicalize=40, approximate=30)
SYMMETRY_MAPS = ("group", "analytic")

PARAM_DOC = {
    "tx": "translation x", "ty": "translation y", "tz": "translation z",
    "sx": "size x (half-extent of the base)", "sy": "size y (half-extent of the base)", "sz": "size z (half-height, the taper axis)",
    "roundness": "corner radius of the base cross-section, relative to min(sx, sy)/2",
    "dilate": "uniform 3D dilation (rounded offset of the whole solid)",
    "taper": "top/base scale along z; 1 = no taper",
    "bulge": "bend of the z axis into an arc; sign = bend direction, 0 is singular (use +-1e-3 as neutral)",
    "onion": "shell thickness ratio; 0 = solid",
    "rx": "rotation, axis-angle x", "ry": "rotation, axis-angle y", "rz": "rotation, axis-angle z",
}

FEATURE_DOC = {
    "aniso_xy": "|sx - sy| / max(size)", "aniso_xz": "|sx - sz| / max(size)", "aniso_yz": "|sy - sz| / max(size)",
    "roundness": "roundness", "dilate_rel": "dilate / max(size)", "taper": "taper", "taper_dev": "|taper - 1|",
    "abs_bulge": "|bulge|", "onion": "onion",
    "round_minus_inset": "roundness - (1 - taper); >= 0 is when a taper flip is exact",
    "rot_angle": "|axis-angle| in radians", "thin": "min(size) / max(size)",
    "dilate_core": "dilate / (dilate + min(size)/2)", "inner_rel": "min(sx, sy)/2 / max(size)",
}

AXIS = {"x": "x", "y": "y", "z": "z"}


# --------------------------------------------------------------------------- preconditions
def _lit(l):
    return (l[0], l[1], float(l[2]))


def _neg(l):
    return (l[0], ">" if l[1] == "<=" else "<=", l[2])


def _tighten(clause):
    """Keep only the tightest bound per (feature, op); drop the clause if it is empty-range."""
    best = {}
    for f, op, t in clause:
        k = (f, op)
        if k not in best:
            best[k] = t
        else:
            best[k] = min(best[k], t) if op == "<=" else max(best[k], t)
    for f in {f for f, _ in best}:
        if (f, "<=") in best and (f, ">") in best and best[(f, ">")] >= best[(f, "<=")]:
            return None
    return frozenset((f, op, t) for (f, op), t in best.items())


def _implies(b, a):
    """Every literal of clause a is at least as loose as some literal of clause b on the same feature/op."""
    for f, op, t in a:
        if not any(f2 == f and op2 == op and (t2 <= t if op == "<=" else t2 >= t) for f2, op2, t2 in b):
            return False
    return True


def simplify_dnf(struct):
    """OR of ANDs from a depth-2 tree -> a shorter equivalent OR of ANDs. [] means always true."""
    cl = {frozenset(_lit(l) for l in c) for c in struct}
    if not cl:
        return []
    changed = True
    while changed:
        changed = False
        cl = {c for c in (_tighten(c) for c in cl) if c is not None}
        if frozenset() in cl:
            return []
        # absorption: if clause B implies clause A (B's region lies inside A's), B is redundant
        for a in list(cl):
            for b in list(cl):
                if a != b and a in cl and b in cl and _implies(b, a):
                    cl.discard(b)
                    changed = True
        # resolution: {L} | X  and  {~L} | X  =>  X
        lst = list(cl)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                for l in a:
                    if _neg(l) in b and (a - {l}) == (b - {_neg(l)}):
                        cl.discard(a), cl.discard(b), cl.add(a - {l})
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        if changed:
            continue
        # A or (~A and B)  =>  A or B   for single-literal clauses A
        for a in [c for c in cl if len(c) == 1]:
            (l,) = tuple(a)
            for b in list(cl):
                if b is not a and _neg(l) in b:
                    cl.discard(b), cl.add(b - {_neg(l)})
                    changed = True
    out = [sorted([list(l) for l in c]) for c in cl]
    return sorted(out, key=lambda c: (len(c), c))


def dnf_text(struct):
    if not struct:
        return "always"
    fmt = lambda t: f"{t:g}"
    parts = [" and ".join(f"{f} {op} {fmt(t)}" for f, op, t in c) for c in struct]
    return parts[0] if len(parts) == 1 else " or ".join(f"({p})" for p in parts)


# --------------------------------------------------------------------------- transforms in words
def _rot_words(name):
    m = re.match(r"r(\d+)\[([+\-0]{3})\]", name)
    if not m:
        return name
    ang, lab = m.groups()
    comps = [{"+": 1, "-": -1, "0": 0}[c] for c in lab]
    nz = [i for i, c in enumerate(comps) if c]
    if len(nz) == 1:
        axis = ("-" if comps[nz[0]] < 0 else "") + "xyz"[nz[0]]
    else:
        axis = "(" + ", ".join(str(c) for c in comps) + ")"
    return f"{ang}° about local {axis}"


def describe_transform(tid, t):
    kind = t.get("kind")
    gens = [g for g in t.get("generators", []) if not g.startswith("alias:")]
    if kind in ("group", "analytic"):
        bits = []
        for g in gens:
            if g.startswith("r") and "[" in g:
                perm = t.get("size_perm") or [0, 1, 2]
                ps = "" if perm == [0, 1, 2] else f", sizes re-ordered to ({', '.join(['sx', 'sy', 'sz'][i] for i in perm)})"
                bits.append(f"re-express the primitive in its own frame turned {_rot_words(g)}, compensating in the world "
                            f"rotation{ps}")
            elif g.startswith("taper_flip_"):
                ax = g[-1]
                bits.append(f"turn the frustum upside down (180° about local {ax}): taper -> 1/taper, base sx, sy shrink by "
                            f"2δ with δ = (1 - taper)·min(sx, sy)/2, roundness reduced by δ"
                            + (", bulge negated" if ax == "y" else ""))
            elif g == "bulge_flip":
                bits.append("negate bulge")
        return "; then ".join(bits) if bits else (t.get("description") or tid)
    if kind == "perm":
        return t.get("description")
    if kind == "wrap":
        return t.get("description")
    if kind == "analytic_rotation":
        return "re-orient the primitive by a non-symmetry rotation while trading the listed shape parameters; approximate"
    if kind == "learned_affine":
        return "trade the listed shape parameters against each other (fitted from example pairs); approximate"
    return t.get("description") or tid


def transform_block(tid, t):
    kind = t.get("kind")
    out = dict(transform_id=tid, type=kind)
    if kind in ("group", "analytic"):
        out.update(local_rotation_matrix=t.get("rotation_matrix"), size_permutation=t.get("size_perm"),
                   negate_bulge=bool((t.get("sign_flips") or {}).get("bulge")), taper_flip_axis=t.get("taper_flip"))
    elif kind == "perm":
        out.update(size_permutation=t.get("size_perm"))
    elif kind == "wrap":
        out.update(axis_angle_op=t.get("aa_op"))
    if kind in ("analytic_rotation", "learned_affine"):
        # Phase 5 stored these under ids that dropped the mechanism direction, so rules with an opposite-direction
        # twin share one stored map (the last, lowest-support one written). Not exported; see SUMMARY.md.
        out = dict(transform_id=None, type=kind, map_exported=False,
                   reason="fitted map not reliable in this run (id collision in Phase 5); refit from this rule's pairs")
    if t.get("varaxis_alias"):
        out["same_as_varaxis_class_change"] = t["varaxis_alias"]
    return out


# --------------------------------------------------------------------------- bundle
def tier_of(r):
    if r["degenerate_valley"]:
        return "canonicalize" if r["map"] in SYMMETRY_MAPS and r["support"] >= MIN_SUPPORT["canonicalize"] else None
    if r["rule_id"] == "prim_swap":
        return None  # assembly-level, reported separately
    if r["kind"] == "exact" and r["map"] in SYMMETRY_MAPS:
        return "exact"
    if r["kind"] == "conditional" and r["map"] in SYMMETRY_MAPS and r["support"] >= MIN_SUPPORT["conditional"]:
        return "conditional"
    if r["kind"] == "approximate" and r["support"] >= MIN_SUPPORT["approximate"]:
        return "approximate"
    return None


def pair_keys(p):
    """Phase 5's grouping key per pair (analyze.extract_rules), with mechanism directions intact."""
    key = [t if (t is not None and t != "none") else "mech:" + "|".join(sorted(m)) + f"@{d}"
           for t, m, d in zip(p.transform_id, p.mechanism_swap, p.driver)]
    return [("valley:" + k) if kind == "degenerate_valley" else k for k, kind in zip(key, p.kind)]


def readable_id(key):
    return key.replace("↑", "+").replace("↓", "-").replace("mech:", "")


def rule_entry(r, rid, T, p1):
    tid = r["transform_id"]
    t = T.get(tid, {})
    pre = simplify_dnf(r.get("precondition_struct") or [])
    topo = r.get("topology_breakdown") or {}
    ntopo = sum(topo.values()) or None
    st = p1.get(tid) or {}
    ev = dict(
        support_pairs=r["support"], pairs_by_phase=r["phases"],
        mean_shape_distance=round(r["mean_d_geo"], 6),
        mean_barrier=None if r["mean_barrier"] is None or np.isnan(r["mean_barrier"]) else round(r["mean_barrier"], 4),
        precondition_balanced_accuracy=None if r["precondition_train_bal_acc"] is None else round(r["precondition_train_bal_acc"], 3),
        same_topology_fraction=None if not ntopo else round(topo.get("same", 0) / ntopo, 3),
    )
    if st:
        ev["holds_on_random_primitives"] = round(st["frac_close"], 4)
    if r.get("mean_delta_iou") is not None and not np.isnan(r["mean_delta_iou"]):
        ev["mean_assembly_delta_iou"] = round(r["mean_delta_iou"], 5)
    return dict(
        rule_id=rid,
        what_it_does=describe_transform(tid, t),
        applies_when=dnf_text(pre),
        applies_when_struct=pre,
        parameters_that_change=sorted({m.rstrip("↑↓") for m in r["mechanism_swap"]}),
        transform=transform_block(tid, t),
        evidence=ev,
        example_pair_ids=r["example_pair_ids"][:3],
    )


def build_rules(run, p):
    rules = json.load(open(os.path.join(run, "rules.json"), encoding="utf-8"))
    T = json.load(open(os.path.join(run, "transforms.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(run, "config.json"), encoding="utf-8"))
    p1 = (json.load(open(os.path.join(run, "phase1_stats.json"), encoding="utf-8")) or {}).get("partial", {})
    key_of = dict(zip(p.pair_id, pair_keys(p)))
    tiers = {k: [] for k in ("exact", "conditional", "canonicalize", "approximate")}
    for r in sorted(rules, key=lambda r: -r["support"]):
        k = tier_of(r)
        if k:
            e = rule_entry(r, readable_id(key_of[r["example_pair_ids"][0]]), T, p1)
            e["legacy_rule_id"] = r["rule_id"]
            tiers[k].append(e)
    prim_swap = next((r for r in rules if r["rule_id"] == "prim_swap"), None)
    c = cfg["config"]
    header = dict(
        title="SuperFrustum rewrite rules (hill-crossing primitive pairs)",
        run_id=cfg["run_id"], git_sha=cfg["git_sha"], preset=c["preset"],
        what_a_rule_is=("Two SuperFrustum parameter settings A and B that give (nearly) the same shape, separated by a hill: "
                        "the straight path between them passes through clearly different shapes, so gradient descent "
                        "started at A does not reach B. A rule maps A to B in one step."),
        tiers={
            "exact": "holds for every primitive; the jump is free",
            "conditional": "holds when applies_when is true; check it first",
            "canonicalize": "A and B are the same shape with no hill between them (a flat valley); pick one canonical "
                            "form rather than jumping",
            "approximate": "B is only close to A (and the map is fitted, not derived); apply, then refine a few steps. "
                           "Low confidence",
        },
        thresholds=dict(shape_distance_eps=c["eps_close"], barrier_delta=c["delta_hill"],
                        shape_distance="mean |sdf_A - sdf_B| over near-surface + uniform samples, divided by the primitive scale",
                        barrier="max shape distance along the straight A->B path above the endpoints"),
        selection=dict(min_support=MIN_SUPPORT, note="rules below these supports are in the run's rules.json"),
        parameter_layout=[dict(index=i, name=n, meaning=PARAM_DOC[n],
                               range=None if i >= P.N_BOUNDED else [round(float(P.PMIN[i]), 4), round(float(P.PMAX[i]), 4)])
                          for i, n in enumerate(P.NAMES)],
        feature_definitions=FEATURE_DOC,
        how_to_apply=("theta is the 14-vector in parameter space (parameter_layout). For exact/conditional/canonicalize "
                      "rules the transform block is complete: optional taper flip, then local rotation (rotate the shape "
                      "frame by local_rotation_matrix, compose into the world rotation, re-order sizes by size_permutation), "
                      "then optional bulge negation. Reference code: op_taper_flip / op_group / op_bulge_flip in "
                      "experiments/rewrite_pairs/transforms.py; transforms.apply_rule takes legacy_rule_id with the run's "
                      "rules.json. Check applies_when with feature_definitions first."),
    )
    bundle = dict(**header, counts={k: len(v) for k, v in tiers.items()}, **{f"{k}_rules": v for k, v in tiers.items()})
    if prim_swap:
        bundle["assembly_level"] = dict(
            rule_id="prim_swap", what_it_does="swap the order of two overlapping primitives in the smooth-union assembly",
            support_pairs=prim_swap["support"], example_pair_ids=prim_swap["example_pair_ids"][:3])
    return bundle, rules


def build_pairs(p, curated):
    rule_ids = [readable_id(k) for k in pair_keys(p)]
    tier = {e["rule_id"]: t for t in ("exact", "conditional", "canonicalize", "approximate") for e in curated[f"{t}_rules"]}
    out = pd.DataFrame(dict(
        pair_id=p.pair_id, phase=p.phase, kind=p.kind,
        rule_id=rule_ids,
        tier=[tier.get(r) for r in rule_ids],
        transform_id=p.transform_id.where(p.transform_id != "none"),
        driver=p.driver, source=p.source,
        parameters_that_change=[", ".join(sorted({m.rstrip("↑↓") for m in ms})) for ms in p.mechanism_swap],
        shape_distance=p.d_geo, iou_ab=p.iou_ab, barrier=p.barrier, loss_barrier=p.loss_barrier,
        topology_class=p.topology_class, near_bounds=[", ".join(x) for x in p.near_bounds],
        delta_iou=p.delta_iou, delta_loss=p.delta_loss,
    ))
    A = np.stack(p.theta_a.values).astype(np.float32)
    B = np.stack(p.theta_b.values).astype(np.float32)
    for i, n in enumerate(P.NAMES):
        out[f"a_{n}"] = A[:, i]
    for i, n in enumerate(P.NAMES):
        out[f"b_{n}"] = B[:, i]
    asm = []
    for ph, f in zip(p.phase, p.features):
        if ph != "assembly":
            asm.append(None)
            continue
        d = json.loads(f) if isinstance(f, str) else (f or {})
        asm.append(json.dumps({k: d[k] for k in ("category", "L_A", "L_from_A", "L_from_B", "L_varaxis", "iou_from_A",
                                                 "iou_from_B", "iou_varaxis") if k in d}))
    out["assembly_metrics"] = asm
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    p = pd.read_parquet(os.path.join(a.run, "pairs.parquet"))
    bundle, _ = build_rules(a.run, p)
    with open(os.path.join(a.out, "rewrite_rules.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    pairs = build_pairs(p, bundle)
    pairs.to_parquet(os.path.join(a.out, "pairs.parquet"), index=False, compression="zstd")
    print("rules:", bundle["counts"], "| pairs:", len(pairs), "| distinct rule_ids:", pairs.rule_id.nunique(),
          "| curated ids unique:", len({e["rule_id"] for t in bundle["counts"] for e in bundle[f"{t}_rules"]}) == sum(bundle["counts"].values()))


if __name__ == "__main__":
    main()
