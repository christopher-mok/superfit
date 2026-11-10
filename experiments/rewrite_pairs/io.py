"""Run directory layout, Parquet shards (resumable), curves, dedup and consolidation."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch as th

from . import params as P

PAIR_COLUMNS = [
    "pair_id", "phase", "kind", "driver", "mechanism_swap", "barrier_path", "topology_a", "topology_b",
    "topology_class", "prim_class", "source", "sample_id", "theta_a_raw", "theta_b_raw", "theta_a", "theta_b",
    "transform_id", "transform_spec", "residual_after_transform", "d_geo", "iou_ab", "param_dist_raw",
    "rot_geodesic", "barrier", "barrier_t", "loss_barrier", "barrier_neb", "loss_barrier_neb", "gd_crosses",
    "gd_final_d", "near_bounds", "features", "delta_iou", "delta_loss", "curve_file", "qualifies", "fail_reason",
]

_SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("phase", pa.string()), ("kind", pa.string()), ("driver", pa.string()),
    ("mechanism_swap", pa.list_(pa.string())), ("barrier_path", pa.string()), ("topology_a", pa.string()),
    ("topology_b", pa.string()), ("topology_class", pa.string()), ("prim_class", pa.string()),
    ("source", pa.string()), ("sample_id", pa.string()),
    ("theta_a_raw", pa.list_(pa.float32(), 14)), ("theta_b_raw", pa.list_(pa.float32(), 14)),
    ("theta_a", pa.list_(pa.float32(), 14)), ("theta_b", pa.list_(pa.float32(), 14)),
    ("transform_id", pa.string()), ("transform_spec", pa.string()), ("residual_after_transform", pa.float32()),
    ("d_geo", pa.float32()), ("iou_ab", pa.float32()), ("param_dist_raw", pa.float32()), ("rot_geodesic", pa.float32()),
    ("barrier", pa.float32()), ("barrier_t", pa.float32()), ("loss_barrier", pa.float32()),
    ("barrier_neb", pa.float32()), ("loss_barrier_neb", pa.float32()), ("gd_crosses", pa.bool_()),
    ("gd_final_d", pa.float32()), ("near_bounds", pa.list_(pa.string())), ("features", pa.string()),
    ("delta_iou", pa.float32()), ("delta_loss", pa.float32()), ("curve_file", pa.string()),
    ("qualifies", pa.bool_()), ("fail_reason", pa.string()),
])


def git_sha(cwd=None) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return "nogit"


def make_run_id(run_id: str, repo_root: str) -> str:
    if run_id != "auto":
        return run_id
    return time.strftime("%Y%m%d-%H%M") + "-" + git_sha(repo_root)


class RunDir:
    def __init__(self, root: str):
        self.root = root
        for d in ("curves", "figures", "shards"):
            os.makedirs(os.path.join(root, d), exist_ok=True)

    def path(self, *p):
        return os.path.join(self.root, *p)

    # --------------------------------------------------------------- shards
    def shard_path(self, phase: str, table: str, k: int) -> str:
        d = self.path("shards", phase)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{table}_{k:05d}.parquet")

    def shard_done(self, phase: str, k: int) -> bool:
        return os.path.exists(self.shard_path(phase, "done", k))

    def write_shard(self, phase: str, k: int, rows: list[dict], samples: list[dict] | None = None):
        if rows:
            write_pairs(self.shard_path(phase, "rows", k), rows)
        if samples:
            write_samples(self.shard_path(phase, "samples", k), samples)
        pq.write_table(pa.table({"k": [k]}), self.shard_path(phase, "done", k))

    def read_rows(self, phases=None) -> pd.DataFrame:
        files = sorted(glob.glob(self.path("shards", "*", "rows_*.parquet")))
        if phases:
            files = [f for f in files if os.path.basename(os.path.dirname(f)) in phases]
        if not files:
            return pd.DataFrame(columns=PAIR_COLUMNS)
        return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)

    def read_samples(self) -> pd.DataFrame:
        files = sorted(glob.glob(self.path("shards", "*", "samples_*.parquet")))
        if not files:
            return pd.DataFrame(columns=["sample_id", "source", "theta_raw", "theta_param"])
        return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True).drop_duplicates("sample_id")

    def save_curve(self, pair_id: str, **arrays) -> str:
        rel = os.path.join("curves", f"{pair_id}.npz")
        np.savez_compressed(self.path(rel), **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
        return rel

    def save_json(self, name, obj):
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=1, default=_json_default)

    def load_json(self, name, default=None):
        p = self.path(name)
        if not os.path.exists(p):
            return default
        with open(p) as f:
            return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, th.Tensor):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def _row_defaults(r: dict) -> dict:
    out = {c: r.get(c) for c in PAIR_COLUMNS}
    for c in ("mechanism_swap", "near_bounds"):
        out[c] = list(out[c] or [])
    for c in ("topology_a", "topology_b", "transform_spec", "features"):
        if out[c] is not None and not isinstance(out[c], str):
            out[c] = json.dumps(out[c], default=_json_default)
    for c in ("theta_a_raw", "theta_b_raw", "theta_a", "theta_b"):
        v = out[c]
        out[c] = [float(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    return out


def write_pairs(path, rows):
    rows = [_row_defaults(r) for r in rows]
    tbl = pa.Table.from_pylist(rows, schema=_SCHEMA)
    pq.write_table(tbl, path)


def write_samples(path, samples):
    tbl = pa.table({
        "sample_id": [s["sample_id"] for s in samples],
        "source": [s["source"] for s in samples],
        "theta_raw": pa.array([list(map(float, s["theta_raw"])) for s in samples], type=pa.list_(pa.float32(), 14)),
        "theta_param": pa.array([list(map(float, s["theta_param"])) for s in samples], type=pa.list_(pa.float32(), 14)),
    })
    pq.write_table(tbl, path)


def pair_id(phase, sample_id, transform, phi) -> str:
    h = hashlib.blake2b(f"{transform}|{phi:.4f}".encode(), digest_size=4).hexdigest()
    return f"{phase}-{sample_id}-{h}"


def canonical_key(theta_raw: np.ndarray, tol: float) -> tuple:
    t = th.tensor(np.asarray(theta_raw, dtype=np.float32))
    aa = P.R_to_aa(P.aa_to_R(t[None, P.ROT]))[0]  # wrap rotation to angle in [0, π]
    t = t.clone()
    t[P.ROT] = aa
    t[P.TRANS] = t[P.TRANS] + 0.0  # -0.0 -> 0.0
    return tuple(np.round(t.numpy() / tol).astype(np.int64).tolist())


def consolidate(rd: RunDir, tol: float, transform_classes: dict | None = None):
    df = rd.read_rows()
    if len(df) and transform_classes:
        # Phase 1 rows are written before the exact/conditional classes exist: finalise `kind` here so
        # pairs.parquet / candidates.parquet agree with rules.json.
        m = (df.qualifies.astype(bool) & (df.kind != "degenerate_valley") & (df.phase != "assembly")
             & df.transform_id.isin(list(transform_classes)))
        df.loc[m, "kind"] = df.loc[m, "transform_id"].map(
            lambda t: transform_classes[t] if transform_classes[t] in ("exact", "conditional") else "approximate")
    if len(df):
        keys = [canonical_key(a, tol) + canonical_key(b, tol) for a, b in zip(df.theta_a_raw, df.theta_b_raw)]
        df["_k"] = keys
        df = df.drop_duplicates("_k").drop(columns="_k")
    tbl = pa.Table.from_pandas(df[PAIR_COLUMNS], schema=_SCHEMA, preserve_index=False) if len(df) else pa.Table.from_pylist([], schema=_SCHEMA)
    pq.write_table(tbl, rd.path("candidates.parquet"))
    pairs = df[df.qualifies.astype(bool)] if len(df) else df
    tblp = pa.Table.from_pandas(pairs[PAIR_COLUMNS], schema=_SCHEMA, preserve_index=False) if len(pairs) else pa.Table.from_pylist([], schema=_SCHEMA)
    pq.write_table(tblp, rd.path("pairs.parquet"))
    s = rd.read_samples()
    if len(s):
        pq.write_table(pa.Table.from_pandas(s, preserve_index=False), rd.path("samples.parquet"))
    return df, pairs
