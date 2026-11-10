"""Entry point:  python -m experiments.rewrite_pairs.run --phases 0-5 --run_id auto [--preset full|dry|smoke]

Resumable: re-running with the same --run_id skips finished shards."""
from __future__ import annotations

import argparse
import os
import platform
import random
import sys
import time

import numpy as np
import torch as th

from . import _compat
from .config import get_config
from .io import RunDir, git_sha, make_run_id
from .transforms import build_registry

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ORDER = ["0", "1", "2", "3", "3b", "4", "5"]


def parse_phases(s: str):
    """'0-5' (3 implies 3b), '3b-5', '0,1,3b'."""
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(ORDER[ORDER.index(a): ORDER.index(b) + 1])
            if b == "3":
                out.add("3b")
        else:
            out.add(part)
    unknown = out - set(ORDER)
    if unknown:
        raise SystemExit(f"unknown phases {unknown}; valid: {ORDER}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="0-5", help="e.g. 0-5, 0,1,3b")
    ap.add_argument("--run_id", default="auto")
    ap.add_argument("--preset", default="full", choices=["full", "dry", "smoke"])
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "rewrite_pairs"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--toys4k_release_root", default=None, help="local clone of bardofcodes/superfit-primitive-assemblies")
    ap.add_argument("--toys4k_mesh_root", default=None)
    ap.add_argument("--eps_close", type=float, default=None)
    ap.add_argument("--delta_hill", type=float, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--synthetic_assemblies", type=int, default=None)
    ap.add_argument("--continue_on_phase0_failure", action="store_true")
    a = ap.parse_args(argv)
    cfg = get_config(a.preset, toys4k_release_root=a.toys4k_release_root, toys4k_mesh_root=a.toys4k_mesh_root,
                     eps_close=a.eps_close, delta_hill=a.delta_hill, M=a.M, synthetic_assemblies=a.synthetic_assemblies)
    cfg.device = ("cuda" if th.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    run_id = make_run_id(a.run_id, REPO)
    rd = RunDir(os.path.join(a.out, run_id))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    logf = open(rd.path("log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    prev = rd.load_json("config.json")
    if prev and any(prev["config"].get(k) != v for k, v in (("eps_close", cfg.eps_close), ("delta_hill", cfg.delta_hill))):
        cfg.notes.append(f"thresholds changed on resume: eps_close {prev['config']['eps_close']}->{cfg.eps_close}, "
                         f"delta_hill {prev['config']['delta_hill']}->{cfg.delta_hill}")
    phases = parse_phases(a.phases)
    if "4" in phases and not cfg.toys4k_release_root and cfg.synthetic_assemblies > 0:
        cfg.notes.append("Phase 4 ran on synthetic assemblies (no --toys4k_release_root given).")
    rd.save_json("config.json", dict(
        config=cfg.to_dict(), run_id=run_id, git_sha=git_sha(REPO), torch=th.__version__,
        cuda=th.version.cuda, device=cfg.device, gpu=th.cuda.get_device_name(0) if th.cuda.is_available() else None,
        python=sys.version, platform=platform.platform(), stubbed_modules=_compat.STUBBED,
        dataset_ids=dict(primitive_assemblies="bardofcodes/superfit-primitive-assemblies", root=cfg.toys4k_release_root),
        phases=sorted(phases), argv=sys.argv,
    ))
    log(f"run {run_id} preset={cfg.preset} device={cfg.device} phases={sorted(phases)} -> {rd.root}")
    reg = build_registry()
    reg.save(rd.path("transforms.json"))
    log(f"registry: {len(reg)} transforms after dedup")
    classes = rd.load_json("transform_classes.json")
    t0 = time.time()
    if "0" in phases:
        from .verify import phase0
        rep = phase0(cfg, rd, log)
        if not rep["ok"] and not a.continue_on_phase0_failure:
            log("PHASE 0 FAILED: evaluator does not behave as described. Stopping (see phase0.json).")
            return 1
    if "1" in phases:
        from . import search_group
        classes = search_group.run(cfg, rd, reg, log)
    if "2" in phases:
        from . import search_sweep
        search_sweep.run(cfg, rd, reg, classes, log)
    if "3" in phases:
        from . import search_compensated
        search_compensated.run(cfg, rd, reg, classes, log)
    if "3b" in phases:
        from . import search_retrieval
        search_retrieval.run(cfg, rd, reg, classes, log)
    if "4" in phases:
        reg.load_learned(rd.path("transforms.json"))
        from . import search_assembly
        search_assembly.run(cfg, rd, reg, classes, log)
    if "5" in phases:
        reg.load_learned(rd.path("transforms.json"))
        from . import analyze
        rd.save_json("config.json", {**rd.load_json("config.json"), "config": cfg.to_dict()})
        analyze.run(cfg, rd, reg, classes, log)
    from .geometry import CACHE_STATS
    log(f"sampling cache: {CACHE_STATS['requests']} requests, {CACHE_STATS['computed']} computed, {CACHE_STATS['bbox_computed']} bboxes computed")
    log(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
