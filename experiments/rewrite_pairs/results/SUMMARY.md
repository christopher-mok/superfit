# SuperFrustum rewrite pairs: results

Run `20261109-2358-9c3c3aa` (full preset, RTX 4080, Nov 9–11 2026). Raw run directory:
`outputs/rewrite_pairs/20261109-2358-9c3c3aa/` on branch `rewrite-pair-search`.

## What this is

A **rewrite pair** is two SuperFrustum parameter settings A and B that produce (nearly) the same shape but sit on
opposite sides of a hill: the straight path from A to B passes through clearly different shapes, so gradient descent
started at A does not reach B. If the fitter knows the pair, it can jump from A to B in one step, for example to escape
a configuration that is stuck, or to pick a canonical form.

The search found **49,395 such pairs** and distilled them into rules of the form *"if the primitive satisfies
condition C, transform T gives the same shape"*.

## Files

| File | What it is |
|---|---|
| `rewrite_rules.json` | 107 curated rules, self-contained: the transform spelled out, a simplified precondition, the evidence for each rule, and a header documenting the 14-parameter layout, ranges and feature definitions |
| `pairs.parquet` | All 49,395 pairs, one row each: both parameter vectors as named columns (`a_tx` … `b_rz`), shape distance, barrier, topology, phase, and the `rule_id` / `tier` it belongs to |
| `SUMMARY.md` | This file |

To regenerate: `python -m experiments.rewrite_pairs.export --run outputs/rewrite_pairs/<run_id> --out <dir>`.

## Headline results

- **One always-valid rule.** Turning the primitive 180° about its own height axis and negating `bulge` gives the
  identical shape for every SuperFrustum (4,604 pairs, mean barrier 0.36). Gradient descent cannot make this move
  because the intermediate rotations are different shapes.
- **Taper flips hold for 84% of random primitives.** Turning a tapered frustum upside down (`taper → 1/taper`, base
  shrunk by the top inset, roundness reduced to match) gives the same shape whenever `taper > 0.477` or the base is
  very thin relative to the height (4,609 and 4,323 pairs about the y and x axes).
- **180° and 90° re-orientations hold when the mechanisms that break symmetry are near neutral.** 180° about local x
  needs `taper` close to 1 (33% of random primitives). 90° and 180° about local z need `|bulge|` below about 0.05
  (about 17%).
- **Equivalences with no hill, to canonicalise rather than jump (23 rules).** These include axis-angle wrap
  (`|r| → |r| − 2π`, always), size permutations when two sizes are nearly equal, and some taper flips.
- **On real assemblies the jumps help 28% of the time.** On 50 Toys4K SuperFrustum assemblies (882 primitives), 3,930
  rewrite opportunities were tried. 1,096 of them (28%; median per assembly 24%) ended at a lower loss than both
  continued descent and a VarAxis baseline with learnable axis choice. The rate varies widely: `giraffe_005` 74%,
  `cow_023` 69%, but `frog_025` 4% and `drum_005` 0 of 2. See the Phase 4 caveat below before reading this as an
  improvement.

## The rules at a glance

| Rule | Tier | Applies when | Pairs | Holds on random primitives |
|---|---|---|---|---|
| `r180[00+]+bulge_flip` | exact | always | 4,604 | 100% |
| `taper_flip_y` | conditional | `inner_rel ≤ 0.038` or `taper > 0.477` | 4,609 | 84% |
| `taper_flip_x` | conditional | same | 4,323 | 84% |
| `r180[+00]` | conditional | `inner_rel ≤ 0.061` or `taper_dev ≤ 0.18` | 1,857 | 33% |
| `r180[+00]+taper_flip_y` | conditional | `inner_rel ≤ 0.053` or `taper_dev ≤ 0.22` | 1,683 | 34% |
| `r180[0+0]+bulge_flip` | conditional | `inner_rel ≤ 0.061` or `taper_dev ≤ 0.18` | 1,607 | 33% |
| `r90[00±]+taper_flip_x` | conditional | `abs_bulge ≤ 0.045` and `taper > 0.38` | 713 each | 15% |
| `r180[00+]` | conditional | `abs_bulge ≤ 0.056` | 544 | 17% |
| `r90[00±]` | conditional | `abs_bulge ≤ 0.045` | 410 / 402 | 17% |
| `valley:aa_wrap` | canonicalize | always | 5,000 | 100% |
| `valley:sizeperm_*` | canonicalize | the two swapped sizes within 4–10% | 86–243 | 2–5% |

`r180[+-0]` means 180° about the local axis (1, −1, 0); `r120[+++]` means 120° about the cube diagonal. Feature names
(`inner_rel`, `taper_dev`, …) are defined in the JSON header. The 61 conditional rules include many variants of the
same symmetry (for example 120° turns about the eight cube diagonals), so there are only 24 distinct preconditions
among them.

## How to use a rule

`theta` is the 14-vector `[tx ty tz | sx sy sz | roundness dilate taper bulge onion | rx ry rz]` in parameter space.
1. Evaluate `applies_when` using the feature definitions in the JSON header.
2. Apply the `transform` block in this order: the optional taper flip, then the local rotation (compose
   `local_rotation_matrix` into the world rotation and re-order sizes by `size_permutation`), then the optional bulge
   negation.

The reference implementation is `op_taper_flip`, `op_group` and `op_bulge_flip` in
`experiments/rewrite_pairs/transforms.py`. `transforms.apply_rule` accepts each rule's `legacy_rule_id` together with
the run's `rules.json`.

## Known issues (results not rerun)

1. **Approximate rules have no usable map (22 rules).** Phase 5 named learned and distilled-rotation maps with IDs that
   dropped the direction of each parameter change (↑/↓). Rules with an opposite-direction twin therefore shared one
   stored map: the last one written, which is the lowest-support one. These maps are **not exported**. Each
   approximate rule's pairs are correct, and in `pairs.parquet` they carry a unique `rule_id`, so a map can be refit
   from them. Exact, conditional and canonicalize rules are unaffected, because their transforms are fixed symmetries,
   not fitted maps. The naming is fixed in `analyze._rule_id` for future runs.
2. **Phase 3b's learned maps were dropped on resume.** `run.py` rewrote `transforms.json` at startup before reloading
   saved maps, so the Phase 4–5 resume lost them, and Phase 5 refit them from the pairs instead. Phase 4 does not use
   these maps. Fixed in `run.py` for future runs.
3. **The ε threshold sits inside a population.** 16,759 near-misses have ε ≤ D < 2ε, against 49,395 that qualified
   (ε = 0.01). Pair counts are sensitive to ε; the rule structure is probably less so.
4. **Phase 4 is limited.**
   - It used 50 of the 200 planned assemblies (`--n_assemblies 50`; same fixed order).
   - Toys4K meshes were not available, so the target is each assembly's own occupancy. The released fit is then
     already optimal, and "beats both baselines" measures equal-loss alternatives, not better fits. ΔIoU values are
     about +0.01 at best.
   - Approximate rules built from Phase 4 pairs are about loss, not shape identity: their mean shape distance can
     exceed ε.
   - `phase4.json` entries for the first 49 assemblies were rebuilt from the log after an interrupted run, so `loss0` is
     missing for them. The per-pair data is unaffected.
5. **One held-out known pair was not recovered.** `dilated_core_sphere ↔ rounded_body` was missed. `taper_flip` and
   `onion_annulus ↔ bulge_ring` were both recovered at rank 1.
6. **Preconditions are shallow trees.** They come from depth-2 decision trees (balanced accuracy 0.78–0.99) and were
   simplified to equivalent shorter forms. Treat thresholds as approximate.
7. **Environment.** This was a Windows run in a venv. Several packages were replaced with stand-ins: `cubvh`, `kaolin`,
   `drjit`, `fastsweep`, `cc3d` and `igl`, none of which SuperFrustum evaluation needs. Phase 0 passed: 90° identity
   D = 3.6e-4, VarAxis-fold error < 5e-7.

## Run details

| Phase | What it searched | Pairs |
|---|---|---|
| 0 | evaluator sanity checks | passed |
| 1 | 102 cube-group × bulge-flip × taper-flip transforms on 5,000 random primitives | 43,291 |
| 2 | 1° rotation sweeps | 400 |
| 3 | compensated sweeps and CMA-ES | 2,671 |
| 3b | kNN retrieval over a 200,000-shape bank | 1,937 |
| 4 | rewrite opportunities on 50 Toys4K assemblies | 1,096 |
| 5 | rule extraction | 725 rules (107 curated here) |

Thresholds: shape distance ε = 0.01 (mean |SDF_A − SDF_B| divided by primitive scale), barrier δ = 0.05. Curation keeps
exact rules; conditional rules with support ≥ 100; canonicalize rules with support ≥ 40 whose transforms are
symmetries; and approximate rules with support ≥ 30. The run's `rules.json` holds all 725.
