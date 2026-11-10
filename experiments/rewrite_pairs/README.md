# Rewrite-pair search ("hill-crossing" primitive pairs)

Implements `superfit_rewrite_pair_search_prompt.md` (phases 0–5).

```bash
pip install pyarrow scikit-learn scikit-image   # + the SuperFit env; faiss-gpu optional (Phase 3b kNN)
# 1) dry run (50 samples), inspect outputs/rewrite_pairs/<run_id>/
python -m experiments.rewrite_pairs.run --phases 0-5 --run_id auto --preset dry
# 2) full run; Phase 4 needs a local copy of the HF release
huggingface-cli download bardofcodes/superfit-primitive-assemblies --repo-type dataset --local-dir data/spa \
    --include "manifest.jsonl" "dataset/toys4k/superfrustum/*"
python -m experiments.rewrite_pairs.run --phases 0-5 --run_id auto --preset full --toys4k_release_root data/spa \
    [--toys4k_mesh_root /path/to/toys4k]     # meshes => real target; otherwise proxy target (noted in summary)
```

Re-running with the same `--run_id` resumes (finished shards are skipped). `--phases` accepts `0-5`, `3b-5`, `0,1,3b`.
`--preset smoke` is a CPU-sized end-to-end check (numbers meaningless).

| module | role |
|---|---|
| `params.py` | 14/17 layouts, raw↔param (algebraic tanh), rotations/quaternions/SLERP, VarAxis fold |
| `geometry.py` | point sampling, D, IoU, paths, barriers, descent check, string relaxation, topology |
| `transforms.py` | transform registry (cube group × bulge flip × taper flip, size perms, axis-angle wrap), learned maps, `apply_rule` |
| `search_group.py` / `search_sweep.py` / `search_compensated.py` / `search_retrieval.py` / `search_assembly.py` | phases 1 / 2 / 3 (+CMA-ES) / 3b / 4 |
| `verify.py` | shared pair evaluation + classification, Phase 0, held-out known pairs |
| `analyze.py` | Phase 5 rules, figures, `summary.md` |
| `io.py`, `config.py`, `run.py` | run dir + Parquet shards, presets, CLI |

## Facts verified in Phase 0 (differ from / sharpen the prompt)
- The squash is the **algebraic** tanh `p = (1-1e-4)·v/√(1+v²)·scale + offset` (`make_packed_var_to_param`), not `torch.tanh`.
- `bulge == 0` exactly is singular (`sign(0)=0` in `map_arc_bulge` collapses x); "neutral" bulge is ±1e-3.
- VarAxis: `batched_varaxis_sf_packed_eval` uses logits[11:14], rotation[14:17]; `..._eval_st` uses rotation[11:14], logits[14:17].
  Class k ≡ plain SF with `R' = P_k R`, `size' = size[perm_k]` (checked to 5e-7). VarAxis therefore covers exactly the two
  120° cyclic transforms `r120[+++]` / `r120[---]` of the registry.

## Interpretation choices (also listed in summary.md where relevant)
- **Descent check**: a qualifying pair already has D(A,B) < ε, so "descent reaches D < ε" is trivially true at step 0.
  `gd_crosses` is instead "descent from A ends in B's basin" (parameter distance to B < ½ of the A–B distance).
- **Nearest transform** = the registry g whose g(A) is closest to B in *parameter* space (must remove ≥75% of the A–B
  distance); ranking by D would let near-identity transforms claim every pair.
- A pair needs `‖raw_A − raw_B‖ > min_sep` (raw axis-angle included, so the axis-angle wrap counts).
- Phase 1 class statistics exclude samples where g(θ) ≈ θ (fixed points are not evidence for an identity).
- Phase 1 sampling snaps 30% of samples to neutral mechanisms / equal sizes so conditional identities are represented.
- Shape-bank shapes are rescaled to bbox half-diagonal 0.5 (the descriptor is scale-normalised; scaled copies are not pairs).
- Phase 4 loss hill: path loss rises > δ_hill·L_A above the endpoints. VarAxis baseline = same K steps with that primitive
  as a VarAxisSF with learnable Gumbel logits (τ 1→0.1).
- Curves are stored for all qualifying pairs except Phase 1/2, where they are capped per shard (thousands of npz otherwise).
