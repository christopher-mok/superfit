"""Run configuration. `full` follows the task spec; `dry` is the 50-sample dry run; `smoke` is a
CPU-sized check that every phase executes end to end (numbers are not meaningful)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass
class Config:
    preset: str = "full"
    seed: int = 0
    device: str = "auto"
    # thresholds
    eps_close: float = 0.01
    delta_hill: float = 0.05
    near_bound_frac: float = 0.05
    near_miss_factor: float = 3.0       # candidates.parquet keeps D < near_miss_factor * eps_close
    dedup_tol: float = 1e-3
    min_sep: float = 0.05              # min raw-space (+rot rad) separation of A and B
    # sampling
    M: int = 32768
    M_check: int = 131072               # stability check of the metric
    M_path: int = 8192                  # per path point
    M_opt: int = 4096                   # inner optimisations / descent checks (fixed point sets)
    chunk: int = 64
    # paths / checks
    path_T: int = 33
    gd_steps: int = 500
    gd_lr: float = 0.02
    neb_images: int = 9
    neb_iters: int = 60
    run_neb: bool = True
    # phase 1
    n_group: int = 5000
    rot_subset: float = 0.10
    # phase 2
    n_sweep: int = 1000
    sweep_step_deg: float = 1.0
    n_random_axes: int = 3
    # phase 3
    n_comp: int = 300
    comp_step_deg: float = 5.0
    comp_axes: tuple = ("x", "y", "z")
    inner_steps: int = 200
    inner_lr: float = 0.03
    driver_grid: int = 21
    cmaes_samples: int = 50
    cmaes_iters: int = 60
    cmaes_pop: int = 16
    cmaes_lambda: float = 0.5
    # phase 3b
    bank_size: int = 200000
    desc_res: int = 16
    desc_pca_dim: int = 128
    knn_k: int = 20
    tau_far: float = 0.35               # normalised (per-range) param distance, pose excluded
    max_refine: int = 5000
    refine_steps: int = 150
    topo_res: tuple = (48, 80)
    topo_eps: float = 0.01
    # phase 4
    toys4k_release_root: str = ""       # local clone of bardofcodes/superfit-primitive-assemblies
    toys4k_mesh_root: str = ""          # optional Toys4K meshes for a real target
    n_assemblies: int = 200
    assembly_points: int = 65536
    assembly_K: int = 200
    assembly_lr: float = 0.01
    assembly_occ_scale: float = 15.0
    synthetic_assemblies: int = 0       # >0: build synthetic assemblies (testing without the release)
    # phase 5
    tree_depth: int = 2
    # sharding
    shard_size: int = 50
    notes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "full": Config(),
    "dry": Config(preset="dry", n_group=50, n_sweep=50, n_comp=20, cmaes_samples=5, bank_size=5000,
                  max_refine=200, n_assemblies=5, M=8192, M_path=4096, M_opt=2048),
    "smoke": Config(preset="smoke", M=1024, M_check=4096, M_path=512, M_opt=512, path_T=9, gd_steps=40,
                    neb_images=5, neb_iters=8, n_group=16, n_sweep=4, sweep_step_deg=5.0, n_random_axes=1,
                    n_comp=2, comp_step_deg=30.0, comp_axes=("z",), inner_steps=25, driver_grid=5,
                    cmaes_samples=1, cmaes_iters=4, cmaes_pop=6, bank_size=400, desc_res=10, desc_pca_dim=32,
                    knn_k=8, max_refine=12, refine_steps=20, topo_res=(24, 32), n_assemblies=2,
                    assembly_points=4096, assembly_K=15, synthetic_assemblies=2, shard_size=8),
}


def get_config(preset: str, **overrides) -> Config:
    return replace(PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None})
