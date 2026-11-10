"""Import shim so the SuperFit evaluators can be imported on machines without CUDA-only deps.

`superfit/__init__.py` eagerly imports the whole package, which pulls in `cubvh`, `kaolin`
and `drjit.cuda`. None of those are needed to evaluate a SuperFrustum SDF. On the GPU box
they are installed and this shim is a no-op; on a laptop it registers inert stub modules
for exactly the names that fail to import, and records them in `STUBBED` so the run's
config.json says so.
"""
from __future__ import annotations

import importlib
import sys
import types

STUBBED: list[str] = []


class _Stub(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {})


def import_superfit():
    for _ in range(40):
        try:
            importlib.import_module("superfit.torch_compute.batched_sf")
            importlib.import_module("superfit.optim.param_conversion.utils")
            return STUBBED
        except ImportError as e:  # ModuleNotFoundError is a subclass
            name = getattr(e, "name", None)
            if not name or name.startswith("superfit") or name in STUBBED:
                raise
            STUBBED.append(name)
            for k in list(sys.modules):
                if k == "superfit" or k.startswith("superfit."):
                    del sys.modules[k]
            m = _Stub(name)
            m.__path__ = []
            sys.modules[name] = m
            parent, _, child = name.rpartition(".")
            if parent and parent in sys.modules:
                setattr(sys.modules[parent], child, m)
    raise RuntimeError("could not import superfit")


import_superfit()
