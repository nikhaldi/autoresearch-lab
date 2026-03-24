"""Dynamic backend loader.

Loads an EvalBackend implementation from the path specified in lab.toml.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from autoresearch_lab.config import BackendConfig
from autoresearch_lab.harness.backend import EvalBackend


def load_backend(lab_root: Path, backend_config: BackendConfig) -> EvalBackend:
    """Dynamically load the EvalBackend from the lab's backend module."""
    module_path = lab_root / backend_config.module

    if not module_path.exists():
        raise FileNotFoundError(
            f"Backend module not found: {module_path}\n"
            f"Create {backend_config.module} with a class "
            f"implementing EvalBackend."
        )

    spec = importlib.util.spec_from_file_location("lab_backend", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["lab_backend"] = module
    spec.loader.exec_module(module)

    cls = getattr(module, backend_config.cls, None)
    if cls is None:
        raise AttributeError(
            f"Class {backend_config.cls!r} not found in {module_path}"
        )

    instance = cls()
    if not isinstance(instance, EvalBackend):
        raise TypeError(
            f"{backend_config.cls} does not implement EvalBackend"
        )
    return instance
