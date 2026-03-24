"""Tests for dynamic backend loading."""

import pytest

from autoresearch_lab.config import BackendConfig
from autoresearch_lab.harness.backend import EvalBackend
from autoresearch_lab.harness.loader import load_backend

VALID_BACKEND = """\
from pathlib import Path
from autoresearch_lab.harness.backend import EvalBackend, EvalResult

class MyBackend(EvalBackend):
    def evaluate(self, pipeline_dir: Path, data_dir: Path) -> EvalResult:
        return EvalResult(score=0.0)
"""


class TestLoadBackend:
    def test_loads_valid_backend(self, tmp_path):
        (tmp_path / "backend.py").write_text(VALID_BACKEND)
        config = BackendConfig(module="backend.py", cls="MyBackend")

        backend = load_backend(tmp_path, config)

        assert isinstance(backend, EvalBackend)

    def test_missing_module(self, tmp_path):
        config = BackendConfig(module="missing.py", cls="Backend")

        with pytest.raises(FileNotFoundError, match="Backend module not found"):
            load_backend(tmp_path, config)

    def test_missing_class(self, tmp_path):
        (tmp_path / "backend.py").write_text(VALID_BACKEND)
        config = BackendConfig(module="backend.py", cls="WrongName")

        with pytest.raises(AttributeError, match="'WrongName' not found"):
            load_backend(tmp_path, config)

    def test_class_not_eval_backend(self, tmp_path):
        (tmp_path / "backend.py").write_text("class NotABackend: pass\n")
        config = BackendConfig(module="backend.py", cls="NotABackend")

        with pytest.raises(TypeError, match="does not implement EvalBackend"):
            load_backend(tmp_path, config)

    def test_subdirectory_module(self, tmp_path):
        subdir = tmp_path / "backends"
        subdir.mkdir()
        (subdir / "my_eval.py").write_text(VALID_BACKEND)
        config = BackendConfig(module="backends/my_eval.py", cls="MyBackend")

        backend = load_backend(tmp_path, config)

        assert isinstance(backend, EvalBackend)
