"""Tests for lab.toml configuration loading."""

from pathlib import Path

import pytest
from dacite import DaciteError

from autoresearch_lab.config import LAB_CONFIG_FILENAME, LabConfig


def _write_lab_toml(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / LAB_CONFIG_FILENAME
    config_path.write_text(content)
    return tmp_path


class TestFindLabRoot:
    def test_finds_in_current_dir(self, tmp_path):
        _write_lab_toml(tmp_path, '[lab]\nname = "test"')
        assert LabConfig.find_lab_root(tmp_path) == tmp_path

    def test_finds_in_parent_dir(self, tmp_path):
        _write_lab_toml(tmp_path, '[lab]\nname = "test"')
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        assert LabConfig.find_lab_root(child) == tmp_path

    def test_raises_when_not_found(self, tmp_path):
        child = tmp_path / "empty"
        child.mkdir()
        with pytest.raises(FileNotFoundError, match="No lab.toml found"):
            LabConfig.find_lab_root(child)


class TestLoad:
    def test_minimal_config(self, tmp_path):
        _write_lab_toml(
            tmp_path,
            """
[lab]
name = "my-lab"
pipeline_dir = "src"

[backend]
module = "eval.py"
class = "MyBackend"
""",
        )
        config = LabConfig.load(tmp_path)
        assert config.name == "my-lab"
        assert config.pipeline_dir == "src"
        assert config.backend.module == "eval.py"
        assert config.backend.cls == "MyBackend"
        assert config.backend.host_service is None
        assert config.sandbox.dockerfile is None

    def test_defaults(self, tmp_path):
        _write_lab_toml(tmp_path, "[lab]\n[backend]")
        config = LabConfig.load(tmp_path)
        assert config.name == tmp_path.name
        assert config.pipeline_dir == "pipeline"
        assert config.agent_instructions == "AGENT.md"
        assert config.results_file == "results.tsv"
        assert config.backend.module == "backend.py"
        assert config.backend.cls == "EvalBackend"

    def test_host_service(self, tmp_path):
        _write_lab_toml(
            tmp_path,
            """
[lab]
name = "with-service"

[backend]
module = "b.py"
class = "B"

[backend.host_service]
command = "python daemon.py"
port = 9100
""",
        )
        config = LabConfig.load(tmp_path)
        hs = config.backend.host_service
        assert hs is not None
        assert hs.command == "python daemon.py"
        assert hs.port == 9100

    def test_sandbox_dockerfile(self, tmp_path):
        _write_lab_toml(
            tmp_path,
            """
[lab]
name = "custom"

[backend]

[sandbox]
dockerfile = "Dockerfile.agent"
""",
        )
        config = LabConfig.load(tmp_path)
        assert config.sandbox.dockerfile == "Dockerfile.agent"

    def test_all_lab_fields(self, tmp_path):
        _write_lab_toml(
            tmp_path,
            """
[lab]
name = "full"
pipeline_dir = "code"
agent_instructions = "INSTRUCTIONS.md"
results_file = "log.tsv"

[backend]
module = "my_backend.py"
class = "Evaluator"
""",
        )
        config = LabConfig.load(tmp_path)
        assert config.name == "full"
        assert config.pipeline_dir == "code"
        assert config.agent_instructions == "INSTRUCTIONS.md"
        assert config.results_file == "log.tsv"
        assert config.backend.module == "my_backend.py"
        assert config.backend.cls == "Evaluator"

    def test_wrong_type_raises(self, tmp_path):
        _write_lab_toml(
            tmp_path,
            """
[lab]
name = "bad"
pipeline_dir = 123

[backend]
""",
        )
        with pytest.raises(DaciteError):
            LabConfig.load(tmp_path)
