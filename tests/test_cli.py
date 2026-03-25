"""Tests for the arl CLI commands."""

import json
from unittest.mock import patch

from click.testing import CliRunner

from autoresearch_lab.cli import cli
from autoresearch_lab.config import LAB_CONFIG_FILENAME, BackendConfig, LabConfig


class TestInit:
    def test_creates_all_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--name", "my-lab"])

        assert result.exit_code == 0
        assert (tmp_path / LAB_CONFIG_FILENAME).exists()
        assert (tmp_path / BackendConfig.module).exists()
        assert (tmp_path / LabConfig.agent_instructions).exists()
        assert (tmp_path / LabConfig.results_file).exists()

    def test_lab_toml_is_valid(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])

        config = LabConfig.load(tmp_path)
        assert config.name == "test-lab"
        assert config.pipeline_dir == "pipeline"

    def test_output_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--name", "my-lab"])

        assert "Initialized lab 'my-lab'" in result.output
        assert "pipeline_dir" in result.output
        assert "AGENT.md" in result.output
        assert BackendConfig.module in result.output

    def test_fails_if_already_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / LAB_CONFIG_FILENAME).write_text("[lab]")

        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--name", "my-lab"])

        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_does_not_overwrite_existing_backend(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        backend_path = tmp_path / BackendConfig.module
        backend_path.write_text("# custom backend")

        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "my-lab"])

        assert backend_path.read_text() == "# custom backend"

    def test_name_substituted_in_agent_md(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "cool-project"])

        agent_md = (tmp_path / LabConfig.agent_instructions).read_text()
        assert "cool-project" in agent_md


class TestRun:
    def _init_lab(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])
        (tmp_path / "pipeline").mkdir(exist_ok=True)
        (tmp_path / "mydata").mkdir()
        return runner

    def test_dry_run(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["run", "--data", "mydata", "--dry-run"])

        assert result.exit_code == 0
        assert "Lab:       test-lab" in result.output
        assert "Pipeline:" in result.output
        assert "Data:      mydata" in result.output
        assert "Model:     claude-opus-4-6" in result.output

    def test_dry_run_custom_options(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                "mydata",
                "--model",
                "claude-sonnet-4-6",
                "--max-iterations",
                "10",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Model:     claude-sonnet-4-6" in result.output

    def test_requires_data_flag(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["run", "--dry-run"])

        assert result.exit_code != 0

    def test_builds_docker_and_runs_session(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        with (
            patch("autoresearch_lab.cli.subprocess.run") as mock_run,
            patch("autoresearch_lab.cli.run_session") as mock_session,
        ):
            mock_run.return_value.returncode = 0

            result = runner.invoke(
                cli, ["run", "--data", "mydata", "--max-iterations", "5"]
            )

        assert result.exit_code == 0
        assert "Building base agent container" in result.output
        assert mock_session.called

        run_cfg = mock_session.call_args[0][0]
        assert run_cfg.max_iterations == 5
        assert run_cfg.data == "mydata"
        assert run_cfg.docker_image == "arl-agent-test-lab"

    def test_docker_build_failure(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        from unittest.mock import MagicMock

        def _mock_run(cmd, **_kwargs):
            result = MagicMock()
            # docker info (preflight) succeeds, docker build fails
            if cmd[0] == "docker" and cmd[1] == "info":
                result.returncode = 0
            else:
                result.returncode = 1
                result.stderr = "build error"
            return result

        with patch("autoresearch_lab.cli.subprocess.run", side_effect=_mock_run):
            result = runner.invoke(cli, ["run", "--data", "mydata"])

        assert result.exit_code != 0
        assert "Docker build failed" in result.output

    def test_custom_dockerfile(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)
        (tmp_path / "Dockerfile").write_text("FROM arl-agent-base\n")

        lab_toml = tmp_path / LAB_CONFIG_FILENAME
        content = lab_toml.read_text() + '\n[sandbox]\ndockerfile = "Dockerfile"\n'
        lab_toml.write_text(content)

        with (
            patch("autoresearch_lab.cli.subprocess.run") as mock_run,
            patch("autoresearch_lab.cli.run_session"),
        ):
            mock_run.return_value.returncode = 0

            result = runner.invoke(cli, ["run", "--data", "mydata"])

        assert result.exit_code == 0
        assert "Building custom agent container" in result.output

    def test_missing_custom_dockerfile(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        lab_toml = tmp_path / LAB_CONFIG_FILENAME
        content = lab_toml.read_text() + '\n[sandbox]\ndockerfile = "Dockerfile"\n'
        lab_toml.write_text(content)

        with patch("autoresearch_lab.cli.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0

            result = runner.invoke(cli, ["run", "--data", "mydata"])

        assert result.exit_code != 0
        assert "sandbox dockerfile not found" in result.output


EVAL_BACKEND = """\
from pathlib import Path
from autoresearch_lab.harness.backend import EvalBackend, EvalResult, SampleResult

class Backend(EvalBackend):
    def evaluate(self, pipeline_dir: Path, data_dir: Path) -> EvalResult:
        return EvalResult(
            score=0.042,
            metrics={"accuracy": 0.958, "latency_ms": 123.4},
            sample_results=[
                SampleResult(sample_id="s1", score=0.03),
                SampleResult(sample_id="s2", score=0.05),
            ],
        )
"""


class TestEval:
    def _init_lab(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])
        (tmp_path / "pipeline").mkdir(exist_ok=True)
        (tmp_path / "mydata").mkdir()
        # Replace stub backend with one that returns known results
        (tmp_path / BackendConfig.module).write_text(EVAL_BACKEND)
        return runner

    def test_prints_score_and_metrics(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["eval", "--data", "mydata"])

        assert result.exit_code == 0

        output = json.loads(result.output)
        assert output["score"] == 0.042
        assert output["accuracy"] == 0.958
        assert output["latency_ms"] == 123.4
        assert output["num_samples"] == 2

    def test_missing_pipeline_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])
        # Don't create pipeline dir
        (tmp_path / "mydata").mkdir()

        result = runner.invoke(cli, ["eval", "--data", "mydata"])

        assert result.exit_code != 0
        assert "pipeline dir not found" in result.output

    def test_missing_data_dir(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["eval", "--data", "nonexistent"])

        assert result.exit_code != 0
        assert "data dir not found" in result.output

    def test_requires_data_flag(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["eval"])

        assert result.exit_code != 0


class TestDiagnose:
    def _init_lab(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])
        (tmp_path / "pipeline").mkdir(exist_ok=True)
        (tmp_path / "mydata").mkdir()
        (tmp_path / BackendConfig.module).write_text(EVAL_BACKEND)
        return runner

    def test_outputs_score_and_samples(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["diagnose", "--data", "mydata"])

        assert result.exit_code == 0

        output = json.loads(result.output)
        assert output["score"] == 0.042
        assert output["metrics"]["accuracy"] == 0.958
        assert len(output["per_sample"]) == 2

    def test_sorted_worst_first(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["diagnose", "--data", "mydata"])

        output = json.loads(result.output)
        scores = [s["score"] for s in output["per_sample"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_flag(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["diagnose", "--data", "mydata", "--top", "1"])

        assert result.exit_code == 0

        output = json.loads(result.output)
        assert len(output["per_sample"]) == 1
        assert output["per_sample"][0]["sample_id"] == "s2"

    def test_sample_flag(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(
            cli, ["diagnose", "--data", "mydata", "--sample", "s1"]
        )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["sample_id"] == "s1"
        assert output["score"] == 0.03
        assert "per_sample" not in output

    def test_sample_flag_not_found(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(
            cli, ["diagnose", "--data", "mydata", "--sample", "nonexistent"]
        )

        assert result.exit_code != 0

    def test_requires_data_flag(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["diagnose"])

        assert result.exit_code != 0


class TestResults:
    def _init_lab_with_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])

        from autoresearch_lab.harness.results import append_result

        results_path = tmp_path / LabConfig.results_file
        append_result(
            results_path,
            experiment_id="exp_001",
            score=0.10,
            metrics={"cer": 0.10, "wer": 0.20},
            kept=True,
            commit_sha="aaa111",
            notes="first",
        )
        append_result(
            results_path,
            experiment_id="exp_002",
            score=0.08,
            metrics={"cer": 0.08, "wer": 0.15},
            kept=False,
            commit_sha="reverted",
            notes="worse",
        )
        append_result(
            results_path,
            experiment_id="exp_003",
            score=0.05,
            metrics={"cer": 0.05, "wer": 0.10},
            kept=True,
            commit_sha="bbb222",
            notes="better",
        )
        return runner

    def test_table_format(self, tmp_path, monkeypatch):
        runner = self._init_lab_with_results(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["results"])

        assert result.exit_code == 0
        assert "exp_001" in result.output
        assert "exp_002" in result.output
        assert "exp_003" in result.output
        assert "KEEP" in result.output
        assert "DISC" in result.output

    def test_json_format(self, tmp_path, monkeypatch):
        runner = self._init_lab_with_results(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["results", "--format", "json"])

        assert result.exit_code == 0

        rows = json.loads(result.output)
        assert len(rows) == 3
        assert rows[0]["experiment_id"] == "exp_001"
        assert rows[2]["experiment_id"] == "exp_003"

    def test_best_flag(self, tmp_path, monkeypatch):
        runner = self._init_lab_with_results(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["results", "--format", "json", "--best"])

        assert result.exit_code == 0

        rows = json.loads(result.output)
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp_003"
        assert rows[0]["score"] == "0.050000"

    def test_last_flag(self, tmp_path, monkeypatch):
        runner = self._init_lab_with_results(tmp_path, monkeypatch)

        result = runner.invoke(cli, ["results", "--format", "json", "--last", "2"])

        assert result.exit_code == 0

        rows = json.loads(result.output)
        assert len(rows) == 2
        assert rows[0]["experiment_id"] == "exp_002"
        assert rows[1]["experiment_id"] == "exp_003"

    def test_empty_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])

        result = runner.invoke(cli, ["results"])

        assert result.exit_code == 0
        assert "No results yet" in result.output
