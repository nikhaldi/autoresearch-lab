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
    def evaluate(
        self,
        pipeline_dir: Path,
        data_dir: Path,
        sample_ids: list[str] | None = None
    ) -> EvalResult:
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

    def test_eval_caches_result(self, tmp_path, monkeypatch):
        runner = self._init_lab(tmp_path, monkeypatch)

        runner.invoke(cli, ["eval", "--data", "mydata"])

        cache_path = tmp_path / ".last_eval.json"
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text())
        assert cached["score"] == 0.042
        assert cached["metrics"] == {"accuracy": 0.958, "latency_ms": 123.4}


class TestVerdict:
    def _init_lab_with_eval(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])
        (tmp_path / "pipeline").mkdir(exist_ok=True)
        (tmp_path / "mydata").mkdir()
        (tmp_path / BackendConfig.module).write_text(EVAL_BACKEND)
        # Run eval to populate the cache
        runner.invoke(cli, ["eval", "--data", "mydata"])
        return runner

    def test_verdict_writes_file(self, tmp_path, monkeypatch):
        runner = self._init_lab_with_eval(tmp_path, monkeypatch)
        vpath = tmp_path / "verdicts" / "verdict.json"

        result = runner.invoke(
            cli,
            [
                "verdict",
                "--action",
                "keep",
                "--verdict-path",
                str(vpath),
                "--experiment-id",
                "exp_001",
                "--notes",
                "improved accuracy",
            ],
        )

        assert result.exit_code == 0
        assert vpath.exists()
        verdict = json.loads(vpath.read_text())
        assert verdict["action"] == "keep"
        assert verdict["score"] == 0.042
        assert verdict["metrics"] == {"accuracy": 0.958, "latency_ms": 123.4}
        assert verdict["experiment_id"] == "exp_001"
        assert verdict["notes"] == "improved accuracy"

    def test_verdict_without_experiment_id(self, tmp_path, monkeypatch):
        runner = self._init_lab_with_eval(tmp_path, monkeypatch)
        vpath = tmp_path / "verdict.json"

        result = runner.invoke(
            cli,
            ["verdict", "--action", "discard", "--verdict-path", str(vpath)],
        )

        assert result.exit_code == 0
        verdict = json.loads(vpath.read_text())
        assert verdict["action"] == "discard"
        assert "experiment_id" not in verdict

    def test_verdict_requires_prior_eval(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])

        result = runner.invoke(
            cli,
            [
                "verdict",
                "--action",
                "keep",
                "--verdict-path",
                str(tmp_path / "v.json"),
            ],
        )

        assert result.exit_code != 0
        assert "Run `arl eval` first" in result.output


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

        result = runner.invoke(cli, ["diagnose", "--data", "mydata", "--sample", "s1"])

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


class TestPlot:
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
            kept=True,
            commit_sha="aaa111",
            notes="baseline",
        )
        append_result(
            results_path,
            experiment_id="exp_002",
            score=0.12,
            kept=False,
            commit_sha="reverted",
            notes="worse",
        )
        append_result(
            results_path,
            experiment_id="exp_003",
            score=0.05,
            kept=True,
            commit_sha="bbb222",
            notes="better",
        )
        return runner

    def test_saves_to_file(self, tmp_path, monkeypatch):
        """Plot renders a real PNG file."""
        import matplotlib

        matplotlib.use("Agg")

        runner = self._init_lab_with_results(tmp_path, monkeypatch)
        out_path = tmp_path / "out.png"

        result = runner.invoke(cli, ["plot", "--output", str(out_path)])

        assert result.exit_code == 0
        assert "Saved plot" in result.output
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_shows_interactive(self, tmp_path, monkeypatch):
        """Plot calls plt.show() when no output file is given."""
        import matplotlib

        matplotlib.use("Agg")

        runner = self._init_lab_with_results(tmp_path, monkeypatch)

        with patch("matplotlib.pyplot.show") as mock_show:
            result = runner.invoke(cli, ["plot"])

        assert result.exit_code == 0
        mock_show.assert_called_once()

    def test_empty_results(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])

        result = runner.invoke(cli, ["plot"])

        assert result.exit_code == 0
        assert "No results yet" in result.output

    def test_missing_matplotlib(self, tmp_path, monkeypatch):
        """Shows helpful error when matplotlib is not installed."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "test-lab"])

        import builtins
        import sys

        real_import = builtins.__import__

        # Remove cached plot module so the import is attempted fresh
        monkeypatch.delitem(sys.modules, "autoresearch_lab.plot", raising=False)

        def mock_import(name, *args, **kwargs):
            if name == "autoresearch_lab.plot":
                raise ImportError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = runner.invoke(cli, ["plot"])

        assert result.exit_code != 0
        assert "matplotlib is required" in result.output

    def test_multi_lab_comparison(self, tmp_path, monkeypatch):
        """Plot multiple results files in the same chart."""
        import matplotlib

        matplotlib.use("Agg")

        runner = self._init_lab_with_results(tmp_path, monkeypatch)

        # Create a second results file in a sibling directory
        from autoresearch_lab.harness.results import append_result

        other_lab = tmp_path / "other-lab"
        other_lab.mkdir()
        other_results = other_lab / "results.tsv"
        append_result(
            other_results,
            experiment_id="exp_001",
            score=0.20,
            kept=True,
            commit_sha="ccc333",
            notes="other baseline",
        )
        append_result(
            other_results,
            experiment_id="exp_002",
            score=0.15,
            kept=True,
            commit_sha="ddd444",
            notes="other improvement",
        )

        out_path = tmp_path / "comparison.png"
        result = runner.invoke(
            cli,
            ["plot", "--output", str(out_path), str(other_results)],
        )

        assert result.exit_code == 0
        assert "Saved plot" in result.output
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestParseLabelPath:
    def test_explicit_label(self, tmp_path):
        from autoresearch_lab.cli import _parse_label_path

        label, path = _parse_label_path(f"My Lab:{tmp_path / 'r.tsv'}")
        assert label == "My Lab"
        assert path == str(tmp_path / "r.tsv")

    def test_infer_from_lab_toml(self, tmp_path, monkeypatch):
        from autoresearch_lab.cli import _parse_label_path

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["init", "--name", "cool-lab"])

        results_path = tmp_path / LabConfig.results_file
        label, path = _parse_label_path(str(results_path))
        assert label == "cool-lab"
        assert path == str(results_path)

    def test_fallback_to_directory_name(self, tmp_path):
        from autoresearch_lab.cli import _parse_label_path

        sub = tmp_path / "my-dir"
        sub.mkdir()
        results_file = sub / "results.tsv"
        results_file.touch()

        label, path = _parse_label_path(str(results_file))
        assert label == "my-dir"
        assert path == str(results_file)

    def test_existing_file_with_colon_not_split(self, tmp_path):
        """A path that exists should not be split on colons."""
        from autoresearch_lab.cli import _parse_label_path

        # Create a file whose full path is passed as arg
        results_file = tmp_path / "results.tsv"
        results_file.touch()

        label, path = _parse_label_path(str(results_file))
        # Should not try to split on any colon in the path
        assert path == str(results_file)
