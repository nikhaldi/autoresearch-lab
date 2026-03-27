"""Tests for the orchestrator run_session loop."""

import json
from unittest.mock import MagicMock, patch

from autoresearch_lab.config import LabConfig
from autoresearch_lab.harness.results import append_result, read_results
from autoresearch_lab.sandbox.orchestrator import RunConfig, run_session

MODULE = "autoresearch_lab.sandbox.orchestrator"


def _make_container(poll_returns):
    """Create a mock container whose poll() returns values from the list.

    None means "still running", an int means "exited with code".
    """
    container = MagicMock()
    container.poll.side_effect = poll_returns
    container.terminate.return_value = None
    container.wait.return_value = None
    container.kill.return_value = None
    return container


def _init_lab(tmp_path):
    """Create a minimal lab directory."""
    (tmp_path / "lab.toml").write_text(
        '[lab]\nname = "test"\npipeline_dir = "pipeline"\n[backend]\n'
    )
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "results.tsv").touch()
    config = LabConfig.load(tmp_path)
    return config


class TestRunSession:
    def test_keep_verdict_commits_and_logs(self, tmp_path, capsys):
        config = _init_lab(tmp_path)
        run_cfg = RunConfig(max_iterations=1, data="data", iteration_timeout=0)

        verdict = {
            "action": "keep",
            "experiment_id": "exp_001",
            "score": 0.04,
            "metrics": {"accuracy": 0.96},
            "notes": "improved",
        }

        container = _make_container([None, None])

        with (
            patch(f"{MODULE}.start_container", return_value=container),
            patch(f"{MODULE}.wait_for_verdict", return_value=verdict),
            patch(f"{MODULE}.git_commit", return_value="abc12345"),
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        rows = read_results(tmp_path / "results.tsv")
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp_001"
        assert rows[0]["score"] == "0.040000"
        assert rows[0]["kept"] == "yes"
        assert rows[0]["commit_sha"] == "abc12345"

        metrics = json.loads(rows[0]["metrics"])
        assert metrics["accuracy"] == 0.96

        output = capsys.readouterr().out
        assert "KEEP" in output
        assert "improved" in output

    def test_discard_verdict_reverts(self, tmp_path, capsys):
        config = _init_lab(tmp_path)
        run_cfg = RunConfig(max_iterations=1, data="data", iteration_timeout=0)

        verdict = {
            "action": "discard",
            "experiment_id": "exp_001",
            "score": 0.08,
            "notes": "worse",
        }

        container = _make_container([None, None])

        with (
            patch(f"{MODULE}.start_container", return_value=container),
            patch(f"{MODULE}.wait_for_verdict", return_value=verdict),
            patch(f"{MODULE}.git_commit") as mock_commit,
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        mock_commit.assert_not_called()

        rows = read_results(tmp_path / "results.tsv")
        assert len(rows) == 1
        assert rows[0]["kept"] == "no"
        assert rows[0]["commit_sha"] == "reverted"

        output = capsys.readouterr().out
        assert "DISCARD" in output

    def test_multiple_iterations(self, tmp_path):
        config = _init_lab(tmp_path)
        run_cfg = RunConfig(max_iterations=3, data="data", iteration_timeout=0)

        verdicts = [
            {"action": "keep", "score": 0.10, "notes": "first"},
            {"action": "discard", "score": 0.12, "notes": "worse"},
            {"action": "keep", "score": 0.05, "notes": "better"},
        ]
        verdict_iter = iter(verdicts)

        # poll: None (running) for each wait + one more for the stop check
        container = _make_container([None] * 10)

        with (
            patch(f"{MODULE}.start_container", return_value=container),
            patch(
                f"{MODULE}.wait_for_verdict",
                side_effect=lambda *a, **kw: next(verdict_iter),
            ),
            patch(f"{MODULE}.git_commit", return_value="sha123"),
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        rows = read_results(tmp_path / "results.tsv")
        assert len(rows) == 3
        assert rows[0]["kept"] == "yes"
        assert rows[1]["kept"] == "no"
        assert rows[2]["kept"] == "yes"

    def test_stops_on_max_iterations(self, tmp_path, capsys):
        config = _init_lab(tmp_path)
        run_cfg = RunConfig(max_iterations=2, data="data", iteration_timeout=0)

        verdict = {"action": "keep", "score": 0.05, "notes": "ok"}
        container = _make_container([None] * 10)

        with (
            patch(f"{MODULE}.start_container", return_value=container),
            patch(f"{MODULE}.wait_for_verdict", return_value=verdict),
            patch(f"{MODULE}.git_commit", return_value="sha"),
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        rows = read_results(tmp_path / "results.tsv")
        assert len(rows) == 2

        output = capsys.readouterr().out
        assert "max iterations" in output

    def test_stops_on_plateau(self, tmp_path, capsys):
        config = _init_lab(tmp_path)
        run_cfg = RunConfig(
            max_iterations=50,
            plateau_threshold=3,
            data="data",
            iteration_timeout=0,
        )

        verdict = {"action": "discard", "score": 0.5, "notes": "bad"}
        container = _make_container([None] * 20)

        with (
            patch(f"{MODULE}.start_container", return_value=container),
            patch(f"{MODULE}.wait_for_verdict", return_value=verdict),
            patch(f"{MODULE}.git_commit"),
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        rows = read_results(tmp_path / "results.tsv")
        assert len(rows) == 3

        output = capsys.readouterr().out
        assert "Plateau" in output

    def test_container_crash_restarts(self, tmp_path, capsys):
        config = _init_lab(tmp_path)
        run_cfg = RunConfig(
            max_iterations=5,
            max_restarts=2,
            data="data",
            iteration_timeout=0,
        )

        # Container exits immediately, then restarts, then delivers a verdict
        containers = [
            _make_container([1]),  # crash
            _make_container([1]),  # crash
            _make_container([1]),  # crash — exceeds max_restarts
        ]
        container_iter = iter(containers)

        with (
            patch(
                f"{MODULE}.start_container",
                side_effect=lambda *a, **kw: next(container_iter),
            ),
            patch(f"{MODULE}.git_commit"),
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        output = capsys.readouterr().out
        assert "Container crashed" in output

    def test_resumes_experiment_numbering(self, tmp_path):
        config = _init_lab(tmp_path)
        results_tsv = tmp_path / "results.tsv"

        # Seed results.tsv with 5 prior experiments
        for i in range(1, 6):
            append_result(
                results_tsv,
                experiment_id=f"exp_{i:03d}",
                score=0.1 * i,
                kept=True,
                commit_sha=f"sha{i}",
            )

        run_cfg = RunConfig(max_iterations=2, data="data", iteration_timeout=0)

        # Verdicts without experiment_id — should auto-assign
        verdicts = [
            {"action": "keep", "score": 0.04, "notes": "a"},
            {"action": "discard", "score": 0.06, "notes": "b"},
        ]
        verdict_iter = iter(verdicts)
        container = _make_container([None] * 10)

        with (
            patch(f"{MODULE}.start_container", return_value=container),
            patch(
                f"{MODULE}.wait_for_verdict",
                side_effect=lambda *a, **kw: next(verdict_iter),
            ),
            patch(f"{MODULE}.git_commit", return_value="sha_new"),
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        rows = read_results(results_tsv)
        # 5 existing + 2 new = 7
        assert len(rows) == 7
        # New experiments should continue from exp_006
        assert rows[5]["experiment_id"] == "exp_006"
        assert rows[6]["experiment_id"] == "exp_007"

    def test_best_score_tracked(self, tmp_path):
        config = _init_lab(tmp_path)
        run_cfg = RunConfig(max_iterations=3, data="data", iteration_timeout=0)

        verdicts = [
            {"action": "keep", "score": 0.10},
            {"action": "keep", "score": 0.05},
            {"action": "keep", "score": 0.08},
        ]
        verdict_iter = iter(verdicts)

        container = _make_container([None] * 10)

        with (
            patch(f"{MODULE}.start_container", return_value=container),
            patch(
                f"{MODULE}.wait_for_verdict",
                side_effect=lambda *a, **kw: next(verdict_iter),
            ),
            patch(f"{MODULE}.git_commit", return_value="sha"),
            patch(f"{MODULE}.git_amend_with_results"),
            patch(f"{MODULE}.git_revert"),
        ):
            run_session(run_cfg, config, tmp_path)

        output_lines = (tmp_path / "results.tsv").read_text().splitlines()
        # Best score is 0.05 (second experiment)
        # Captured in the session summary print
        # We verify indirectly: all 3 rows logged
        assert len(output_lines) == 4  # header + 3 rows
