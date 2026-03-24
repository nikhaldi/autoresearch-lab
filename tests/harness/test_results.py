"""Tests for experiment results logging."""

import json

from autoresearch_lab.harness.results import HEADER, append_result, read_results


class TestAppendResult:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "results.tsv"
        append_result(
            path,
            experiment_id="exp_001",
            score=0.042,
            kept=True,
            commit_sha="abc123",
            notes="first run",
        )

        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert lines[0] == "\t".join(HEADER)

    def test_appends_without_duplicating_header(self, tmp_path):
        path = tmp_path / "results.tsv"
        for i in range(3):
            append_result(
                path,
                experiment_id=f"exp_{i:03d}",
                score=0.1 * i,
                kept=True,
                commit_sha=f"sha{i}",
            )

        lines = path.read_text().splitlines()
        assert len(lines) == 4  # 1 header + 3 rows

    def test_score_is_formatted(self, tmp_path):
        path = tmp_path / "results.tsv"
        append_result(
            path,
            experiment_id="exp_001",
            score=0.042,
            kept=True,
            commit_sha="abc",
        )

        rows = read_results(path)
        assert rows[0]["score"] == "0.042000"

    def test_metrics_stored_as_json(self, tmp_path):
        path = tmp_path / "results.tsv"
        append_result(
            path,
            experiment_id="exp_001",
            score=0.042,
            metrics={"accuracy": 0.95, "latency_ms": 120.0},
            kept=True,
            commit_sha="abc",
        )

        rows = read_results(path)
        metrics = json.loads(rows[0]["metrics"])
        assert metrics["accuracy"] == 0.95
        assert metrics["latency_ms"] == 120.0

    def test_empty_metrics(self, tmp_path):
        path = tmp_path / "results.tsv"
        append_result(
            path,
            experiment_id="exp_001",
            score=0.5,
            kept=False,
            commit_sha="reverted",
        )

        rows = read_results(path)
        assert json.loads(rows[0]["metrics"]) == {}

    def test_kept_values(self, tmp_path):
        path = tmp_path / "results.tsv"
        append_result(path, experiment_id="a", score=0.1, kept=True, commit_sha="x")
        append_result(path, experiment_id="b", score=0.2, kept=False, commit_sha="y")

        rows = read_results(path)
        assert rows[0]["kept"] == "yes"
        assert rows[1]["kept"] == "no"

    def test_notes_preserved(self, tmp_path):
        path = tmp_path / "results.tsv"
        append_result(
            path,
            experiment_id="exp_001",
            score=0.1,
            kept=True,
            commit_sha="abc",
            notes="Added perspective correction",
        )

        rows = read_results(path)
        assert rows[0]["notes"] == "Added perspective correction"


class TestReadResults:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "results.tsv"
        path.touch()
        assert read_results(path) == []

    def test_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.tsv"
        assert read_results(path) == []

    def test_round_trip(self, tmp_path):
        path = tmp_path / "results.tsv"
        for i in range(5):
            append_result(
                path,
                experiment_id=f"exp_{i:03d}",
                score=0.1 - i * 0.01,
                kept=i % 2 == 0,
                commit_sha=f"sha{i}",
                notes=f"run {i}",
            )

        rows = read_results(path)
        assert len(rows) == 5
        assert rows[0]["experiment_id"] == "exp_000"
        assert rows[4]["experiment_id"] == "exp_004"
        assert rows[0]["commit_sha"] == "sha0"
