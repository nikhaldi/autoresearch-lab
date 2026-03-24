"""Experiment results logging to TSV."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

HEADER = [
    "experiment_id",
    "timestamp",
    "score",
    "metrics",
    "kept",
    "commit_sha",
    "notes",
]


def append_result(
    results_path: Path,
    *,
    experiment_id: str,
    score: float,
    metrics: dict[str, float] | None = None,
    kept: bool,
    commit_sha: str,
    notes: str = "",
) -> None:
    """Append a single experiment result to the TSV log."""
    write_header = not results_path.exists() or results_path.stat().st_size == 0

    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(HEADER)
        writer.writerow(
            [
                experiment_id,
                datetime.datetime.now(datetime.UTC).isoformat(),
                f"{score:.6f}",
                json.dumps(metrics) if metrics else "{}",
                "yes" if kept else "no",
                commit_sha,
                notes,
            ]
        )


def read_results(results_path: Path) -> list[dict[str, str]]:
    """Read all results from the TSV log."""
    if not results_path.exists():
        return []
    with open(results_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)
