"""Abstract evaluation backend interface.

Every lab provides a backend that knows how to evaluate the pipeline code
against reference data. The harness dispatches to it without knowing
platform details.

The backend returns a single `score` (lower is better) that the framework
uses for keep/discard decisions and stopping conditions. Additional metrics
can be included for logging but the framework only tracks the score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SampleResult:
    """Result for a single sample."""

    sample_id: str
    score: float
    error: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    """Evaluation result returned by a backend.

    Attributes:
        score: The single metric the framework tracks (lower is better).
            Used for keep/discard decisions, stopping conditions, and
            identifying the best experiment.
        metrics: Optional additional metrics for logging (e.g. accuracy,
            latency, error rate). These are written to results.tsv but
            the framework does not interpret them.
        sample_results: Per-sample results for diagnosis.
    """

    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    sample_results: list[SampleResult] = field(default_factory=list)


class EvalBackend(ABC):
    """Abstract interface for evaluation backends."""

    @abstractmethod
    def evaluate(
        self,
        pipeline_dir: Path,
        data_dir: Path,
        sample_ids: list[str] | None = None,
    ) -> EvalResult:
        """Run the pipeline and evaluate against data.

        Args:
            pipeline_dir: Path to the pipeline code the agent wrote.
            data_dir: Path to the read-only data directory.
            sample_ids: Optional list of sample IDs to evaluate. When
                provided, the backend may restrict evaluation to only
                these samples, which can significantly speed up
                diagnosis during the research loop. Implementing this
                filtering is optional — if not implemented, the full
                evaluation runs and the framework filters the results.

        Returns:
            EvalResult with a score and optional per-sample details.
        """
        ...

    def setup(self) -> None:
        """Optional one-time setup (start emulator, build container, etc.)."""

    def teardown(self) -> None:
        """Optional cleanup after evaluation."""
