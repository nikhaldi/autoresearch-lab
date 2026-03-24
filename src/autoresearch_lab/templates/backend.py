"""Evaluation backend for this lab."""

from pathlib import Path

from autoresearch_lab.harness.backend import EvalBackend, EvalResult, SampleResult


class Backend(EvalBackend):
    def evaluate(self, pipeline_dir: Path, data_dir: Path) -> EvalResult:
        raise NotImplementedError("Implement evaluation logic here")
        # Example:
        # return EvalResult(
        #     score=0.042,                   # single number, lower is better
        #     metrics={"latency_ms": 150.0}, # optional, logged to results.tsv
        #     sample_results=[               # optional, used by `arl diagnose`
        #         SampleResult(sample_id="sample_1", score=0.03),
        #         SampleResult(sample_id="sample_2", score=0.05),
        #     ],
        # )
