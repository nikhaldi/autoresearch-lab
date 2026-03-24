"""Tests for orchestrator stop conditions."""

import time

from autoresearch_lab.sandbox.orchestrator import (
    RunConfig,
    SessionState,
    check_stop_conditions,
)


class TestCheckStopConditions:
    def test_no_stop_by_default(self):
        config = RunConfig()
        state = SessionState(start_time=time.time())
        assert check_stop_conditions(config, state) is None

    def test_max_iterations(self):
        config = RunConfig(max_iterations=5)
        state = SessionState(start_time=time.time(), iteration=5)
        result = check_stop_conditions(config, state)
        assert result is not None
        assert "max iterations" in result

    def test_below_max_iterations(self):
        config = RunConfig(max_iterations=5)
        state = SessionState(start_time=time.time(), iteration=4)
        assert check_stop_conditions(config, state) is None

    def test_max_hours(self):
        config = RunConfig(max_hours=1.0)
        state = SessionState(start_time=time.time() - 3601)
        result = check_stop_conditions(config, state)
        assert result is not None
        assert "max time" in result

    def test_below_max_hours(self):
        config = RunConfig(max_hours=1.0)
        state = SessionState(start_time=time.time() - 3599)
        assert check_stop_conditions(config, state) is None

    def test_plateau(self):
        config = RunConfig(plateau_threshold=3)
        state = SessionState(
            start_time=time.time(), consecutive_discards=3
        )
        result = check_stop_conditions(config, state)
        assert result is not None
        assert "Plateau" in result

    def test_below_plateau(self):
        config = RunConfig(plateau_threshold=3)
        state = SessionState(
            start_time=time.time(), consecutive_discards=2
        )
        assert check_stop_conditions(config, state) is None

    def test_target_score_reached(self):
        config = RunConfig(target_score=0.05)
        state = SessionState(start_time=time.time(), best_score=0.03)
        result = check_stop_conditions(config, state)
        assert result is not None
        assert "target score" in result

    def test_target_score_not_reached(self):
        config = RunConfig(target_score=0.05)
        state = SessionState(start_time=time.time(), best_score=0.10)
        assert check_stop_conditions(config, state) is None

    def test_target_score_disabled(self):
        config = RunConfig(target_score=0.0)
        state = SessionState(start_time=time.time(), best_score=0.001)
        assert check_stop_conditions(config, state) is None

    def test_max_cost(self):
        config = RunConfig(max_cost=10.0)
        state = SessionState(start_time=time.time())
        result = check_stop_conditions(config, state, cost_usd=10.50)
        assert result is not None
        assert "cost limit" in result

    def test_below_max_cost(self):
        config = RunConfig(max_cost=10.0)
        state = SessionState(start_time=time.time())
        assert check_stop_conditions(config, state, cost_usd=9.99) is None

    def test_max_cost_disabled(self):
        config = RunConfig(max_cost=0.0)
        state = SessionState(start_time=time.time())
        assert check_stop_conditions(config, state, cost_usd=999.0) is None
