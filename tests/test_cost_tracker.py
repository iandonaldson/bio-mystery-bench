"""Tests for harness/cost_tracker.py."""

import pytest

from harness.cost_tracker import CostTracker


class TestCostTrackerAdd:
    def test_starts_at_zero(self):
        ct = CostTracker()
        assert ct.total_input_tokens == 0
        assert ct.total_output_tokens == 0
        assert ct.total_cache_read_tokens == 0

    def test_add_accumulates_tokens(self):
        ct = CostTracker()
        ct.add(1000, 200)
        ct.add(500, 100)
        assert ct.total_input_tokens == 1500
        assert ct.total_output_tokens == 300

    def test_add_with_cache_hits(self):
        ct = CostTracker()
        ct.add(1000, 200, cache_read_tokens=800)
        assert ct.total_cache_read_tokens == 800


class TestCostTrackerTotalCost:
    def test_zero_tokens_zero_cost(self):
        ct = CostTracker()
        assert ct.total_cost_usd == 0.0

    def test_input_only_cost(self):
        # 1M input tokens at $3/M = $3.00
        ct = CostTracker(cost_per_million_input=3.0, cost_per_million_output=15.0)
        ct.add(1_000_000, 0)
        assert ct.total_cost_usd == pytest.approx(3.0)

    def test_output_only_cost(self):
        # 1M output tokens at $15/M = $15.00
        ct = CostTracker(cost_per_million_input=3.0, cost_per_million_output=15.0)
        ct.add(0, 1_000_000)
        assert ct.total_cost_usd == pytest.approx(15.0)

    def test_cache_reads_billed_at_10_percent(self):
        # The API reports input_tokens (new) and cache_read_input_tokens separately.
        # 1M cache-read tokens with 0 new input: billed at 10% of $3/M = $0.30
        ct = CostTracker(cost_per_million_input=3.0, cost_per_million_output=15.0)
        ct.add(0, 0, cache_read_tokens=1_000_000)
        # input_cost = 0, cache_cost = 1M * 3.0 * 0.1 / 1M = $0.30
        assert ct.total_cost_usd == pytest.approx(0.30)

    def test_mixed_tokens(self):
        # 500k new input tokens + 500k cache-read tokens + 100k output tokens
        # (input and cache_read are disjoint — both billed separately)
        ct = CostTracker(cost_per_million_input=3.0, cost_per_million_output=15.0)
        ct.add(500_000, 100_000, cache_read_tokens=500_000)
        # input_cost  = 500k * 3.0 / 1M = $1.50
        # cache_cost  = 500k * 0.30 / 1M = $0.15
        # output_cost = 100k * 15.0 / 1M = $1.50
        expected = 1.50 + 0.15 + 1.50
        assert ct.total_cost_usd == pytest.approx(expected)


class TestCostTrackerCheckLimit:
    def test_under_limit_does_not_raise(self):
        ct = CostTracker(max_session_cost_usd=10.0)
        ct.add(100, 100)  # tiny cost
        ct.check_limit()  # should not raise

    def test_at_limit_raises(self):
        ct = CostTracker(
            cost_per_million_input=3.0,
            cost_per_million_output=15.0,
            max_session_cost_usd=0.001,
        )
        ct.add(1_000_000, 0)  # cost = $3.00, well over limit
        with pytest.raises(RuntimeError, match="Session cost limit reached"):
            ct.check_limit()


class TestCostTrackerEstimate:
    def test_estimate_returns_positive_float(self):
        ct = CostTracker()
        est = ct.estimate(n_problems=5, n_attempts=5)
        assert isinstance(est, float)
        assert est > 0

    def test_more_problems_costs_more(self):
        ct = CostTracker()
        est_small = ct.estimate(n_problems=5, n_attempts=1)
        est_large = ct.estimate(n_problems=99, n_attempts=1)
        assert est_large > est_small

    def test_more_attempts_costs_less_than_linear_due_to_caching(self):
        ct = CostTracker()
        est_1 = ct.estimate(n_problems=1, n_attempts=1)
        est_5 = ct.estimate(n_problems=1, n_attempts=5)
        # With caching, 5 attempts should cost less than 5× the single attempt
        assert est_5 < est_1 * 5


class TestCostTrackerSummary:
    def test_summary_contains_token_counts(self):
        ct = CostTracker()
        ct.add(1000, 200, cache_read_tokens=500)
        summary = ct.summary()
        assert "1,000" in summary
        assert "200" in summary
        assert "500" in summary

    def test_summary_contains_cost(self):
        ct = CostTracker()
        ct.add(0, 0)
        assert "$" in ct.summary()
