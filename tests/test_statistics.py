"""
Unit tests for statistical analysis utilities (Wilson CIs, exact McNemar test).
"""

import pytest
from src.statistics import wilson_score_interval, exact_mcnemar_test


def test_wilson_score_interval_properties():
    # 0 successes
    low, high = wilson_score_interval(0, 100)
    assert low == 0.0
    assert 0.0 < high < 10.0

    # 100% successes
    low, high = wilson_score_interval(100, 100)
    assert 90.0 < low < 100.0
    assert high == 100.0

    # 50% successes
    low, high = wilson_score_interval(50, 100)
    assert 40.0 < low < 50.0
    assert 50.0 < high < 60.0
    # Symmetric around 50%
    assert abs((50.0 - low) - (high - 50.0)) < 0.1


def test_exact_mcnemar_test():
    # Zero discordant pairs -> p=1.0
    assert exact_mcnemar_test(0, 0) == 1.0

    # Equal discordant pairs -> p=1.0
    assert exact_mcnemar_test(5, 5) == 1.0

    # Strong asymmetry: 0 vs 10 discordant pairs -> highly significant
    p_asym = exact_mcnemar_test(0, 10)
    assert p_asym < 0.01

    # Marginal difference: 2 vs 4 discordant pairs
    p_marginal = exact_mcnemar_test(2, 4)
    assert p_marginal > 0.05  # Not statistically significant at alpha=0.05
