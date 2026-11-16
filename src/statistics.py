"""
Statistical analysis utilities for model evaluation and paired hypothesis testing.

Provides:
- Wilson score 95% confidence intervals for binomial proportions.
- Exact McNemar's test for paired binary classification outcomes.
- Comparative metric aggregations.
"""

import math
from typing import Dict, List, Tuple, Any, Optional


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of positive outcomes (k).
        total: Total number of trials (n).
        confidence: Confidence level (default: 0.95, z ~ 1.95996).

    Returns:
        Tuple of (lower_bound_pct, upper_bound_pct) in percentage points [0.0, 100.0].
    """
    if total <= 0:
        return 0.0, 0.0

    # Two-sided standard normal quantiles
    z_map = {
        0.90: 1.64485,
        0.95: 1.95996,
        0.99: 2.57583,
    }
    z = z_map.get(confidence, 1.95996)

    p_hat = successes / total
    z2 = z * z
    n = total

    denominator = 1.0 + z2 / n
    center_adj = p_hat + z2 / (2.0 * n)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    lower = max(0.0, (center_adj - spread) / denominator)
    upper = min(1.0, (center_adj + spread) / denominator)

    return round(lower * 100.0, 2), round(upper * 100.0, 2)


def exact_mcnemar_test(b: int, c: int) -> float:
    """
    Compute the exact two-sided p-value for McNemar's test on discordant pairs.

    Null Hypothesis (H0): Both models have the same marginal error rate (P(b) == P(c)).
    Under H0, the discordant pairs b and c follow Binomial(n = b + c, p = 0.5).

    Args:
        b: Count where Model A is correct and Model B is incorrect.
        c: Count where Model A is incorrect and Model B is correct.

    Returns:
        Two-sided exact p-value in [0.0, 1.0].
    """
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0

    k = min(b, c)
    # Exact cumulative two-sided binomial probability
    # p = 2 * sum(comb(n, i) * 0.5^n for i in 0..k)
    prob_tail = sum(math.comb(n_discordant, i) for i in range(k + 1)) * (0.5 ** n_discordant)
    p_value = min(1.0, 2.0 * prob_tail)
    return round(p_value, 4)
