"""
Unit tests for train_grpo module configuration and reward wrapper.
"""

from pathlib import Path
import numpy as np
import pytest

from src.reward import grpo_sql_reward_func


def test_grpo_reward_wrapper_zero_variance_logic(temp_logistics_db: Path):
    """
    Verifies the zero-variance tracking logic used inside GRPOTrainer reward callback.
    """
    # 4 completions where all 4 score identical rewards -> std is 0.0
    identical_completions = [
        "```sql\nSELECT * FROM containers;\n```",
        "```sql\nSELECT * FROM containers;\n```",
        "```sql\nSELECT * FROM containers;\n```",
        "```sql\nSELECT * FROM containers;\n```",
    ]
    gold_sql = "SELECT container_id FROM containers;"
    db_paths = [temp_logistics_db] * 4

    rewards = grpo_sql_reward_func(identical_completions, [gold_sql] * 4, db_paths)
    assert len(rewards) == 4
    assert np.std(rewards) == 0.0, "Expected zero standard deviation across identical completions"

    # 4 completions where 1 matches and 3 fail -> non-zero variance
    diverse_completions = [
        "```sql\nSELECT container_id FROM containers;\n```",
        "```sql\nSELECT FROM WHERE;\n```",
        "```sql\nSELECT tracking_number FROM containers;\n```",
        "```sql\nSELECT weight_kg FROM containers;\n```",
    ]
    diverse_rewards = grpo_sql_reward_func(diverse_completions, [gold_sql] * 4, db_paths)
    assert len(diverse_rewards) == 4
    assert np.std(diverse_rewards) > 0.0, "Expected non-zero standard deviation across diverse completions"
