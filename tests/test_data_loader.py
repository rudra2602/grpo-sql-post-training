"""
Unit tests for data_loader module.

Verifies:
1. format_sql_prompt generates exact prompt template with system instructions.
2. extract_schema_ddl extracts valid CREATE TABLE statements.
3. load_logistics_dataset correctly loads and formats train (406) and test (108) records.
4. prepare_grpo_dataset outputs dict format matching TRL GRPOTrainer requirements.
"""

from pathlib import Path
import pytest

from src.data_loader import (
    format_sql_prompt,
    extract_schema_ddl,
    load_logistics_dataset,
    prepare_grpo_dataset,
    SYSTEM_INSTRUCTION,
)


def test_format_sql_prompt():
    """Verify prompt formatting matches protocol specification."""
    ddl = "CREATE TABLE vessels (id INT PRIMARY KEY, name TEXT);"
    question = "List all vessel names."
    prompt = format_sql_prompt(ddl, question)

    assert SYSTEM_INSTRUCTION in prompt
    assert "### Schema:\n" + ddl in prompt
    assert "### Question:\n" + question in prompt
    assert "```sql ... ```" in prompt


def test_extract_schema_ddl():
    """Verify schema extraction from existing downloaded database."""
    db_path = Path("data/databases/ship_1/ship_1.sqlite")
    if not db_path.exists():
        pytest.skip("ship_1.sqlite not found")

    ddl = extract_schema_ddl(db_path)
    assert "CREATE TABLE" in ddl
    assert "captain" in ddl.lower()
    assert "ship" in ddl.lower()
    assert "PRAGMA" not in ddl


def test_load_train_dataset():
    """Verify train dataset loading (406 pairs across 8 schemas)."""
    train_records = load_logistics_dataset(split="train")
    assert len(train_records) == 406

    first = train_records[0]
    required_keys = {"prompt", "gold_sql", "db_path", "db_id", "question", "difficulty", "schema_ddl"}
    assert required_keys.issubset(first.keys())
    assert Path(first["db_path"]).exists()
    assert first["schema_ddl"] in first["prompt"]
    assert first["question"] in first["prompt"]


def test_load_test_dataset():
    """Verify test dataset loading (108 pairs across 2 schemas)."""
    test_records = load_logistics_dataset(split="test")
    assert len(test_records) == 108

    first = test_records[0]
    required_keys = {"prompt", "gold_sql", "db_path", "db_id", "question", "difficulty", "schema_ddl"}
    assert required_keys.issubset(first.keys())
    assert Path(first["db_path"]).exists()


def test_prepare_grpo_dataset():
    """Verify format matches Hugging Face Dataset.from_dict requirements."""
    grpo_dict = prepare_grpo_dataset(split="train")
    assert set(grpo_dict.keys()) == {"prompt", "gold_sql", "db_path"}
    assert len(grpo_dict["prompt"]) == 406
    assert len(grpo_dict["gold_sql"]) == 406
    assert len(grpo_dict["db_path"]) == 406
