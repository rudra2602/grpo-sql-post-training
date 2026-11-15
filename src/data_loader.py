"""
Data loader and prompt formatting module for GRPO text-to-SQL training and evaluation.

Provides standard formatting for SQLite DDL schema prompts, dataset loading from
curated logistics splits, and conversion to Hugging Face Dataset format.
"""

import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Union

SYSTEM_INSTRUCTION = (
    "You are an expert SQL assistant. Given the following SQLite database schema, "
    "write a valid SQL query that answers the user question.\n"
    "Return ONLY the SQL query enclosed in ```sql ... ``` markdown tags."
)


def format_sql_prompt(schema_ddl: str, question: str) -> str:
    """
    Format system instructions, database DDL schema, and question into a prompt.

    Args:
        schema_ddl: The SQLite CREATE TABLE statements representing the database schema.
        question: The natural language user question.

    Returns:
        Formatted prompt string.
    """
    cleaned_ddl = schema_ddl.strip()
    cleaned_question = question.strip()
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"### Schema:\n{cleaned_ddl}\n\n"
        f"### Question:\n{cleaned_question}\n"
    )


def extract_schema_ddl(db_source: Union[str, Path, sqlite3.Connection]) -> str:
    """
    Extract clean CREATE TABLE statements for a SQLite database.

    Supports reading companion schema.sql files or dynamically querying sqlite_master.
    """
    # 1. If db_source is a file path, check if companion schema.sql exists
    if not isinstance(db_source, sqlite3.Connection):
        path_obj = Path(db_source)
        if path_obj.is_file():
            schema_file = path_obj.parent / "schema.sql"
            if schema_file.exists():
                text = schema_file.read_text(encoding="utf-8")
                # Remove PRAGMA statements
                cleaned = re.sub(r"PRAGMA\s+[^;]+;\s*", "", text, flags=re.IGNORECASE).strip()
                if cleaned:
                    return cleaned

    # 2. Dynamic query from SQLite connection
    conn: Optional[sqlite3.Connection] = None
    should_close = False
    try:
        if isinstance(db_source, sqlite3.Connection):
            conn = db_source
        elif str(db_source) == ":memory:":
            conn = sqlite3.connect(":memory:")
            should_close = True
        else:
            path_obj = Path(db_source)
            file_uri = f"{path_obj.resolve().as_uri()}?mode=ro"
            conn = sqlite3.connect(file_uri, uri=True)
            should_close = True

        cur = conn.cursor()
        cur.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL;"
        )
        statements = [row[0].strip() for row in cur.fetchall() if row[0]]
        return "\n\n".join(statements)

    finally:
        if should_close and conn is not None:
            conn.close()


def load_logistics_dataset(
    split: str = "train",
    data_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Load curated logistics dataset split (train or test), format prompts, and attach schema DDL.

    Args:
        split: Either 'train' or 'test'.
        data_dir: Directory containing train_logistics.json, test_logistics.json, and databases/.
                  Defaults to project root 'data/'.

    Returns:
        List of dataset dicts containing prompt, gold_sql, db_path, db_id, question, difficulty, schema_ddl.
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        base_dir = Path(data_dir)

    json_file = base_dir / f"{split}_logistics.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Dataset split file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    databases_dir = base_dir / "databases"
    schema_cache: Dict[str, str] = {}
    formatted_dataset: List[Dict[str, Any]] = []

    for item in records:
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["gold_sql"]
        difficulty = item.get("difficulty", "medium")

        # Resolve database file path
        db_path = databases_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        # Fetch or cache schema DDL
        if db_id not in schema_cache:
            schema_cache[db_id] = extract_schema_ddl(db_path)

        schema_ddl = schema_cache[db_id]
        prompt = format_sql_prompt(schema_ddl, question)

        formatted_dataset.append({
            "prompt": prompt,
            "gold_sql": gold_sql,
            "db_path": str(db_path.resolve()),
            "db_id": db_id,
            "question": question,
            "difficulty": difficulty,
            "schema_ddl": schema_ddl,
        })

    return formatted_dataset


def prepare_grpo_dataset(
    split: str = "train",
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, List[Any]]:
    """
    Prepares a dictionary format suitable for Hugging Face `Dataset.from_dict(...)`.

    Returns dictionary with keys: 'prompt', 'gold_sql', 'db_path'.
    """
    records = load_logistics_dataset(split=split, data_dir=data_dir)
    return {
        "prompt": [r["prompt"] for r in records],
        "gold_sql": [r["gold_sql"] for r in records],
        "db_path": [r["db_path"] for r in records],
    }
