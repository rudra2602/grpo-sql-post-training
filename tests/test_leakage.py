"""
Strict zero-leakage verification suite.

Enforces:
1. Zero database overlap between train and test splits (set intersection is empty).
2. Exact expected counts: 406 train pairs across 8 schemas, 108 test pairs across 2 held-out schemas.
3. Immutability of test databases (tracking_orders, ship_1 must never appear in training).
4. Physical existence and non-empty execution of databases and sample queries.
"""

import json
from pathlib import Path
import pytest

from src.executor import SafeSQLExecutor

EXPECTED_TRAIN_DBS = {
    "customers_and_invoices",
    "department_store",
    "customers_and_addresses",
    "product_catalog",
    "store_product",
    "ship_mission",
    "products_for_hire",
    "customer_deliveries",
}

EXPECTED_TEST_DBS = {
    "tracking_orders",
    "ship_1",
}


def test_train_test_db_ids_strictly_disjoint(project_data_dir: Path):
    """
    CRITICAL CHECK: Asserts that train and test database IDs share zero overlap.
    """
    train_file = project_data_dir / "train_logistics.json"
    test_file = project_data_dir / "test_logistics.json"

    assert train_file.exists(), f"Missing train file: {train_file}"
    assert test_file.exists(), f"Missing test file: {test_file}"

    with open(train_file, "r", encoding="utf-8") as f:
        train_records = json.load(f)

    with open(test_file, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    train_db_ids = {r["db_id"] for r in train_records}
    test_db_ids = {r["db_id"] for r in test_records}

    # Verify sets are disjoint
    overlap = train_db_ids.intersection(test_db_ids)
    assert len(overlap) == 0, f"FATAL DATA LEAKAGE DETECTED! Overlapping db_ids: {overlap}"
    assert train_db_ids.isdisjoint(test_db_ids), "Train and test db_ids must be completely disjoint."


def test_train_split_exact_schemas_and_counts(project_data_dir: Path):
    """
    Verifies that the train split matches the 8 designated logistics schemas
    and totals exactly 406 pairs.
    """
    train_file = project_data_dir / "train_logistics.json"
    with open(train_file, "r", encoding="utf-8") as f:
        train_records = json.load(f)

    train_db_ids = {r["db_id"] for r in train_records}
    assert train_db_ids == EXPECTED_TRAIN_DBS, f"Mismatch in train schemas: {train_db_ids ^ EXPECTED_TRAIN_DBS}"
    assert len(train_records) == 406, f"Expected 406 train pairs, got {len(train_records)}"

    # Ensure none of the test db_ids appear in train
    for r in train_records:
        assert r["db_id"] not in EXPECTED_TEST_DBS, f"Test db {r['db_id']} leaked into train set!"


def test_test_split_exact_schemas_and_counts(project_data_dir: Path):
    """
    Verifies that the test split matches the 2 held-out logistics schemas
    and totals exactly 108 pairs.
    """
    test_file = project_data_dir / "test_logistics.json"
    with open(test_file, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    test_db_ids = {r["db_id"] for r in test_records}
    assert test_db_ids == EXPECTED_TEST_DBS, f"Mismatch in test schemas: {test_db_ids ^ EXPECTED_TEST_DBS}"
    assert len(test_records) == 108, f"Expected 108 test pairs, got {len(test_records)}"

    # Ensure none of the train db_ids appear in test
    for r in test_records:
        assert r["db_id"] not in EXPECTED_TRAIN_DBS, f"Train db {r['db_id']} leaked into test set!"


def test_database_files_exist_and_accessible(project_data_dir: Path):
    """
    Verifies all 10 databases exist in data/databases/<db_id>/<db_id>.sqlite.
    """
    all_dbs = EXPECTED_TRAIN_DBS | EXPECTED_TEST_DBS
    assert len(all_dbs) == 10

    for db_id in all_dbs:
        db_file = project_data_dir / "databases" / db_id / f"{db_id}.sqlite"
        schema_file = project_data_dir / "databases" / db_id / "schema.sql"

        assert db_file.exists(), f"Database file missing: {db_file}"
        assert schema_file.exists(), f"Schema file missing: {schema_file}"
        assert db_file.stat().st_size > 0, f"Database file is empty: {db_file}"


def test_sample_gold_queries_execute_with_data(project_data_dir: Path):
    """
    Executes a sample of gold queries from both train and test to guarantee
    they execute cleanly and return non-empty row sets.
    """
    for split_file in ["train_logistics.json", "test_logistics.json"]:
        path = project_data_dir / split_file
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        # Check first 5 records of each split
        for record in records[:5]:
            db_id = record["db_id"]
            gold_sql = record["gold_sql"]
            db_path = project_data_dir / "databases" / db_id / f"{db_id}.sqlite"

            success, rows, err = SafeSQLExecutor.execute(db_path, gold_sql)
            assert success is True, f"Gold SQL failed on {db_id}: {gold_sql} -> {err}"
            assert len(rows) >= 1, f"Gold query returned 0 rows on {db_id}: {gold_sql}"
