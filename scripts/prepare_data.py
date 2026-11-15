"""
Data preparation script for GRPO text-to-SQL post-training.

Downloads 10 verified Spider SQLite databases from Hugging Face (Chinastark/spider_datasets)
and query pairs, filters for valid queries returning >= 1 rows, and produces:
  - data/train_logistics.json (406 pairs across 8 schemas)
  - data/test_logistics.json (108 pairs across 2 held-out schemas)
"""

import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Tuple
import urllib.request

TRAIN_DB_IDS = [
    "customers_and_invoices",
    "department_store",
    "customers_and_addresses",
    "product_catalog",
    "store_product",
    "ship_mission",
    "products_for_hire",
    "customer_deliveries",
]

TEST_DB_IDS = [
    "tracking_orders",
    "ship_1",
]

ALL_DB_IDS = TRAIN_DB_IDS + TEST_DB_IDS

BASE_HF_URL = "https://huggingface.co/datasets/Chinastark/spider_datasets/resolve/main"
QUERIES_URL = f"{BASE_HF_URL}/train_spider.json"


def download_file(url: str, dest_path: Path, force: bool = False) -> None:
  """Download a file with User-Agent header if it does not already exist."""
  dest_path.parent.mkdir(parents=True, exist_ok=True)
  if dest_path.exists() and not force:
    return
  print(f"Downloading {url} -> {dest_path}...")
  req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
  with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
    f.write(resp.read())


def download_databases(data_dir: Path) -> None:
  """Download SQLite db and schema.sql for all 10 target databases."""
  db_base_dir = data_dir / "databases"
  for db_id in ALL_DB_IDS:
    db_folder = db_base_dir / db_id
    sqlite_url = f"{BASE_HF_URL}/database/{db_id}/{db_id}.sqlite"
    schema_url = f"{BASE_HF_URL}/database/{db_id}/schema.sql"

    sqlite_path = db_folder / f"{db_id}.sqlite"
    schema_path = db_folder / "schema.sql"

    download_file(sqlite_url, sqlite_path)
    download_file(schema_url, schema_path)


def compute_difficulty(sql_ast: Dict[str, Any], query_str: str) -> str:
  """Classify SQL difficulty into 'easy', 'medium', 'hard', 'extra' following Spider criteria.

  Evaluates component count (WHERE, GROUP BY, ORDER BY, LIMIT, HAVING),
  subqueries, JOINs, and set operations.
  """
  comp1 = 0
  if sql_ast.get("where") and len(sql_ast["where"]) > 0:
    comp1 += 1
  if sql_ast.get("groupBy") and len(sql_ast["groupBy"]) > 0:
    comp1 += 1
  if sql_ast.get("orderBy") and len(sql_ast["orderBy"]) > 0:
    comp1 += 1
  if sql_ast.get("limit") is not None:
    comp1 += 1
  if sql_ast.get("having") and len(sql_ast["having"]) > 0:
    comp1 += 1

  has_set_op = any([
      sql_ast.get("intersect"),
      sql_ast.get("union"),
      sql_ast.get("except"),
  ])

  from_units = sql_ast.get("from", {}).get("table_units", [])
  num_tables = len(from_units)

  has_subquery = False
  for unit in from_units:
    if isinstance(unit, list) and len(unit) > 0 and unit[0] == "sql":
      has_subquery = True
  where_conds = sql_ast.get("where", [])
  for cond in where_conds:
    if isinstance(cond, list):
      for item in cond:
        if isinstance(item, dict):
          has_subquery = True

  score = comp1 + (1 if num_tables > 1 else 0) + (1 if num_tables > 2 else 0)
  if has_set_op or (has_subquery and score >= 2):
    return "extra"
  elif has_subquery or score >= 3:
    return "hard"
  elif score >= 2:
    return "medium"
  else:
    return "easy"


def test_query_execution(db_path: Path, sql: str) -> Tuple[bool, int]:
  """Execute SQL query against SQLite database.

  Returns (success, row_count).
  """
  try:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return True, len(rows)
  except Exception as e:
    return False, 0


def process_dataset(
    data_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
  """Download, filter, and construct train and test logistics datasets."""
  download_databases(data_dir)

  raw_queries_path = data_dir / "raw_train_spider.json"
  download_file(QUERIES_URL, raw_queries_path)

  with open(raw_queries_path, "r", encoding="utf-8") as f:
    all_queries = json.load(f)

  train_records: List[Dict[str, Any]] = []
  test_records: List[Dict[str, Any]] = []

  train_stats: Dict[str, Dict[str, int]] = {
      db: {"total": 0, "valid": 0, "zero_row": 0, "error": 0}
      for db in TRAIN_DB_IDS
  }
  test_stats: Dict[str, Dict[str, int]] = {
      db: {"total": 0, "valid": 0, "zero_row": 0, "error": 0}
      for db in TEST_DB_IDS
  }

  for item in all_queries:
    db_id = item["db_id"]
    if db_id not in ALL_DB_IDS:
      continue

    question = item["question"]
    gold_sql = item["query"]
    sql_ast = item.get("sql", {})

    db_path = data_dir / "databases" / db_id / f"{db_id}.sqlite"
    success, row_count = test_query_execution(db_path, gold_sql)

    stats = train_stats[db_id] if db_id in TRAIN_DB_IDS else test_stats[db_id]
    stats["total"] += 1

    if not success:
      stats["error"] += 1
      print(f"Skipping error query in {db_id}: {gold_sql}")
      continue

    if row_count < 1:
      stats["zero_row"] += 1
      print(f"Skipping zero-row query in {db_id}: {gold_sql}")
      continue

    stats["valid"] += 1
    difficulty = compute_difficulty(sql_ast, gold_sql)

    record = {
        "db_id": db_id,
        "question": question,
        "gold_sql": gold_sql,
        "difficulty": difficulty,
    }

    if db_id in TRAIN_DB_IDS:
      train_records.append(record)
    else:
      test_records.append(record)

  return train_records, test_records, train_stats, test_stats


def main() -> None:
  root_dir = Path(__file__).resolve().parent.parent
  data_dir = root_dir / "data"
  data_dir.mkdir(parents=True, exist_ok=True)

  print("=" * 60)
  print("STARTING DATA PREPARATION FOR LOGISTICS SQL SUITE")
  print("=" * 60)

  train_records, test_records, train_stats, test_stats = process_dataset(
      data_dir
  )

  # Save output files
  train_out = data_dir / "train_logistics.json"
  test_out = data_dir / "test_logistics.json"

  with open(train_out, "w", encoding="utf-8") as f:
    json.dump(train_records, f, indent=2)

  with open(test_out, "w", encoding="utf-8") as f:
    json.dump(test_records, f, indent=2)

  print("\n" + "=" * 60)
  print("DATA PREPARATION SUMMARY")
  print("=" * 60)

  print("\nTRAIN SPLIT (8 Schemas):")
  print(f"{'Database ID':<30} {'Valid':<8} {'Zero-Row':<10} {'Total':<8}")
  print("-" * 58)
  for db_id in TRAIN_DB_IDS:
    s = train_stats[db_id]
    print(
        f"{db_id:<30} {s['valid']:<8} {s['zero_row']:<10} {s['total']:<8}"
    )
  print("-" * 58)
  print(f"{'TOTAL TRAIN PAIRS':<30} {len(train_records):<8}")

  print("\nTEST SPLIT (2 Held-Out Schemas):")
  print(f"{'Database ID':<30} {'Valid':<8} {'Zero-Row':<10} {'Total':<8}")
  print("-" * 58)
  for db_id in TEST_DB_IDS:
    s = test_stats[db_id]
    print(
        f"{db_id:<30} {s['valid']:<8} {s['zero_row']:<10} {s['total']:<8}"
    )
  print("-" * 58)
  print(f"{'TOTAL TEST PAIRS':<30} {len(test_records):<8}")

  train_dbs = {r["db_id"] for r in train_records}
  test_dbs = {r["db_id"] for r in test_records}

  print("\n" + "=" * 60)
  print("LEAKAGE & DISJOINT VERIFICATION")
  print("=" * 60)
  print(f"Train db_ids ({len(train_dbs)}): {sorted(list(train_dbs))}")
  print(f"Test db_ids ({len(test_dbs)}): {sorted(list(test_dbs))}")
  is_disjoint = train_dbs.isdisjoint(test_dbs)
  print(f"Are train and test db_ids disjoint? {is_disjoint}")
  assert is_disjoint, "FATAL: Train and Test db_ids overlap!"

  print("\nDone! Saved:")
  print(f"  - {train_out} ({len(train_records)} pairs)")
  print(f"  - {test_out} ({len(test_records)} pairs)")


if __name__ == "__main__":
  main()
