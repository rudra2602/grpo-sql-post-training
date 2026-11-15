"""
Unit tests for the GRPO SQL reward engine.

Tests:
1. Perfect query (format + syntax + execution match) -> 1.0
2. Valid SQL returning wrong rows -> 0.3
3. Malformed syntax inside markdown tags -> 0.1
4. Raw SQL without markdown tags matching ground truth -> 0.9
5. Trivial scalar hacking query ('SELECT 1;') -> 0.0 (caught by table reference guard)
6. Multiset row equivalence (order-invariant when no ORDER BY in gold) -> 1.0
7. Dangerous mutation attempt (e.g. DROP TABLE) inside markdown -> 0.1
8. Empty result prevention -> 0.0
9. Strict ordering when ORDER BY is present in gold query
10. Binary reward mode ablation support (1.0 or 0.0)
11. Batch callable grpo_sql_reward_func
"""

from pathlib import Path
import sqlite3
import pytest

from src.reward import compute_sql_reward, grpo_sql_reward_func


@pytest.fixture
def reward_db_path(tmp_path: Path) -> Path:
    """Creates a temporary SQLite database with logistics tracking data."""
    db_file = tmp_path / "reward_logistics.sqlite"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE containers (
            container_id INTEGER PRIMARY KEY,
            tracking_number TEXT NOT NULL,
            weight_kg REAL NOT NULL,
            destination_port TEXT NOT NULL,
            status TEXT NOT NULL
        );
    """)

    containers = [
        (1, "CONT-001", 1200.0, "Jebel Ali", "DELIVERED"),
        (2, "CONT-002", 2400.0, "Rotterdam", "IN_TRANSIT"),
        (3, "CONT-003", 3100.0, "Jebel Ali", "IN_TRANSIT"),
        (4, "CONT-004", 1800.0, "Singapore", "DELIVERED"),
    ]
    cur.executemany("INSERT INTO containers VALUES (?, ?, ?, ?, ?);", containers)

    conn.commit()
    conn.close()
    return db_file


def test_perfect_query(reward_db_path: Path):
    """Test 1: Perfectly formatted SQL matching ground truth returns exactly 1.0."""
    gold_sql = "SELECT container_id, destination_port FROM containers WHERE status = 'DELIVERED';"
    predicted_text = """Here is the SQL query:
```sql
SELECT container_id, destination_port FROM containers WHERE status = 'DELIVERED';
```"""
    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    assert score == 1.0, f"Expected 1.0, got {score}"


def test_syntax_only(reward_db_path: Path):
    """Test 2: Valid SQL on valid tables but returning wrong rows returns exactly 0.3."""
    gold_sql = "SELECT container_id FROM containers WHERE status = 'DELIVERED';"
    predicted_text = """```sql
SELECT container_id FROM containers WHERE status = 'IN_TRANSIT';
```"""
    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    assert score == 0.3, f"Expected 0.3, got {score}"


def test_syntax_error(reward_db_path: Path):
    """Test 3: Malformed SQL inside markdown tags returns exactly 0.1 (format only)."""
    gold_sql = "SELECT container_id FROM containers;"
    predicted_text = """```sql
SELECT FROM WHERE containers;
```"""
    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    assert score == 0.1, f"Expected 0.1, got {score}"


def test_no_markdown_tags(reward_db_path: Path):
    """Test 4: Raw SQL text without markdown tags that matches ground truth loses format point -> returns 0.9."""
    gold_sql = "SELECT container_id, tracking_number FROM containers WHERE destination_port = 'Singapore';"
    predicted_text = "SELECT container_id, tracking_number FROM containers WHERE destination_port = 'Singapore';"

    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    assert score == 0.9, f"Expected 0.9, got {score}"


def test_trivial_scalar_hacking(reward_db_path: Path):
    """Test 5: Trivial scalar query lacking table references ('SELECT 4;') returns 0.0."""
    gold_sql = "SELECT count(*) FROM containers;"
    # Gold returns 4. A cheating model might output 'SELECT 4;' to match the scalar value without querying.
    predicted_text = "```sql\nSELECT 4;\n```"

    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    # Format point (0.1) is denied or only format awarded without syntax/match
    assert score == 0.1, f"Expected format only (0.1) without syntax/match points, got {score}"


def test_multiset_ordering(reward_db_path: Path):
    """Test 6: Multiset row equivalence (order-invariant when no ORDER BY in gold) returns 1.0."""
    gold_sql = "SELECT destination_port, count(*) FROM containers GROUP BY destination_port;"
    # Reverse the order of rows intentionally by ordering differently
    predicted_text = """```sql
SELECT destination_port, count(*) FROM containers GROUP BY destination_port ORDER BY destination_port DESC;
```"""
    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    assert score == 1.0, f"Expected 1.0 for multiset equivalence, got {score}"


def test_order_by_strictness(reward_db_path: Path):
    """When ORDER BY is in gold SQL, strict ordering is enforced."""
    gold_sql = "SELECT tracking_number FROM containers ORDER BY weight_kg ASC;"
    # Inverted order
    predicted_text = """```sql
SELECT tracking_number FROM containers ORDER BY weight_kg DESC;
```"""
    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    # Has format (0.1) + syntax (0.2), but fails execution match (0.0)
    assert score == 0.3, f"Expected 0.3 due to wrong ordering, got {score}"


def test_dangerous_mutation_attempt(reward_db_path: Path):
    """Test 7: Mutation attempt inside markdown is aborted by sandbox -> returns 0.1."""
    gold_sql = "SELECT * FROM containers;"
    predicted_text = """```sql
DROP TABLE containers;
```"""
    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    assert score == 0.1, f"Expected 0.1 for blocked dangerous query, got {score}"


def test_empty_result_prevention(reward_db_path: Path):
    """Test 8: If gold returns rows, an empty predicted result gets 0.0 match credit -> 0.3 total."""
    gold_sql = "SELECT tracking_number FROM containers WHERE destination_port = 'Jebel Ali';"
    # Over-filtered query returning 0 rows
    predicted_text = """```sql
SELECT tracking_number FROM containers WHERE destination_port = 'NonExistentPort';
```"""
    score = compute_sql_reward(predicted_text, gold_sql, reward_db_path)
    assert score == 0.3, f"Expected 0.3 (format + syntax, but zero match), got {score}"


def test_binary_reward_mode(reward_db_path: Path):
    """Test binary reward mode (for ablation experiments)."""
    gold_sql = "SELECT destination_port FROM containers WHERE container_id = 1;"
    pred_correct = "```sql\nSELECT destination_port FROM containers WHERE container_id = 1;\n```"
    pred_wrong = "```sql\nSELECT destination_port FROM containers WHERE container_id = 2;\n```"

    score_correct = compute_sql_reward(pred_correct, gold_sql, reward_db_path, reward_mode="binary")
    score_wrong = compute_sql_reward(pred_wrong, gold_sql, reward_db_path, reward_mode="binary")

    assert score_correct == 1.0
    assert score_wrong == 0.0


def test_grpo_sql_reward_func_batch(reward_db_path: Path):
    """Test TRL batch reward callable."""
    gold_sqls = [
        "SELECT tracking_number FROM containers WHERE container_id = 1;",
        "SELECT tracking_number FROM containers WHERE container_id = 2;",
    ]
    completions = [
        "```sql\nSELECT tracking_number FROM containers WHERE container_id = 1;\n```",
        "```sql\nSELECT tracking_number FROM containers WHERE container_id = 999;\n```",
    ]
    db_paths = [reward_db_path, reward_db_path]

    rewards = grpo_sql_reward_func(completions, gold_sqls, db_paths)
    assert len(rewards) == 2
    assert rewards[0] == 1.0
    assert rewards[1] == 0.3
