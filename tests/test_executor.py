"""
Unit tests for SafeSQLExecutor.

Tests:
1. Valid SELECT returns expected rows.
2. Blocked mutating/administrative statements raise security exceptions.
3. Runaway recursive queries terminate within timeout limit (<1.0s).
4. Results are strictly truncated at 100 rows.
5. Syntax errors and schema hallucinations produce descriptive error messages.
6. Read-only and in-memory execution guarantees zero disk file mutation.
"""

from pathlib import Path
import sqlite3
import tempfile
import time
import pytest

from src.executor import SafeSQLExecutor, MAX_ROWS_LIMIT


@pytest.fixture
def sample_db_path(tmp_path: Path) -> Path:
    """Creates a temporary SQLite database with logistics sample data."""
    db_file = tmp_path / "test_logistics.sqlite"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE containers (
            container_id INTEGER PRIMARY KEY,
            tracking_number TEXT NOT NULL,
            weight_kg REAL NOT NULL,
            status TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE shipments (
            shipment_id INTEGER PRIMARY KEY,
            container_id INTEGER,
            destination_port TEXT NOT NULL,
            FOREIGN KEY (container_id) REFERENCES containers(container_id)
        );
    """)

    # Insert 150 rows to test truncation
    container_data = [
        (i, f"CONT-{i:04d}", 1000.0 + i * 10, "IN_TRANSIT" if i % 2 == 0 else "DELIVERED")
        for i in range(1, 151)
    ]
    cur.executemany("INSERT INTO containers VALUES (?, ?, ?, ?);", container_data)

    shipment_data = [
        (i, i, "Jebel Ali" if i % 3 == 0 else "Rotterdam")
        for i in range(1, 151)
    ]
    cur.executemany("INSERT INTO shipments VALUES (?, ?, ?);", shipment_data)

    conn.commit()
    conn.close()
    return db_file


def test_valid_select_returns_rows(sample_db_path: Path):
    """Test 1: Valid SELECT query executes and returns correct data."""
    sql = "SELECT container_id, tracking_number FROM containers WHERE status = 'DELIVERED' LIMIT 5;"
    success, rows, err = SafeSQLExecutor.execute(sample_db_path, sql)

    assert success is True
    assert err is None
    assert len(rows) == 5
    assert rows[0][1] == "CONT-0001"


def test_join_aggregation_query(sample_db_path: Path):
    """Test multi-table JOIN with aggregation, GROUP BY, and HAVING."""
    sql = """
        SELECT s.destination_port, COUNT(c.container_id), AVG(c.weight_kg)
        FROM containers c
        JOIN shipments s ON c.container_id = s.container_id
        GROUP BY s.destination_port
        HAVING COUNT(c.container_id) > 10;
    """
    success, rows, err = SafeSQLExecutor.execute(sample_db_path, sql)

    assert success is True
    assert err is None
    assert len(rows) == 2
    ports = {r[0] for r in rows}
    assert ports == {"Jebel Ali", "Rotterdam"}


def test_blocked_mutating_statements(sample_db_path: Path):
    """Test 2: Blocked mutating/administrative statements are rejected."""
    disallowed_queries = [
        "DROP TABLE containers;",
        "DELETE FROM containers WHERE container_id = 1;",
        "UPDATE containers SET status = 'HACKED';",
        "INSERT INTO containers VALUES (999, 'CONT-9999', 500.0, 'NEW');",
        "ALTER TABLE containers ADD COLUMN hacked_col TEXT;",
        "ATTACH DATABASE ':memory:' AS test_attach;",
        "PRAGMA table_info(containers);",
    ]

    for bad_sql in disallowed_queries:
        success, rows, err = SafeSQLExecutor.execute(sample_db_path, bad_sql)
        assert success is False, f"Expected '{bad_sql}' to fail, but it succeeded."
        assert len(rows) == 0
        assert "SecurityError" in err


def test_runaway_recursive_query_terminates(sample_db_path: Path):
    """Test 3: Runaway query with aggregation terminates within the timeout limit."""
    # Using MAX(n) forces SQLite to evaluate the infinite recursive sequence rather than lazily yielding 100 rows
    runaway_sql = """
        WITH RECURSIVE runaway(n) AS (
            SELECT 1
            UNION ALL
            SELECT n + 1 FROM runaway
        )
        SELECT MAX(n) FROM runaway;
    """
    start_time = time.perf_counter()
    success, rows, err = SafeSQLExecutor.execute(sample_db_path, runaway_sql, timeout=0.5)
    elapsed = time.perf_counter() - start_time

    assert success is False
    assert len(rows) == 0
    assert "TimeoutError" in err
    assert elapsed < 1.5, f"Execution took too long: {elapsed:.2f}s"


def test_truncation_at_100_rows(sample_db_path: Path):
    """Test 4: Output rows are strictly truncated at MAX_ROWS_LIMIT (100)."""
    sql = "SELECT * FROM containers;"
    success, rows, err = SafeSQLExecutor.execute(sample_db_path, sql)

    assert success is True
    assert err is None
    assert len(rows) == MAX_ROWS_LIMIT, f"Expected {MAX_ROWS_LIMIT} rows, got {len(rows)}"


def test_syntax_error_handling(sample_db_path: Path):
    """Test 5A: Malformed SQL raises syntax error."""
    bad_syntax = "SELECT * FORM containers;"
    success, rows, err = SafeSQLExecutor.execute(sample_db_path, bad_syntax)

    assert success is False
    assert len(rows) == 0
    assert "OperationalError" in err or "syntax error" in err.lower()


def test_schema_hallucination_handling(sample_db_path: Path):
    """Test 5B: Hallucinated table or column raises descriptive error."""
    hallucinated_sql = "SELECT imaginary_col FROM non_existent_table;"
    success, rows, err = SafeSQLExecutor.execute(sample_db_path, hallucinated_sql)

    assert success is False
    assert len(rows) == 0
    assert "no such table" in err.lower()


def test_disk_immutability(sample_db_path: Path):
    """Test 6: Verify disk database file is never modified."""
    # Record initial state
    conn = sqlite3.connect(sample_db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM containers;")
    initial_count = cur.fetchone()[0]
    conn.close()

    # Execute queries
    SafeSQLExecutor.execute(sample_db_path, "SELECT * FROM containers;")
    SafeSQLExecutor.execute(sample_db_path, "DROP TABLE containers;")

    # Verify count is identical
    conn = sqlite3.connect(sample_db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM containers;")
    final_count = cur.fetchone()[0]
    conn.close()

    assert initial_count == final_count == 150


def test_real_dataset_database_execution():
    """Test execution against one of the downloaded Spider databases."""
    db_path = Path("data/databases/ship_1/ship_1.sqlite")
    if not db_path.exists():
        pytest.skip("data/databases/ship_1/ship_1.sqlite not present")

    sql = "SELECT count(DISTINCT rank) FROM captain;"
    success, rows, err = SafeSQLExecutor.execute(db_path, sql)

    assert success is True
    assert err is None
    assert len(rows) == 1
    assert rows[0][0] > 0
