"""
Shared pytest fixtures and test configurations for GRPO SQL test suite.
"""

from pathlib import Path
import sqlite3
import tempfile
from typing import Dict, List, Tuple
import pytest


@pytest.fixture
def in_memory_logistics_db() -> sqlite3.Connection:
    """
    Creates an in-memory SQLite database populated with realistic DP World logistics tables:
    containers, vessels, and shipments.
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE containers (
            container_id INTEGER PRIMARY KEY,
            tracking_number TEXT NOT NULL UNIQUE,
            weight_kg REAL NOT NULL,
            destination_port TEXT NOT NULL,
            status TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE vessels (
            vessel_id INTEGER PRIMARY KEY,
            vessel_name TEXT NOT NULL,
            capacity_teu INTEGER NOT NULL,
            flag_country TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE shipments (
            shipment_id INTEGER PRIMARY KEY,
            container_id INTEGER NOT NULL,
            vessel_id INTEGER NOT NULL,
            booking_date TEXT NOT NULL,
            delivery_date TEXT,
            FOREIGN KEY (container_id) REFERENCES containers(container_id),
            FOREIGN KEY (vessel_id) REFERENCES vessels(vessel_id)
        );
    """)

    containers = [
        (1, "CONT-001", 12000.0, "Jebel Ali", "DELIVERED"),
        (2, "CONT-002", 24000.0, "Rotterdam", "IN_TRANSIT"),
        (3, "CONT-003", 18500.0, "Jebel Ali", "IN_TRANSIT"),
        (4, "CONT-004", 31000.0, "Singapore", "DELIVERED"),
        (5, "CONT-005", 15200.0, "London Gateway", "PENDING"),
    ]
    cur.executemany("INSERT INTO containers VALUES (?, ?, ?, ?, ?);", containers)

    vessels = [
        (1, "DPW Voyager", 18000, "UAE"),
        (2, "Rotterdam Express", 14000, "Netherlands"),
        (3, "Pacific Pioneer", 21000, "Singapore"),
    ]
    cur.executemany("INSERT INTO vessels VALUES (?, ?, ?, ?);", vessels)

    shipments = [
        (1, 1, 1, "2026-11-01", "2026-11-10"),
        (2, 2, 2, "2026-11-05", None),
        (3, 3, 1, "2026-11-06", None),
        (4, 4, 3, "2026-11-02", "2026-11-12"),
    ]
    cur.executemany("INSERT INTO shipments VALUES (?, ?, ?, ?, ?);", shipments)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def temp_logistics_db(tmp_path: Path) -> Path:
    """
    Creates a physical SQLite database file on disk with the logistics schema.
    """
    db_file = tmp_path / "fixture_logistics.sqlite"
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

    cur.executemany("INSERT INTO containers VALUES (?, ?, ?, ?);", [
        (1, "CONT-0001", 15000.0, "DELIVERED"),
        (2, "CONT-0002", 22000.0, "IN_TRANSIT"),
        (3, "CONT-0003", 18000.0, "DELIVERED"),
    ])

    cur.executemany("INSERT INTO shipments VALUES (?, ?, ?);", [
        (1, 1, "Jebel Ali"),
        (2, 2, "Rotterdam"),
        (3, 3, "Jebel Ali"),
    ])

    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def project_data_dir() -> Path:
    """Returns absolute path to the data/ directory."""
    return Path(__file__).resolve().parent.parent / "data"
