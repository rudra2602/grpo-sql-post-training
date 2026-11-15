"""
Safe in-memory SQLite query executor with security guards and execution timeouts.

Enforces zero data mutation, authorizer-level sandboxing, deterministic 1.0s timeout,
and output row truncation to prevent cartesian explosion or GPU hangs during RL rollouts.
"""

from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, List, Optional, Tuple, Union

# Action codes explicitly denied by the SQLite authorizer callback
DENIED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_DROP_VTABLE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_CREATE_VTABLE,
}

# Static regex guard to reject dangerous statements before invoking the SQLite engine
DISALLOWED_KEYWORDS_REGEX = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|ATTACH|DETACH|PRAGMA|LOAD_EXTENSION|WRITEFILE)\b",
    re.IGNORECASE,
)

MAX_ROWS_LIMIT = 100
DEFAULT_TIMEOUT_SECONDS = 1.0


def _authorizer_callback(
    action: int,
    arg1: Optional[str],
    arg2: Optional[str],
    db_name: Optional[str],
    trigger_name: Optional[str],
) -> int:
  """Authorizer hook invoked by SQLite AST compiler.

  Returns SQLITE_DENY for any mutating, administrative, or extension-loading
  actions.
  """
  if action in DENIED_AUTHORIZER_ACTIONS:
    return sqlite3.SQLITE_DENY
  return sqlite3.SQLITE_OK


class SafeSQLExecutor:
  """Sandboxed SQLite executor for evaluating generated SQL queries safely."""

  @classmethod
  def execute(
      cls,
      db_path: Union[str, Path, sqlite3.Connection],
      sql: str,
      timeout: float = DEFAULT_TIMEOUT_SECONDS,
  ) -> Tuple[bool, List[Tuple[Any, ...]], Optional[str]]:
    """Execute SQL query against an in-memory copy of the database.

    Args:
        db_path: Path to the SQLite database file, ':memory:', or existing
          sqlite3.Connection.
        sql: The SQL query string to execute.
        timeout: Hard timeout in seconds (default 1.0s).

    Returns:
        Tuple of (success, rows, error_message):
            - success: True if executed without error, False otherwise.
            - rows: List of tuples up to MAX_ROWS_LIMIT rows.
            - error_message: None on success, or string describing the error.
    """
    if not sql or not sql.strip():
      return False, [], "Empty SQL query."

    # 1. Static security check
    cleaned_sql = sql.strip()
    match = DISALLOWED_KEYWORDS_REGEX.search(cleaned_sql)
    if match:
      blocked_keyword = match.group(0).upper()
      return (
          False,
          [],
          (
              f"SecurityError: Disallowed SQL keyword '{blocked_keyword}'"
              " detected."
          ),
      )

    # 2. Establish ephemeral in-memory database copy
    mem_conn: Optional[sqlite3.Connection] = None
    disk_conn: Optional[sqlite3.Connection] = None
    timer: Optional[threading.Timer] = None

    try:
      if isinstance(db_path, sqlite3.Connection):
        mem_conn = sqlite3.connect(":memory:")
        db_path.backup(mem_conn)
      elif str(db_path) == ":memory:":
        mem_conn = sqlite3.connect(":memory:")
      else:
        path_obj = Path(db_path)
        if not path_obj.exists():
          return (
              False,
              [],
              f"DatabaseNotFoundError: Database file '{path_obj}' does not"
              " exist.",
          )

        # Open source in strict read-only mode, then clone into memory
        file_uri = f"{path_obj.resolve().as_uri()}?mode=ro"
        disk_conn = sqlite3.connect(file_uri, uri=True)
        mem_conn = sqlite3.connect(":memory:")
        disk_conn.backup(mem_conn)
        disk_conn.close()
        disk_conn = None

      # 3. Configure authorizer guard on the in-memory instance
      mem_conn.set_authorizer(_authorizer_callback)

      # 4. Configure dual timeout guards:
      #    A. Progress handler (checks opcode count and wall-clock time)
      #    B. Threading timer interrupt (breaks execution if stuck in C extension/wait)
      start_time = time.perf_counter()

      def _progress_handler() -> int:
        if (time.perf_counter() - start_time) > timeout:
          return 1  # 1 interrupts SQLite execution immediately
        return 0

      mem_conn.set_progress_handler(_progress_handler, 1000)

      timer = threading.Timer(timeout, mem_conn.interrupt)
      timer.daemon = True
      timer.start()

      # 5. Execute query and fetch up to MAX_ROWS_LIMIT rows
      cursor = mem_conn.cursor()
      cursor.execute(cleaned_sql)
      rows = cursor.fetchmany(MAX_ROWS_LIMIT)

      return True, rows, None

    except sqlite3.OperationalError as e:
      err_msg = str(e).lower()
      if "interrupted" in err_msg or "interrupt" in err_msg:
        return (
            False,
            [],
            f"TimeoutError: Execution exceeded timeout limit of {timeout}s.",
        )
      return False, [], f"OperationalError: {e}"

    except sqlite3.DatabaseError as e:
      err_msg = str(e).lower()
      if "not authorized" in err_msg:
        return False, [], f"SecurityError: {e}"
      return False, [], f"DatabaseError: {e}"

    except Exception as e:
      return False, [], f"ExecutionError: {e}"

    finally:
      if timer is not None:
        timer.cancel()
      if disk_conn is not None:
        try:
          disk_conn.close()
        except Exception:
          pass
      if mem_conn is not None:
        try:
          mem_conn.close()
        except Exception:
          pass
