"""
Unit tests for the evaluation harness and failure taxonomy classifier.

Verifies:
1. Deterministic classification across all 5 error categories:
   - SYNTAX_ERROR
   - SCHEMA_HALLUCINATION
   - TIMEOUT_ABORT
   - EMPTY_RESULT
   - VALUE_MISMATCH
2. evaluate_single_prediction metrics and schema handling.
"""

from pathlib import Path
import pytest

from src.evaluate import (
    classify_failure,
    evaluate_single_prediction,
    TAXONOMY_CORRECT,
    TAXONOMY_SYNTAX_ERROR,
    TAXONOMY_SCHEMA_HALLUCINATION,
    TAXONOMY_TIMEOUT_ABORT,
    TAXONOMY_EMPTY_RESULT,
    TAXONOMY_VALUE_MISMATCH,
)


def test_classify_correct():
    """Verify correct predictions are assigned CORRECT."""
    cat = classify_failure(
        exec_success=True,
        pred_rows=[(1, "A")],
        err_msg=None,
        gold_rows=[(1, "A")],
        is_match=True,
    )
    assert cat == TAXONOMY_CORRECT


def test_classify_syntax_error():
    """Verify SQLite syntax errors map to SYNTAX_ERROR."""
    cat = classify_failure(
        exec_success=False,
        pred_rows=[],
        err_msg="OperationalError: near 'FROM': syntax error",
        gold_rows=[(1,)],
        is_match=False,
    )
    assert cat == TAXONOMY_SYNTAX_ERROR


def test_classify_schema_hallucination():
    """Verify non-existent tables or columns map to SCHEMA_HALLUCINATION."""
    cat_table = classify_failure(
        exec_success=False,
        pred_rows=[],
        err_msg="OperationalError: no such table: non_existent_table",
        gold_rows=[(1,)],
        is_match=False,
    )
    assert cat_table == TAXONOMY_SCHEMA_HALLUCINATION

    cat_col = classify_failure(
        exec_success=False,
        pred_rows=[],
        err_msg="OperationalError: no such column: imaginary_col",
        gold_rows=[(1,)],
        is_match=False,
    )
    assert cat_col == TAXONOMY_SCHEMA_HALLUCINATION


def test_classify_timeout_abort():
    """Verify runaway executions map to TIMEOUT_ABORT."""
    cat = classify_failure(
        exec_success=False,
        pred_rows=[],
        err_msg="TimeoutError: Execution exceeded timeout limit of 1.0s.",
        gold_rows=[(1,)],
        is_match=False,
    )
    assert cat == TAXONOMY_TIMEOUT_ABORT


def test_classify_empty_result():
    """Verify empty predicted results against non-empty gold map to EMPTY_RESULT."""
    cat = classify_failure(
        exec_success=True,
        pred_rows=[],
        err_msg=None,
        gold_rows=[("CONT-001",), ("CONT-002",)],
        is_match=False,
    )
    assert cat == TAXONOMY_EMPTY_RESULT


def test_classify_value_mismatch():
    """Verify clean executions returning wrong rows map to VALUE_MISMATCH."""
    cat = classify_failure(
        exec_success=True,
        pred_rows=[(100,)],
        err_msg=None,
        gold_rows=[(200,)],
        is_match=False,
    )
    assert cat == TAXONOMY_VALUE_MISMATCH


def test_evaluate_single_prediction_end_to_end(temp_logistics_db: Path):
    """Verify evaluate_single_prediction works end-to-end against a real database."""
    gold_sql = "SELECT container_id FROM containers WHERE status = 'DELIVERED';"

    # Case A: Correct query
    pred_correct = "```sql\nSELECT container_id FROM containers WHERE status = 'DELIVERED';\n```"
    res_correct = evaluate_single_prediction(pred_correct, gold_sql, temp_logistics_db)
    assert res_correct["is_match"] is True
    assert res_correct["error_category"] == TAXONOMY_CORRECT
    assert res_correct["has_markdown"] is True
    assert res_correct["pred_success"] is True

    # Case B: Value mismatch
    pred_wrong = "```sql\nSELECT container_id FROM containers WHERE status = 'IN_TRANSIT';\n```"
    res_wrong = evaluate_single_prediction(pred_wrong, gold_sql, temp_logistics_db)
    assert res_wrong["is_match"] is False
    assert res_wrong["error_category"] == TAXONOMY_VALUE_MISMATCH

    # Case C: Hallucinated schema
    pred_hallu = "```sql\nSELECT non_existent FROM containers;\n```"
    res_hallu = evaluate_single_prediction(pred_hallu, gold_sql, temp_logistics_db)
    assert res_hallu["is_match"] is False
    assert res_hallu["error_category"] == TAXONOMY_SCHEMA_HALLUCINATION
