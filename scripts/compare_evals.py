"""
Comparative analysis script for evaluation harness v2 results.

Compares docs/eval_v2_base.json and docs/eval_v2_grpo.json:
- Computes Wilson 95% confidence intervals for EX, VSR, FAR.
- Computes conditional accuracy (correct / SQL block emitted).
- Computes paired 2x2 contingency matrix (both, only base, only GRPO, neither).
- Computes exact two-sided McNemar test p-value.
- Computes per-database breakdown (ship_1 vs tracking_orders).
- Outputs docs/comparison_v2.json and prints human-readable tables.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.statistics import exact_mcnemar_test, wilson_score_interval


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison(base_data: Dict[str, Any], grpo_data: Dict[str, Any]) -> Dict[str, Any]:
    total = base_data["total_queries"]
    assert total == grpo_data["total_queries"], "Mismatched total query counts"

    base_detailed = base_data["detailed_results"]
    grpo_detailed = grpo_data["detailed_results"]
    assert len(base_detailed) == len(grpo_detailed) == total

    # 1. Metric summaries with Wilson 95% CIs
    def get_metric_pack(data: Dict[str, Any]) -> Dict[str, Any]:
        c = data["counts"]
        m = data["metrics"]
        ex_ci = wilson_score_interval(c["correct"], total)
        vsr_ci = wilson_score_interval(c["valid_sql"], total)
        far_ci = wilson_score_interval(c["format_adherent"], total)

        # Emitted SQL blocks (has_markdown or valid SQL)
        emitted_count = sum(1 for r in data["detailed_results"] if r.get("has_markdown", False))
        cond_acc = (c["correct"] / emitted_count * 100.0) if emitted_count > 0 else 0.0

        ts = data.get("token_stats", {})
        return {
            "execution_accuracy": {
                "count": c["correct"],
                "total": total,
                "pct": m["execution_accuracy_pct"],
                "wilson_ci_95": list(ex_ci),
            },
            "valid_sql_rate": {
                "count": c["valid_sql"],
                "total": total,
                "pct": m["valid_sql_rate_pct"],
                "wilson_ci_95": list(vsr_ci),
            },
            "format_adherence_rate": {
                "count": c["format_adherent"],
                "total": total,
                "pct": m["format_adherence_rate_pct"],
                "wilson_ci_95": list(far_ci),
            },
            "conditional_accuracy_on_emitted_sql": {
                "correct": c["correct"],
                "emitted": emitted_count,
                "pct": round(cond_acc, 2),
            },
            "token_stats": {
                "total_hit_token_limit": ts.get("total_hit_token_limit", 0),
                "hit_token_limit_pct": ts.get("hit_token_limit_pct", 0.0),
                "avg_generated_tokens": ts.get("avg_generated_tokens", 0.0),
            },
            "failure_taxonomy": data["failure_taxonomy"],
            "per_database": data["per_database"],
        }

    base_metrics = get_metric_pack(base_data)
    grpo_metrics = get_metric_pack(grpo_data)

    # 2. Paired 2x2 Contingency Analysis
    both_correct = 0
    only_base_correct = 0  # b
    only_grpo_correct = 0  # c
    neither_correct = 0

    per_db_paired: Dict[str, Dict[str, int]] = {}

    for b_item, g_item in zip(base_detailed, grpo_detailed):
        assert b_item["index"] == g_item["index"]
        db_id = b_item["db_id"]
        if db_id not in per_db_paired:
            per_db_paired[db_id] = {
                "both_correct": 0,
                "only_base_correct": 0,
                "only_grpo_correct": 0,
                "neither_correct": 0,
            }

        b_corr = bool(b_item["is_match"])
        g_corr = bool(g_item["is_match"])

        if b_corr and g_corr:
            both_correct += 1
            per_db_paired[db_id]["both_correct"] += 1
        elif b_corr and not g_corr:
            only_base_correct += 1
            per_db_paired[db_id]["only_base_correct"] += 1
        elif not b_corr and g_corr:
            only_grpo_correct += 1
            per_db_paired[db_id]["only_grpo_correct"] += 1
        else:
            neither_correct += 1
            per_db_paired[db_id]["neither_correct"] += 1

    mcnemar_p = exact_mcnemar_test(b=only_base_correct, c=only_grpo_correct)

    paired_table = {
        "contingency_matrix": {
            "both_correct": both_correct,
            "only_base_correct_b": only_base_correct,
            "only_grpo_correct_c": only_grpo_correct,
            "neither_correct": neither_correct,
            "total": total,
        },
        "mcnemar_test": {
            "discordant_pairs": only_base_correct + only_grpo_correct,
            "b_base_better": only_base_correct,
            "c_grpo_better": only_grpo_correct,
            "net_shift": only_grpo_correct - only_base_correct,
            "exact_two_sided_p_value": mcnemar_p,
            "is_statistically_significant_005": bool(mcnemar_p < 0.05),
        },
        "per_database_paired": per_db_paired,
    }

    return {
        "metadata": {
            "base_model": base_data.get("model_path"),
            "grpo_adapter": grpo_data.get("adapter_path"),
            "precision": base_data.get("precision"),
            "max_new_tokens": base_data.get("max_new_tokens"),
            "total_queries": total,
        },
        "base_model": base_metrics,
        "grpo_model": grpo_metrics,
        "paired_analysis": paired_table,
    }


def print_comparison_tables(comp: Dict[str, Any]) -> None:
    meta = comp["metadata"]
    b = comp["base_model"]
    g = comp["grpo_model"]
    p = comp["paired_analysis"]
    cm = p["contingency_matrix"]
    mc = p["mcnemar_test"]

    print("\n" + "=" * 80)
    print("V2 PAIRED EVALUATION COMPARISON REPORT")
    print(f"Setting: Precision={meta['precision']} | max_new_tokens={meta['max_new_tokens']} | Total Queries={meta['total_queries']}")
    print("=" * 80)

    # 1. Headline Metrics Table
    print("\n1. PRIMARY METRICS (WITH WILSON 95% CONFIDENCE INTERVALS):")
    print("-" * 80)
    print(f"{'Metric':<25} {'Base Model':<24} {'GRPO Model':<24} {'Absolute Delta':<10}")
    print("-" * 80)

    for m_key, label in [
        ("execution_accuracy", "Execution Accuracy (EX)"),
        ("valid_sql_rate", "Valid SQL Rate (VSR)"),
        ("format_adherence_rate", "Format Adherence (FAR)"),
    ]:
        b_m = b[m_key]
        g_m = g[m_key]
        b_str = f"{b_m['pct']:.2f}% [{b_m['wilson_ci_95'][0]:.1f}, {b_m['wilson_ci_95'][1]:.1f}]"
        g_str = f"{g_m['pct']:.2f}% [{g_m['wilson_ci_95'][0]:.1f}, {g_m['wilson_ci_95'][1]:.1f}]"
        diff = g_m["pct"] - b_m["pct"]
        print(f"{label:<25} {b_str:<24} {g_str:<24} {diff:+.2f}%")
    print("-" * 80)

    # 2. Conditional Accuracy & Token Stats
    print("\n2. CONDITIONAL ACCURACY & GENERATION PROFILE:")
    b_cond = b["conditional_accuracy_on_emitted_sql"]
    g_cond = g["conditional_accuracy_on_emitted_sql"]
    print(f"  Conditional Accuracy (Correct / Emitted SQL):")
    print(f"    Base: {b_cond['pct']:.2f}% ({b_cond['correct']}/{b_cond['emitted']})")
    print(f"    GRPO: {g_cond['pct']:.2f}% ({g_cond['correct']}/{g_cond['emitted']})")

    b_tok = b["token_stats"]
    g_tok = g["token_stats"]
    print(f"  Hit Token Limit (>=512):")
    print(f"    Base: {b_tok['total_hit_token_limit']}/{meta['total_queries']} ({b_tok['hit_token_limit_pct']}%) | Avg Tokens: {b_tok['avg_generated_tokens']}")
    print(f"    GRPO: {g_tok['total_hit_token_limit']}/{meta['total_queries']} ({g_tok['hit_token_limit_pct']}%) | Avg Tokens: {g_tok['avg_generated_tokens']}")

    # 3. 6-Class Taxonomy Breakdown
    print("\n3. 6-CLASS FAILURE TAXONOMY BREAKDOWN:")
    print("-" * 75)
    print(f"{'Category':<25} {'Base Count':<15} {'GRPO Count':<15} {'Shift':<10}")
    print("-" * 75)
    for cat in b["failure_taxonomy"]:
        b_cnt = b["failure_taxonomy"][cat]
        g_cnt = g["failure_taxonomy"][cat]
        shift = g_cnt - b_cnt
        print(f"{cat:<25} {b_cnt:<15} {g_cnt:<15} {shift:+d}")
    print("-" * 75)

    # 4. Paired Contingency Matrix & McNemar Test
    print("\n4. PAIRED 2x2 CONTINGENCY MATRIX (108 QUESTIONS):")
    print("-" * 60)
    print(f"  Both Models Correct:             {cm['both_correct']:<4} ({cm['both_correct']/meta['total_queries']*100:.1f}%)")
    print(f"  Only Base Correct (b):           {cm['only_base_correct_b']:<4} ({cm['only_base_correct_b']/meta['total_queries']*100:.1f}%)")
    print(f"  Only GRPO Correct (c):           {cm['only_grpo_correct_c']:<4} ({cm['only_grpo_correct_c']/meta['total_queries']*100:.1f}%)")
    print(f"  Neither Model Correct:           {cm['neither_correct']:<4} ({cm['neither_correct']/meta['total_queries']*100:.1f}%)")
    print("-" * 60)
    print(f"  Discordant Pairs (b + c):        {mc['discordant_pairs']}")
    print(f"  Net Shift (c - b):               {mc['net_shift']:+d} queries")
    print(f"  Exact McNemar Test p-value:      p = {mc['exact_two_sided_p_value']:.4f}")
    sig_str = "YES (Significant)" if mc["is_statistically_significant_005"] else "NO (NOT Significant; within statistical noise)"
    print(f"  Statistically Significant (0.05):{sig_str}")
    print("-" * 60)

    # 5. Per-Database Breakdown
    print("\n5. PER-DATABASE BREAKDOWN:")
    print("-" * 75)
    print(f"{'Database':<20} {'Base Correct':<18} {'GRPO Correct':<18} {'Delta':<10}")
    print("-" * 75)
    for db in b["per_database"]:
        b_db = b["per_database"][db]
        g_db = g["per_database"][db]
        b_db_pct = (b_db["correct"] / b_db["total"]) * 100.0
        g_db_pct = (g_db["correct"] / g_db["total"]) * 100.0
        b_s = f"{b_db['correct']}/{b_db['total']} ({b_db_pct:.1f}%)"
        g_s = f"{g_db['correct']}/{g_db['total']} ({g_db_pct:.1f}%)"
        delta = g_db["correct"] - b_db["correct"]
        print(f"{db:<20} {b_s:<18} {g_s:<18} {delta:+d} ({g_db_pct - b_db_pct:+.1f}%)")
    print("=" * 80 + "\n")


def main() -> None:
    base_file = Path("docs/eval_v2_base.json")
    grpo_file = Path("docs/eval_v2_grpo.json")
    out_file = Path("docs/comparison_v2.json")

    assert base_file.exists(), f"Missing {base_file}"
    assert grpo_file.exists(), f"Missing {grpo_file}"

    base_data = load_json(base_file)
    grpo_data = load_json(grpo_file)

    comp = build_comparison(base_data, grpo_data)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2)

    print(f"Comparison JSON saved to {out_file}.")
    print_comparison_tables(comp)


if __name__ == "__main__":
    main()
