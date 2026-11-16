"""
Evaluation harness and automated 5-class failure taxonomy classifier for text-to-SQL models.

Evaluates greedy decoding (T=0.0) on held-out test questions, computing:
  - Execution Accuracy (EX %)
  - Valid SQL Rate (VSR %)
  - Format Adherence Rate (FAR %)
  - 5-Class Error Taxonomy (SYNTAX_ERROR, SCHEMA_HALLUCINATION, TIMEOUT_ABORT, EMPTY_RESULT, VALUE_MISMATCH)
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path when invoked directly as python src/evaluate.py
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import format_sql_prompt, load_logistics_dataset
from src.executor import SafeSQLExecutor
from src.reward import check_result_equivalence, extract_sql_query

TAXONOMY_NO_SQL_EXTRACTED = "NO_SQL_EXTRACTED"
TAXONOMY_SYNTAX_ERROR = "SYNTAX_ERROR"
TAXONOMY_SCHEMA_HALLUCINATION = "SCHEMA_HALLUCINATION"
TAXONOMY_TIMEOUT_ABORT = "TIMEOUT_ABORT"
TAXONOMY_EMPTY_RESULT = "EMPTY_RESULT"
TAXONOMY_VALUE_MISMATCH = "VALUE_MISMATCH"
TAXONOMY_CORRECT = "CORRECT"

ALL_TAXONOMY_CATEGORIES = [
    TAXONOMY_NO_SQL_EXTRACTED,
    TAXONOMY_SYNTAX_ERROR,
    TAXONOMY_SCHEMA_HALLUCINATION,
    TAXONOMY_TIMEOUT_ABORT,
    TAXONOMY_EMPTY_RESULT,
    TAXONOMY_VALUE_MISMATCH,
]


def classify_failure(
    exec_success: bool,
    pred_rows: List[Tuple[Any, ...]],
    err_msg: Optional[str],
    gold_rows: List[Tuple[Any, ...]],
    is_match: bool,
    has_markdown: bool = True,
    extracted_sql: str = "",
) -> str:
  """Classify query execution into the 6-class deterministic failure taxonomy.

  Returns TAXONOMY_CORRECT if is_match is True, otherwise one of the 6 error
  codes:
    1. NO_SQL_EXTRACTED: Outputs with no fenced SQL block and no query starting with SELECT/WITH.
    2. TIMEOUT_ABORT: Execution exceeded 1.0s limit.
    3. SCHEMA_HALLUCINATION: Reference to non-existent table or column.
    4. SYNTAX_ERROR: SQLite parser failure or malformed query.
    5. EMPTY_RESULT: Query returned 0 rows while gold returned data.
    6. VALUE_MISMATCH: Clean execution with non-empty rows differing from gold.
  """
  if is_match:
    return TAXONOMY_CORRECT

  # Check if SQL was extracted: either has markdown fenced block, or query starts with SELECT/WITH
  cleaned_sql = (extracted_sql or "").strip().upper()
  starts_with_sql = cleaned_sql.startswith("SELECT") or cleaned_sql.startswith("WITH")
  if not has_markdown and not starts_with_sql:
    return TAXONOMY_NO_SQL_EXTRACTED

  err_lower = (err_msg or "").lower()

  if not exec_success:
    if "timeout" in err_lower or "interrupted" in err_lower:
      return TAXONOMY_TIMEOUT_ABORT
    if "no such table" in err_lower or "no such column" in err_lower:
      return TAXONOMY_SCHEMA_HALLUCINATION
    return TAXONOMY_SYNTAX_ERROR

  # Executed cleanly but failed match
  if len(pred_rows) == 0 and len(gold_rows) > 0:
    return TAXONOMY_EMPTY_RESULT

  return TAXONOMY_VALUE_MISMATCH


def evaluate_single_prediction(
    predicted_text: str,
    gold_sql: str,
    db_path: Union[str, Path],
) -> Dict[str, Any]:
  """Evaluate a single prediction against gold SQL using SafeSQLExecutor."""
  pred_sql, has_markdown = extract_sql_query(predicted_text)

  # Execute prediction
  pred_success, pred_rows, pred_err = SafeSQLExecutor.execute(db_path, pred_sql)

  # Execute gold query
  gold_success, gold_rows, gold_err = SafeSQLExecutor.execute(db_path, gold_sql)

  # Check equivalence
  is_match = False
  if pred_success and gold_success:
    if len(gold_rows) > 0 and len(pred_rows) == 0:
      is_match = False
    elif len(gold_rows) == 0 and len(pred_rows) == 0:
      is_match = True
    else:
      is_match = check_result_equivalence(pred_rows, gold_rows, gold_sql)

  # Classify failure
  error_category = classify_failure(
      exec_success=pred_success,
      pred_rows=pred_rows,
      err_msg=pred_err,
      gold_rows=gold_rows,
      is_match=is_match,
      has_markdown=has_markdown,
      extracted_sql=pred_sql,
  )

  return {
      "predicted_text": predicted_text,
      "extracted_sql": pred_sql,
      "has_markdown": has_markdown,
      "pred_success": pred_success,
      "pred_error": pred_err,
      "pred_row_count": len(pred_rows),
      "gold_row_count": len(gold_rows),
      "is_match": is_match,
      "error_category": error_category,
  }


def run_evaluation(
    model: Any,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    device: str = "cpu",
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
  """Run greedy decoding evaluation across a dataset split."""
  results: List[Dict[str, Any]] = []

  total_queries = len(dataset)
  correct_count = 0
  valid_sql_count = 0
  format_adherent_count = 0

  taxonomy_counts: Dict[str, int] = {
      cat: 0 for cat in ALL_TAXONOMY_CATEGORIES
  }
  per_db_accuracy: Dict[str, Dict[str, int]] = {}

  for idx, record in enumerate(dataset):
    prompt = record["prompt"]
    gold_sql = record["gold_sql"]
    db_path = record["db_path"]
    db_id = record["db_id"]
    difficulty = record.get("difficulty", "medium")

    if db_id not in per_db_accuracy:
      per_db_accuracy[db_id] = {"total": 0, "correct": 0}
    per_db_accuracy[db_id]["total"] += 1

    # Format for chat model if template is supported
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
      messages = [{"role": "user", "content": prompt}]
      formatted_input = tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
    else:
      formatted_input = prompt

    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

    # Deterministic greedy decoding (temperature=None / do_sample=False)
    import torch

    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          pad_token_id=tokenizer.eos_token_id,
      )

    # Slice off input tokens to get completion only
    input_len = inputs["input_ids"].shape[1]
    completion_tokens = outputs[0][input_len:]
    generated_tokens = int(len(completion_tokens))
    hit_token_limit = bool(generated_tokens >= max_new_tokens)
    completion_text = tokenizer.decode(
        completion_tokens, skip_special_tokens=True
    ).strip()

    eval_result = evaluate_single_prediction(completion_text, gold_sql, db_path)
    eval_result.update({
        "index": idx,
        "db_id": db_id,
        "question": record["question"],
        "gold_sql": gold_sql,
        "difficulty": difficulty,
        "generated_tokens": generated_tokens,
        "hit_token_limit": hit_token_limit,
    })
    results.append(eval_result)

    if eval_result["is_match"]:
      correct_count += 1
      per_db_accuracy[db_id]["correct"] += 1
    if eval_result["pred_success"]:
      valid_sql_count += 1
    if eval_result["has_markdown"]:
      format_adherent_count += 1

    cat = eval_result["error_category"]
    if cat in taxonomy_counts:
      taxonomy_counts[cat] += 1

  ex_rate = (correct_count / total_queries) * 100.0 if total_queries > 0 else 0.0
  vsr_rate = (
      (valid_sql_count / total_queries) * 100.0 if total_queries > 0 else 0.0
  )
  far_rate = (
      (format_adherent_count / total_queries) * 100.0
      if total_queries > 0
      else 0.0
  )

  hit_limit_count = sum(1 for r in results if r.get("hit_token_limit", False))
  total_gen_tokens = sum(r.get("generated_tokens", 0) for r in results)
  avg_gen_tokens = (total_gen_tokens / total_queries) if total_queries > 0 else 0.0

  summary = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "total_queries": total_queries,
      "metrics": {
          "execution_accuracy_pct": round(ex_rate, 2),
          "valid_sql_rate_pct": round(vsr_rate, 2),
          "format_adherence_rate_pct": round(far_rate, 2),
      },
      "counts": {
          "correct": correct_count,
          "valid_sql": valid_sql_count,
          "format_adherent": format_adherent_count,
      },
      "token_stats": {
          "total_hit_token_limit": hit_limit_count,
          "hit_token_limit_pct": round((hit_limit_count / total_queries) * 100.0, 2) if total_queries > 0 else 0.0,
          "avg_generated_tokens": round(avg_gen_tokens, 2),
      },
      "failure_taxonomy": taxonomy_counts,
      "per_database": per_db_accuracy,
      "detailed_results": results,
  }

  return summary


def print_evaluation_summary(summary: Dict[str, Any]) -> None:
  """Print human-readable evaluation summary table to console."""
  m = summary["metrics"]
  c = summary["counts"]
  tax = summary["failure_taxonomy"]
  total = summary["total_queries"]

  print("\n" + "=" * 60)
  print("EVALUATION RESULTS SUMMARY (Harness v2)")
  print("=" * 60)
  print(f"Total Evaluated Questions: {total}")
  print(f"Precision Setting:         {summary.get('precision', 'unknown')}")
  print(f"Max New Tokens:            {summary.get('max_new_tokens', 512)}")
  if "token_stats" in summary:
    ts = summary["token_stats"]
    print(f"Hit Token Limit:           {ts['total_hit_token_limit']}/{total} ({ts['hit_token_limit_pct']}%)")
    print(f"Avg Generated Tokens:      {ts['avg_generated_tokens']}")
  print(f"Execution Accuracy (EX):   {m['execution_accuracy_pct']:.2f}% ({c['correct']}/{total})")
  print(f"Valid SQL Rate (VSR):       {m['valid_sql_rate_pct']:.2f}% ({c['valid_sql']}/{total})")
  print(f"Format Adherence Rate (FAR):{m['format_adherence_rate_pct']:.2f}% ({c['format_adherent']}/{total})")

  print("\n6-CLASS FAILURE TAXONOMY BREAKDOWN:")
  print("-" * 50)
  for cat, count in tax.items():
    pct = (count / total) * 100.0 if total > 0 else 0.0
    print(f"  {cat:<25} {count:<5} ({pct:.1f}%)")
  print("-" * 50)

  print("\nPER-DATABASE ACCURACY:")
  for db, data in summary["per_database"].items():
    db_pct = (data["correct"] / data["total"]) * 100.0 if data["total"] > 0 else 0.0
    print(f"  {db:<25} {data['correct']}/{data['total']} ({db_pct:.1f}%)")
  print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL Model.")
  parser.add_argument(
      "--model_path",
      type=str,
      required=True,
      help="Hugging Face model ID or local directory.",
  )
  parser.add_argument(
      "--adapter_path",
      type=str,
      default=None,
      help="Path to trained LoRA adapter weights (optional).",
  )
  parser.add_argument(
      "--split",
      type=str,
      default="test",
      choices=["train", "test"],
      help="Dataset split to evaluate.",
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default=None,
      help="Path to data directory containing databases/ and json files.",
  )
  parser.add_argument(
      "--output_file",
      type=str,
      default="docs/baseline_eval.json",
      help="Path to save evaluation JSON artifact.",
  )
  parser.add_argument(
      "--device",
      type=str,
      default=None,
      help="Compute device ('cuda', 'cpu', or auto-detect).",
  )
  parser.add_argument(
      "--max_new_tokens",
      type=int,
      default=512,
      help="Max generation tokens.",
  )
  parser.add_argument(
      "--precision",
      type=str,
      default="float16",
      choices=["float16", "bfloat16", "4bit", "float32"],
      help="Inference precision/quantization setting ('float16', 'bfloat16', '4bit', or 'float32').",
  )
  return parser.parse_args()


def main() -> None:
  import os
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

  args = parse_args()

  # Workaround for Colab pre-installed torchao (<0.16.0) triggering PEFT version mismatch error
  try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
    peft.tuners.lora.torchao.dispatch_torchao = lambda *args, **kwargs: None
  except Exception:
    pass

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer

  if args.device:
    device = args.device
  else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

  print("=" * 60)
  print("STARTING EVALUATION RUN (Harness v2)")
  print(f"Model:          {args.model_path}")
  print(f"Adapter:        {args.adapter_path or 'None (Zero-Shot Base)'}")
  print(f"Split:          {args.split}")
  print(f"Device:         {device}")
  print(f"Precision:      {args.precision}")
  print(f"Max New Tokens: {args.max_new_tokens}")
  print(f"Output File:    {args.output_file}")
  print("=" * 60)

  # Load Dataset
  dataset = load_logistics_dataset(split=args.split, data_dir=args.data_dir)
  print(f"Loaded {len(dataset)} questions for split '{args.split}'.")

  # Load Tokenizer & Model
  print("Loading tokenizer and model...")
  tokenizer = AutoTokenizer.from_pretrained(
      args.model_path, trust_remote_code=True
  )

  if args.precision == "4bit":
    from transformers import BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading base model in 4-bit NF4 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
  else:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.precision, torch.float16 if device == "cuda" else torch.float32)
    print(f"Loading base model in {args.precision} ({torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

  if args.adapter_path:
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter_path)

  if device != "cuda" and args.precision != "4bit":
    model.to(device)

  model.eval()

  summary = run_evaluation(
      model=model,
      tokenizer=tokenizer,
      dataset=dataset,
      device=device,
      max_new_tokens=args.max_new_tokens,
  )

  summary["model_path"] = args.model_path
  summary["adapter_path"] = args.adapter_path
  summary["split"] = args.split
  summary["precision"] = args.precision
  summary["max_new_tokens"] = args.max_new_tokens
  summary["harness_version"] = "v2"

  # Save artifact
  out_path = Path(args.output_file)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

  print(f"Evaluation artifact saved to {out_path}.")
  print_evaluation_summary(summary)


if __name__ == "__main__":
  main()
