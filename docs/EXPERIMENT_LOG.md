# Experiment Tracking Log

> **Strict Honesty Rule:** All cells in this document record real measured values from completed runs. Never write placeholder numbers or estimated values.

---

## 1. Master Experiment Summary Table

| Run ID | Date & Time | Model Name | Training Method | Group Size ($G$) | LR / Schedule | Steps | Precision | Hardware | Runtime (min) | Test EX (%) | Test VSR (%) | Status / Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `RUN-00-BASE` | 2026-11-15 | `Qwen2.5-Coder-1.5B-Instruct` | Zero-Shot Greedy ($T=0.0$) | N/A | None | 0 | 16-bit | Colab T4 | 8.0 min | 13.89 % | 19.44 % | Frozen v1 eval; 128-token wall caused truncation artifacts; JSON uncommitted |
| `RUN-01-SMOKE` | 2026-11-15 | `Qwen2.5-Coder-0.5B-Instruct` | GRPO (16-bit LoRA) | 4 | 1e-5 / Constant | 5 | 16-bit | Laptop RTX 4050 | 1.88 min | N/A | N/A | Pipeline smoke test; verified reward math & rollouts |
| `RUN-02-DRY5` | 2026-11-15 | `Qwen2.5-Coder-1.5B-Instruct` | GRPO (4-bit QLoRA) | 4 | 1e-5 / Constant | 5 | 4-bit NF4 | Laptop RTX 4050 | 5.48 min | 15.74 % | 22.22 % | 5-step test adapter; frozen as `docs/eval_v1_5step.json` |
| `RUN-03-PRIMARY`| 2026-11-15 | `Qwen2.5-Coder-1.5B-Instruct` | GRPO (Tiered Composite) | 4 | 1e-5 / Linear | 30 | 4-bit NF4 | Laptop RTX 4050 | 34.53 min | 16.67 % | 24.07 % | Primary 30-step run; frozen as `docs/eval_v1_grpo.json` |
| `RUN-00-BASE-V2`| 2026-11-16 | `Qwen2.5-Coder-1.5B-Instruct` | Zero-Shot Greedy ($T=0.0$) | N/A | None | 0 | `float16` | Laptop RTX 4050 | 12.0 min | **58.33 %** | **87.04 %** | **Harness v2 (512 tokens); saved in `docs/eval_v2_base.json`** |
| `RUN-03-PRIMARY-V2`| 2026-11-16 | `Qwen2.5-Coder-1.5B-Instruct` | GRPO LoRA Adapter | N/A | None | 0 | `float16` | Laptop RTX 4050 | 12.5 min | **60.19 %** | **87.04 %** | **Harness v2 (512 tokens); +1.85% (63->65); `docs/eval_v2_grpo.json`** |

---

## 2. Detailed Run Sheets

### Run `RUN-00-BASE` (Baseline Zero-Shot)
* **Execution Date:** 2026-11-15 12:25 IST
* **Command:** `python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --split test --output_file docs/baseline_eval.json`
* **Decoding:** Temperature = 0.0, Greedy, max_tokens = 128
* **Artifact Status:** `docs/baseline_eval.json` was generated in Google Colab session but was **NEVER committed to git** (file is missing in the repository). Baseline numbers below are transcribed from Colab cell outputs.
* **Measured Metrics:**
  * Execution Accuracy (EX): **13.89 %** (15/108)
  * Valid SQL Rate (VSR): **19.44 %** (21/108)
  * Format Adherence Rate (FAR): **19.44 %** (21/108)
* **Top Failure Modes Diagnosed:**
  1. `SYNTAX_ERROR`: **87** / 93 failures (80.6% of test set) — unclosed markdown tags, broken SQL syntax, conversational preambles
  2. `VALUE_MISMATCH`: **6** / 93 failures (5.6% of test set) — executed cleanly but returned wrong rows
  3. `SCHEMA_HALLUCINATION`, `TIMEOUT_ABORT`, `EMPTY_RESULT`: 0 occurrences under greedy decoding

---

### Run `RUN-01-SMOKE` (Dry-Run / Smoke Test on 0.5B)
* **Execution Date:** 2026-11-15 19:35 IST
* **Command:** `python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --max_steps 5 --group_size 4 --output_dir checkpoints/grpo_smoke_test`
* **Objective:** Verify loss computation, reward function callable, advantage normalization, and checkpoint export without crashing on local hardware.
* **Peak VRAM:** 3.1 GB
* **Average Step Time:** 22.6 s
* **Outcome:** Completed 5 steps in 113s; adapter saved to `checkpoints/grpo_smoke_test`.

---

### Run `RUN-02-DRY5` (5-Step Feasibility Run on 1.5B)
* **Execution Date:** 2026-11-15 20:15 IST
* **Command:** `python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --max_steps 5 --group_size 4 --output_dir checkpoints/grpo_1.5b_test`
* **Objective:** Verify 4-bit NF4 QLoRA on 1.5B parameter model with chunked logprob tile size 512.
* **Peak VRAM:** 5.1 GB
* **Total Runtime:** 329s (5.48 min)
* **Evaluation (`docs/eval_v1_5step.json`, renamed from `docs/test_5step_eval.json`):**
  * Execution Accuracy: **15.74 %** (17/108) (+1.85% over base)
  * Valid SQL Rate: **22.22 %** (24/108) (+2.78% over base)

---

### Run `RUN-03-PRIMARY` (Primary GRPO Post-Training Run)
* **Execution Date:** 2026-11-15 21:03–21:38 IST
* **Command:** `python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --max_steps 30 --group_size 4 --lr 1e-5 --save_steps 10 --output_dir checkpoints/grpo_final`
* **Hyperparameters:**
  * Base Model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
  * LoRA: $r=16, \alpha=32$, dropout=0.05, target: `["q_proj", "k_proj", "v_proj", "o_proj"]`
  * Precision: 4-bit NF4 double quant (`bitsandbytes`), compute dtype: bfloat16
  * Optimizer: AdamW, LR = $1\times 10^{-5}$, linear decay
  * KL Beta: $0.005$
  * Group Size: $G=4$ completions per prompt
  * Micro-Batch: 1 prompt $\times$ 4 gradient accumulation steps = 4 prompts (16 completions/step)
  * Total Policy Rollouts: 480 completions over 30 steps (~1.2 full epochs of 406 logistics pairs)
* **Training Dynamics:**
  * Total Runtime: 2,072 seconds (34.53 minutes)
  * Average Step Speed: ~69.0s per step
  * Checkpoints Persisted: `checkpoint-10`, `checkpoint-20`, `checkpoint-30`, and final adapter in `checkpoints/grpo_final/`
* **Evaluation Results (Test Set, 108 Held-Out Questions, Greedy $T=0.0$, saved in `docs/eval_v1_grpo.json`, renamed from `docs/grpo_eval.json`):**
  * Execution Accuracy (EX): **16.67 %** (18/108) -> **+2.78 % absolute** (+20.01% relative)
  * Valid SQL Rate (VSR): **24.07 %** (26/108) -> **+4.63 % absolute** (+23.82% relative)
  * Format Adherence (FAR): **24.07 %** (26/108) -> **+4.63 % absolute** (+23.82% relative)
  * `ship_1` (Maritime): **27.08 %** (13/48) vs. 22.92% base
  * `tracking_orders` (Fulfillment): **8.33 %** (5/60) vs. 6.67% base
  * `SYNTAX_ERROR` reduction: **87 -> 82** (-5 syntax errors)

---

### Run `RUN-00-BASE-V2` (Base Model Re-Evaluation under Harness v2)
* **Execution Date:** 2026-11-16 19:28 IST
* **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, Windows 11)
* **Precision:** 16-bit (`float16`)
* **Command:** `python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --split test --precision float16 --max_new_tokens 512 --output_file docs/eval_v2_base.json`
* **Artifact:** `docs/eval_v2_base.json` (committed and verified)
* **Measured Metrics:**
  * Execution Accuracy (EX): **58.33 %** (63 / 108) [Wilson 95% CI: 48.9%, 67.2%]
  * Valid SQL Rate (VSR): **87.04 %** (94 / 108) [Wilson 95% CI: 79.4%, 92.1%]
  * Format Adherence Rate (FAR): **100.00 %** (108 / 108) [Wilson 95% CI: 96.6%, 100.0%]
  * Average Completion Tokens: 246.0 tokens
  * Hit Token Limit ($\ge 512$): 1 / 108 (0.93%)
* **Taxonomy Breakdown:**
  * `NO_SQL_EXTRACTED`: 0 (0.0%)
  * `SYNTAX_ERROR`: 1 (0.9%)
  * `SCHEMA_HALLUCINATION`: 13 (12.0%)
  * `TIMEOUT_ABORT`: 0 (0.0%)
  * `EMPTY_RESULT`: 8 (7.4%)
  * `VALUE_MISMATCH`: 23 (21.3%)
* **Per-Database Accuracy:**
  * `ship_1`: **50.00 %** (24 / 48)
  * `tracking_orders`: **65.00 %** (39 / 60)

---

### Run `RUN-03-PRIMARY-V2` (GRPO Model Re-Evaluation under Harness v2)
* **Execution Date:** 2026-11-16 20:15 IST
* **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, Windows 11)
* **Precision:** 16-bit (`float16`)
* **Command:** `python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter_path checkpoints/grpo_final --split test --precision float16 --max_new_tokens 512 --output_file docs/eval_v2_grpo.json`
* **Artifacts:** `docs/eval_v2_grpo.json` and `docs/comparison_v2.json` (committed and verified)
* **Measured Metrics:**
  * Execution Accuracy (EX): **60.19 %** (65 / 108) [Wilson 95% CI: 50.8%, 68.9%]
  * Valid SQL Rate (VSR): **87.04 %** (94 / 108) [Wilson 95% CI: 79.4%, 92.1%]
  * Format Adherence Rate (FAR): **100.00 %** (108 / 108) [Wilson 95% CI: 96.6%, 100.0%]
  * Average Completion Tokens: 238.9 tokens
  * Hit Token Limit ($\ge 512$): 2 / 108 (1.85%)
* **Taxonomy Breakdown:**
  * `NO_SQL_EXTRACTED`: 0 (0.0%)
  * `SYNTAX_ERROR`: 1 (0.9%)
  * `SCHEMA_HALLUCINATION`: 13 (12.0%)
  * `TIMEOUT_ABORT`: 0 (0.0%)
  * `EMPTY_RESULT`: 6 (5.6%) (-2 reduction)
  * `VALUE_MISMATCH`: 23 (21.3%)
* **Per-Database Accuracy:**
  * `ship_1`: **54.17 %** (26 / 48) (+2 queries / +4.17% over base)
  * `tracking_orders`: **65.00 %** (39 / 60) (0 delta / exact parity)
* **Paired Statistical Comparison:**
  * Both correct: 63 | Only base correct: 0 | Only GRPO correct: 2 | Neither correct: 43
  * Net Shift: +2 queries (+1.85% absolute)
  * Exact McNemar Test: $p = 0.5000$ (statistically non-significant; within expected noise)
