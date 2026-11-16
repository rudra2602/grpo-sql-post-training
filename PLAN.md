# Project Execution Plan

**Target Goal:** Post-train `Qwen2.5-Coder-1.5B-Instruct` with Group Relative Policy Optimization (GRPO) using SQLite verifiable execution rewards on logistics schemas, with empirical baseline comparison and failure analysis.  
**Hard Deadline:** Tuesday, 17 Nov 2026, 9:00 AM IST.  
**Status:** **100% COMPLETE & VERIFIED** (Completed Sunday 15 Nov 2026, ~36 hours ahead of deadline).

---

## 1. Timeline & Milestones Status

```text
[COMPLETED] Sun 15 Nov, 7:00 AM - 10:30 AM IST
  │
  ├─► Phase 1: Local Setup, Data Curation, Sandbox, Conftest & Unit Tests 
  │            Status: PASSED (39/39 unit tests passing in 1.11s)
  │
  ├─► Phase 2: Baseline Zero-Shot Evaluation on 108 Test Pairs & 5-Class Error Taxonomy
  │            Status: PASSED (EX: 13.89%, VSR: 19.44%, FAR: 19.44%)
  │
  ├─► Phase 3: Pipeline Smoke Tests & Primary 1.5B GRPO Training (480 policy rollouts, 30 steps)
  │            Status: PASSED (Trained in 34.5 min on local RTX 4050 GPU with 4-bit QLoRA)
  │
  ├─► Phase 4: Post-Training Evaluation on 108 Test Pairs & 3 Qualitative Case Studies
  │            Status: PASSED (EX: 16.67% [+20.01% rel], VSR: 24.07% [+23.82% rel] - Frozen as v1)
  │
  ├─► Phase 5: Evaluation Harness v2 & Scientific Audit (512 tokens, float16, exact McNemar test)
  │            Status: PASSED (Base EX: 58.33% [63/108], GRPO EX: 60.19% [65/108], +2 queries, p=0.5000)
  │
  ├─► Phase 6: Documentation Overhaul & Statistical Integrity Audit
  │            Status: PASSED (RESULTS.md, EXPERIMENT_LOG.md, EVAL_PROTOCOL.md, README.md updated)
  │
[AHEAD OF DEADLINE] Ready for internship submission.
```

---

## 2. Phase Task Checklists

### **Phase 1: Environment, Safe Sandbox, Reward Engine & Unit Tests**
- [x] Create clean conda environment `grpo-sql` with Python 3.12 and PyTorch CUDA.
- [x] `scripts/prepare_data.py`:
  - Downloads 10 verified Spider SQLite databases and query pairs.
  - Filters out queries returning 0 rows or syntax errors (retaining only queries returning $\ge 1$ rows).
  - Generates `data/train_logistics.json` (406 pairs across 8 schemas).
  - Generates `data/test_logistics.json` (108 pairs across 2 held-out schemas: `tracking_orders`, `ship_1`).
- [x] `src/data_loader.py`:
  - Implements `format_sql_prompt(schema_ddl, question)`.
  - Formats prompt with system instructions, SQLite DDL schema, and question.
- [x] `src/executor.py` (`SafeSQLExecutor`):
  - In-memory database loading (`:memory:`).
  - `set_authorizer` callback blocking 24 mutating action codes.
  - Dual 1.0s timeout mechanism via `threading.Timer` with `conn.interrupt()` and `set_progress_handler`.
  - Truncation at 100 rows (`cursor.fetchmany(100)`).
- [x] `src/reward.py` (`compute_sql_reward`):
  - Tiered composite reward (+0.1 format, +0.2 syntax, +0.7 execution match).
  - Multiset row equivalence with strict `ORDER BY` enforcement.
  - Anti-hacking logic for trivial scalar and empty-set protection.
- [x] `tests/conftest.py`, `tests/test_executor.py`, `tests/test_reward.py`, `tests/test_leakage.py`:
  - All 39 unit tests pass 100% in 1.11s.
- [x] Git committed and pushed to GitHub.

---

### **Phase 2: Baseline Zero-Shot Evaluation**
- [x] `src/evaluate.py`:
  - Evaluates greedy decoding ($T=0.0$) on `data/test_logistics.json` (108 pairs).
  - Computes Execution Accuracy (EX), Valid SQL Rate (VSR), Format Adherence Rate (FAR).
  - 5-class deterministic taxonomy (`SYNTAX_ERROR`, `SCHEMA_HALLUCINATION`, `TIMEOUT_ABORT`, `EMPTY_RESULT`, `VALUE_MISMATCH`).
- [x] Baseline evaluation completed:
  - Execution Accuracy: **13.89 %** (15/108)
  - Valid SQL Rate: **19.44 %** (21/108)
  - Format Adherence: **19.44 %** (21/108)
  - Failures: 87 `SYNTAX_ERROR` (80.6%), 6 `VALUE_MISMATCH` (5.6%).

---

### **Phase 3: Pipeline Smoke Test & Primary GRPO Training**
- [x] `src/train_grpo.py`:
  - 4-bit NF4 QLoRA (`bitsandbytes`) with bfloat16 compute for 6GB consumer GPU efficiency.
  - Chunked logprob tile size reduced to 512 tokens (64x memory savings).
  - `SafeGRPOTrainer` with prompt-by-prompt rollout generation.
  - DDL schema sanitization capping sequence lengths to <987 tokens (138x attention memory savings).
- [x] **0.5B Smoke Test:** 5 steps in 113s; adapter saved to `checkpoints/grpo_smoke_test`.
- [x] **1.5B Feasibility Dry Run:** 5 steps in 329s; adapter saved to `checkpoints/grpo_1.5b_test`.
- [x] **Primary GRPO Training Run:** `Qwen2.5-Coder-1.5B-Instruct`
  - 30 steps, group size $G=4$, effective batch size 4 prompts (16 completions/step, 480 policy rollouts total).
  - Completed in 2,072 seconds (~34.5 minutes) on NVIDIA RTX 4050 GPU.
  - Checkpoints persisted to `checkpoints/grpo_final/` (`checkpoint-10`, `checkpoint-20`, `checkpoint-30`, final adapter).

---

### **Phase 4: Post-Training Evaluation & Error Taxonomy Analysis**
- [x] Evaluated post-trained model on 108 test questions ($T=0.0$):
  - Execution Accuracy (EX): **16.67 %** (18/108) -> **+2.78 % absolute gain (+20.01% relative)**.
  - Valid SQL Rate (VSR): **24.07 %** (26/108) -> **+4.63 % absolute gain (+23.82% relative)**.
  - Format Adherence (FAR): **24.07 %** (26/108) -> **+4.63 % absolute gain (+23.82% relative)**.
- [x] Failure Taxonomy Shift:
  - `SYNTAX_ERROR` reduced from 87 down to 82 (-5 syntax errors).
  - `VALUE_MISMATCH` increased from 6 to 8 (due to queries newly executing cleanly with inverted projection order).
- [x] 3 Qualitative Case Studies documented:
  1. *Eliminating Conversational Rambling (Index 40)*
  2. *Correlated Subquery vs. Explicit Join (Index 64)*
  3. *Multi-Table Join with Group-By and Having (Index 100)*

---

### **Phase 6: Final Documentation, Resume Bullets & Repo Finalization**
- [x] `docs/RESULTS.md`: Populated with all verified measured numbers and qualitative case studies.
- [x] `docs/EXPERIMENT_LOG.md`: Master run log with all training details and hyperparameter tables.
- [x] `docs/LEARNING_NOTES.md`: Technical deep-dive handbook and 30-second elevator pitch for DP World interview.
- [x] `README.md`: Updated with benchmark tables, architecture diagram, directory map, and quickstart commands.
- [x] All 39 unit tests verified passing.
