# Consolidated Review Report: `PLAN.md`

**Target File:** `c:\Users\Rudra\ml-prep\GRPO with an SQL-execution reward\PLAN.md`  
**Review Preset:** Consolidated report with all review rubrics (Uninterrupted Full Flow)  
**Date:** 2026-11-14  

---

## 1. Executive Summary & Review Stage Pipeline

This consolidated audit was performed across five sequential review stages in accordance with the EGM Goldfish Review Protocol:
1. **Stage 5 (Initial Zoom Out)**: High-altitude architecture, simplicity, and TRIZ contradiction analysis (`egm-zoom-out`, `egm-inventive-catalyst`).
2. **Stage 6 (Comprehension Test)**: Reader self-containment, undefined terminology, and workflow gaps (`egm-comprehension`).
3. **Stage 7 (Critic Review)**: System architecture, infrastructure limits, template compliance, and copy editing on FeatureRange (`egm-design`, `egm-structural`, `egm-style`).
4. **Stage 8 (Intermediate Zoom Out)**: High-altitude complexity audit, schedule feasibility, and anti-overengineering filter on prior findings (`egm-zoom-out`).
5. **Stage 9 (Implementation Readiness Test)**: Zero-Decision Rule audit, actionability, codebase parity, and file boundaries on DetailedRange (`egm-implementation-readiness`, `egm-codebase-details`, `egm-style`).

### Review Status Summary
* **Stage 5 (`zoom_out_report.md`)**: 5 Major, 3 Nit findings recorded.
* **Stage 6 (`comprehension_report.md`)**: Paraphrase confirmed; 3 Major, 2 Nit findings recorded.
* **Stage 7 (`critic_report.md`)**: 2 Blocker, 4 Major, 5 Nit findings recorded.
* **Stage 8 (`intermediate_zoom_out_report.md`)**: 1 Blocker, 5 Major, 2 Nit findings recorded.
* **Stage 9 (`implementation_readiness_report.md`)**: 3 Blocker, 8 Major, 3 Nit findings recorded.

---

## 2. Consolidated Actionable Tasks (Prioritized)

### 🔴 Blockers (Must resolve before execution)

#### ACTION-B1: Right-Size GRPO Rollout Generation & Hyperparameters on Colab T4
* **Severity:** 🔴 Blocker
* **Problem:** Generating $G=4$ completions across an effective batch size of 8 with 192 max tokens across 200 steps on an unquantized 1.5B model on a single free-tier Colab T4 (~20–30 tokens/sec decoding) requires 8.5 to 13 hours of autoregressive decoding alone, blowing past the 4-hour Phase 3 window.
* **Actionable Fix:** 
  1. Cap `max_new_tokens` strictly to 96 (SQL queries on logistics schemas average 30–50 tokens).
  2. Right-size total training steps from 200 to 100 steps (3,200 rollouts is sufficient for initial policy adaptation).
  3. Mandate `torch_dtype=torch.float16` (T4 compute capability 7.5 does not natively support `bfloat16`).
  4. Enforce `GRPOConfig(..., use_vllm=False)` to prevent immediate CUDA OOM on a single 15GB GPU.
  5. Specify LoRA parameters: $r=16, \alpha=32$, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`, `lora_dropout=0.05`.
  6. Reject 4-bit QLoRA to avoid fragile external quantization dependencies on Colab. This brings Phase 3 into a viable ~90–120 minute window on native 16-bit LoRA.

#### ACTION-B2: Document Scope, Goals, and Architecture References via Modularity
* **Severity:** 🔴 Blocker
* **Problem:** The Feature Specification Standard mandates documenting Problem Statement, Goals & Non-Goals, and an Alternatives Considered matrix. However, duplicating multi-page architectural matrices into `PLAN.md` violates document modularity and Anti-Weeds guidelines, as `PLAN.md` is strictly an operational execution plan.
* **Actionable Fix:** 
  1. Add an architectural preface banner at the top of `PLAN.md` linking directly to `README.md` and `docs/DECISIONS.md` (which already details all 12 architectural decisions, trade-offs, and rejected alternatives for Compute, Model, Framework, and Reward).
  2. Add concise structural sections: **Problem Statement** (why standard SFT falls short on logistics Text-to-SQL due to syntax memorization vs schema execution) and **Goals & Non-Goals** (in-scope: LoRA fine-tuning for exact execution match on unseen logistics schemas; out-of-scope: full weight fine-tuning, web UI, multi-turn dialogue).

#### ACTION-B3: Add `src/data_loader.py` and Prompt Formatting Contract
* **Severity:** 🔴 Blocker
* **Problem:** `AGENTS.md` mandates `src/data_loader.py`, but `PLAN.md` omits it from Phase 1 task checklists and provides no prompt formatting template for `Qwen2.5-Coder-1.5B-Instruct`. Implementers cannot write tokenization, evaluation, or training pipelines without making independent design decisions.
* **Actionable Fix:** 
  1. Add `[Create] src/data_loader.py` to the Phase 1 task checklist.
  2. Document the prompt formatting contract:
     ```python
     def format_sql_prompt(schema_ddl: str, question: str) -> str: ...
     ```
  3. Specify system prompt instructions, schema presentation, and markdown SQL code fence delimiters (```` ```sql\n...\n``` ````) adhering to tokenizer chat templates.

#### ACTION-B4: Define Dataset Source and JSON Record Contract
* **Severity:** 🔴 Blocker
* **Problem:** Phase 1 directs downloading/curating 6 logistics databases and ~550 query pairs without defining the upstream source repository, download mechanism, or JSON record schema.
* **Actionable Fix:** 
  1. Specify the curation source: a lightweight extraction script `[Create] scripts/prepare_data.py` that isolates logistics schemas (`warehouse_management`, `store_inventory`, `customers_and_invoices`, `products_catalog`, `shipping_and_containers`, `freight_tracking`) from Spider or synthesized fixtures.
  2. Document the exact JSON record schema:
     ```json
     {
       "db_id": "warehouse_management",
       "question": "What is the total quantity of items in warehouse W1?",
       "gold_sql": "SELECT SUM(quantity) FROM inventory WHERE warehouse_id = 'W1';",
       "difficulty": "medium"
     }
     ```

#### ACTION-B5: Scope and Define Deterministic Failure Taxonomy in `src/evaluate.py`
* **Severity:** 🔴 Blocker
* **Problem:** `PLAN.md` mandates that `src/evaluate.py` automatically classify all query failures into an 8-class taxonomy in a 3-hour Saturday morning window. Programmatically distinguishing semantic intent errors (`JOIN_ERROR`, `CONDITION_ERROR`, `AGGREGATION_ERROR`, `VALUE_MISMATCH`) on queries that ran without runtime errors requires a complex SQL AST diff engine, creating an unrealistic scope trap.
* **Actionable Fix:** 
  1. Scope automated evaluation in `src/evaluate.py` to 5 deterministic execution categories: `SYNTAX_ERROR`, `SCHEMA_HALLUCINATION`, `TIMEOUT_ABORT`, `EMPTY_RESULT`, and `VALUE_MISMATCH`.
  2. Document the classification waterfall in Phase 2.
  3. Reserve fine-grained relational root-cause diagnoses (`JOIN_ERROR`, `CONDITION_ERROR`, `AGGREGATION_ERROR`) for the 3 qualitative case studies in Phase 4.

---

### 🟡 Major (Important architectural and operational improvements)

#### ACTION-M1: Tag All File Boundaries with Explicit Action Tags (`[Create]`, `[Modify]`, `[Delete]`)
* **Problem:** File references across all phases lack operation tags, violating the Implementation Readiness Standard's Target Scope & File Boundary rule.
* **Actionable Fix:** Prefix every file reference across Phase 1 through Phase 6 with `[Create]`, `[Modify]`, or `[Delete]` (e.g., `[Create] src/executor.py`, `[Modify] docs/RESULTS.md`).

#### ACTION-M2: Define API Signatures & Sandboxing Architecture in `src/executor.py` and `src/reward.py`
* **Problem:** Key function contracts are missing, sandboxing has redundant layers, and anti-hacking logic is ambiguous.
* **Actionable Fix:** 
  1. Document explicit signatures:
     ```python
     class SafeSQLExecutor:
         def execute(self, db_path: str, sql: str, timeout: float = 1.0) -> Tuple[bool, List[Tuple], Optional[str]]: ...

     def compute_sql_reward(predicted_sql: str, gold_sql: str, db_path: str) -> float: ...
     ```
  2. Keep sandboxing in pure Python standard library: use `sqlite3` in-memory loading (`:memory:`) with `set_authorizer` and `threading.Timer` with `conn.interrupt()`. Reject complex multiprocessing pools or custom AST parsers.
  3. Enforce simple anti-hacking rule: return 0.0 match credit if $V_{\text{pred}} = []$ against non-empty $V_{\text{gold}}$, or if candidate query lacks table references. Reject complex AST Jaccard reward shaping.

#### ACTION-M3: Specify Test Plan Assertions & Shared Test Infrastructure
* **Problem:** Unit tests are listed as bare filenames without test scenarios or shared test fixtures.
* **Actionable Fix:** 
  1. Add `[Create] tests/conftest.py` with shared in-memory SQLite fixtures and sample logistics schemas.
  2. Specify required test assertions:
     - `test_executor.py`: (1) Valid SELECT returns rows; (2) Blocked commands (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `DETACH`, `PRAGMA`, `load_extension`, `WRITEFILE`) raise security error; (3) Runaway recursive query terminates within 1.0s; (4) Truncates at 100 rows.
     - `test_reward.py`: (1) Exact multiset match scores 1.0; (2) Format adherence with syntax error scores 0.1; (3) Clean execution with wrong rows scores 0.3; (4) Unordered row equivalence scores 1.0; (5) Empty result against non-empty gold scores 0.0.
     - `test_leakage.py`: Asserts `set(train_db_ids).isdisjoint(set(test_db_ids))`.

#### ACTION-M4: Define TRL `GRPOTrainer` Batch Reward Callable Interface & Dataset Columns
* **Problem:** Missing integration contracts between `GRPOTrainer` and `compute_sql_reward`.
* **Actionable Fix:** Document the dataset columns fed into `GRPOTrainer` (`prompt`, `gold_sql`, `db_path`), rollout exploration sampling parameters (`temperature=0.7`, `top_p=0.9`), and the batch reward wrapper signature:
  ```python
  def grpo_sql_reward_func(completions: List[str], gold_sql: List[str], db_path: List[str], **kwargs) -> List[float]: ...
  ```

#### ACTION-M5: Add Git Synchronization Checkpoint to Phase 1
* **Problem:** Phase 2 starts by cloning the repository in Colab, but git push was scheduled only in Phase 6, leaving Colab unable to access the code.
* **Actionable Fix:** Explicitly add a task at the end of Phase 1 to commit and push local changes to GitHub (`git add .`, `git commit`, `git push origin main`) before starting Phase 2.

#### ACTION-M6: Mitigate Colab Preemption Risks with Compute Failover & Checkpoint Recovery
* **Problem:** Chaining Phase 2 and Phase 3 sequentially on Colab free-tier creates a single point of failure without backup infrastructure.
* **Actionable Fix:** 
  1. Specify backup compute options (Kaggle Notebooks free dual T4 GPUs or RunPod spot instance).
  2. Mandate checkpointing LoRA adapters directly to Google Drive every 25–50 steps.
  3. Include an intermediate checkpoint fallback: evaluate the latest healthy intermediate checkpoint (step 50 or 100) before declaring total training failure.

#### ACTION-M7: Align Cut-Line Decision Trigger to Saturday 10:00 PM IST
* **Problem:** Placing the Cut-Line at Sunday 12:00 PM introduces a 15-hour vacuum after Phase 4 completes on Saturday 9:00 PM.
* **Actionable Fix:** Move the Cut-Line trigger immediately to the end of Phase 4 (Saturday 15 Nov, 10:00 PM IST), allowing overnight planning and enabling an immediate Sunday morning pivot if blocked.

#### ACTION-M8: Calibrate Risk Claims, Strip Emotive Phrasing, and Remove Subjective Criteria
* **Problem:** Text contains uncalibrated capability claims ("ensuring gradients", "hanging GPU"), subjective operator-state criteria ("fatigue sets in"), and emotive qualifiers ("Honest Fallback", "no fake numbers").
* **Actionable Fix:** 
  1. Calibrate Line 159 to *"providing partial reward signals to promote non-zero gradient variance"*.
  2. Calibrate Line 160 to *"Runaway cartesian joins blocking rollout execution thread"*.
  3. Replace Phase 5 Line 129 with *"If elapsed time exceeds schedule allocation, omit Phase 5."*
  4. Replace `"Honest Fallback"` with `"Fallback Protocol"`, replace `"Real Resume Bullets"` with `"Empirical Metric Summary"`, drop redundant Line 119 (`"4. Never fabricate numbers"`), and calibrate Line 172 to *"empirical evaluation metrics"*.

---

### 🟢 Nits (Refinement and formatting polish)

#### ACTION-N1: Fix AST Blocklist Discrepancy with `AGENTS.md`
* **Actionable Fix:** Update line 47 blocked command list to include `ALTER`, `DETACH`, and `WRITEFILE` as mandated by `AGENTS.md` Rule 3.

#### ACTION-N2: Document CLI Arguments & JSON Artifact Protocol for `src/evaluate.py`
* **Actionable Fix:** Document exact CLI invocation (`--model_path`, `--adapter_path`, `--split`, `--output_file`) and specify that evaluation results are saved to JSON artifacts (`docs/baseline_eval.json`, `docs/grpo_eval.json`) before transcribing into `docs/RESULTS.md`.

#### ACTION-N3: Style, Formatting, and Syntax Polish
* **Actionable Fix:** 
  1. Add language specifier to milestone diagram (` ```text `).
  2. Simplify document title to `# Project Execution Plan`.
  3. Expand acronym on first mention: `Group Relative Policy Optimization (GRPO)`.
  4. Change `"Tonight"` to `Fri 14 Nov: 9:00 PM - 11:45 PM IST` and `"Safe Sandbox"` to `"SQLite Sandbox"`.
  5. Remove wordy intensifiers (`"exact same"` ➔ `"same"`, `"in-memory ephemeral"` ➔ `"in-memory"`).

---

## 3. Traceability Matrix

| Raw Finding ID | Review Stage & Reviewer Lens | Severity | Consolidated Action Item |
|---|---|---|---|
| `ZO-5-1` | Stage 5: `egm-zoom-out` | 🟡 Major | **ACTION-B5** (Deterministic Failure Taxonomy) |
| `ZO-5-2` | Stage 5: `egm-inventive-catalyst` | 🟡 Major | **ACTION-B1** & **ACTION-M2** (Execution Bottleneck & In-memory Pre-warming) |
| `ZO-5-3` | Stage 5: `egm-inventive-catalyst` | 🟡 Major | **ACTION-M2** (Reject AST Reward Shaping) |
| `ZO-5-4` | Stage 5: `egm-zoom-out` | 🟡 Major | **ACTION-M2** (Simplify Sandboxing to stdlib) |
| `ZO-5-5` | Stage 5: `egm-zoom-out` | 🟡 Major | **ACTION-M2** (Anti-Hacking Empty Set Scoring) |
| `ZO-5-6` | Stage 5: `egm-inventive-catalyst` | 🟢 Nit | **ACTION-M2** (Reject ProcessPool, use `set_authorizer` / `interrupt`) |
| `ZO-5-7` | Stage 5: `egm-inventive-catalyst` | 🟢 Nit | **ACTION-M6** (Intermediate Checkpoint Recovery) |
| `ZO-5-8` | Stage 5: `egm-zoom-out` | 🟢 Nit | **ACTION-M2** (Keep `train_grpo.py` lean, omit premature CLI flags) |
| `COMP-6-1` | Stage 6: `egm-comprehension` | 🟡 Major | **ACTION-B5** (Enumerate Taxonomy Categories) |
| `COMP-6-2` | Stage 6: `egm-comprehension` | 🟡 Major | **ACTION-B4** (Define Curation Source & JSON Schema) |
| `COMP-6-3` | Stage 6: `egm-comprehension` | 🟡 Major | **ACTION-M5** (Add Git Sync Before Colab) |
| `COMP-6-4` | Stage 6: `egm-comprehension` | 🟢 Nit | **ACTION-M2** (Define Trivial Query Rejection) |
| `COMP-6-5` | Stage 6: `egm-comprehension` | 🟢 Nit | **ACTION-N3** (Expand GRPO Acronym) |
| `CRIT-7-1` | Stage 7: `egm-design`, `egm-structural` | 🔴 Blocker | **ACTION-B2** (Architecture Prefaces & Modularity Links) |
| `CRIT-7-2` | Stage 7: `egm-design` | 🔴 Blocker | **ACTION-B1** (Right-Size Tokens & Step Budget) |
| `CRIT-7-3` | Stage 7: `egm-structural` | 🟡 Major | **ACTION-B2** (Section Ordering: Preface before Timeline) |
| `CRIT-7-4` | Stage 7: `egm-design` | 🟡 Major | **ACTION-M6** (Colab Quota Failover & Persistent Checkpoints) |
| `CRIT-7-5` | Stage 7: `egm-design` | 🟡 Major | **ACTION-M7** (Align Cut-Line to Sat 10:00 PM) |
| `CRIT-7-6` | Stage 7: `egm-style` | 🟡 Major | **ACTION-M8** (Calibrate "Honest" to "Empirical") |
| `CRIT-7-7` | Stage 7: `egm-style` | 🟢 Nit | **ACTION-N3** (Simplify Title) |
| `CRIT-7-8` | Stage 7: `egm-style` | 🟢 Nit | **ACTION-N3** (Fix Temporal Inconsistency in Milestone 1) |
| `CRIT-7-9` | Stage 7: `egm-style` | 🟢 Nit | **ACTION-M8** & **ACTION-N3** (Colloquial Milestone Terminology) |
| `CRIT-7-10` | Stage 7: `egm-structural` | 🟢 Nit | **ACTION-N3** (Code Fence Language Tag) |
| `CRIT-7-11` | Stage 7: `egm-design` | 🟢 Nit | **ACTION-M2** (Cross-Platform Path Handling) |
| `IZO-8-1` | Stage 8: `egm-zoom-out` | 🔴 Blocker | **ACTION-B1** (Rollout Token Ceiling & Step Budget) |
| `IZO-8-2` | Stage 8: `egm-zoom-out` | 🟡 Major | **ACTION-B2** (Reject Architecture Matrix Duplication, Link Modularity) |
| `IZO-8-3` | Stage 8: `egm-zoom-out` | 🟡 Major | **ACTION-M2** (Reject AST Reward Shaping) |
| `IZO-8-4` | Stage 8: `egm-zoom-out` | 🟡 Major | **ACTION-M2** (Reject ProcessPool Subprocesses) |
| `IZO-8-5` | Stage 8: `egm-zoom-out` | 🟡 Major | **ACTION-B5** (5-Category Deterministic Evaluation Taxonomy) |
| `IZO-8-6` | Stage 8: `egm-zoom-out` | 🟡 Major | **ACTION-B4** (Specify Data Provisioning Path) |
| `IZO-8-7` | Stage 8: `egm-zoom-out` | 🟢 Nit | **ACTION-M5** (Git Sync Before Colab) |
| `IZO-8-8` | Stage 8: `egm-zoom-out` | 🟢 Nit | **ACTION-M7** (Cut-Line Schedule Timing) |
| `READ-9-1` | Stage 9: `egm-implementation-readiness` | 🔴 Blocker | **ACTION-B3** (Omission of `src/data_loader.py` & Prompt Template) |
| `READ-9-2` | Stage 9: `egm-implementation-readiness` | 🔴 Blocker | **ACTION-B4** (Missing Dataset Curation Source & JSON Contract) |
| `READ-9-3` | Stage 9: `egm-implementation-readiness` | 🔴 Blocker | **ACTION-B5** (Underspecified Failure Taxonomy Logic) |
| `READ-9-4` | Stage 9: `egm-codebase-details` | 🟡 Major | **ACTION-M1** (Missing Action Tags on File Boundaries) |
| `READ-9-5` | Stage 9: `egm-implementation-readiness` | 🟡 Major | **ACTION-M2** (API Signatures for Executor & Reward) |
| `READ-9-6` | Stage 9: `egm-implementation-readiness` | 🟡 Major | **ACTION-M3** (Test Plan Assertions & `conftest.py`) |
| `READ-9-7` | Stage 9: `egm-implementation-readiness` | 🟡 Major | **ACTION-B1** & **ACTION-M4** (Colab T4 Hardware & Training Configuration) |
| `READ-9-8` | Stage 9: `egm-implementation-readiness` | 🟡 Major | **ACTION-M4** (TRL `GRPOTrainer` Reward Wrapper & Dataset Columns) |
| `READ-9-9` | Stage 9: `egm-style` | 🟡 Major | **ACTION-M8** (Calibrate Gradient & Host CPU Execution Claims) |
| `READ-9-10` | Stage 9: `egm-style` | 🟡 Major | **ACTION-M8** (Objective Schedule Criteria in Phase 5) |
| `READ-9-11` | Stage 9: `egm-style` | 🟡 Major | **ACTION-M8** (Strip Emotive Qualifiers & Defensive Caveats) |
| `READ-9-12` | Stage 9: `egm-codebase-details` | 🟢 Nit | **ACTION-N1** (AST Blocklist Discrepancy with `AGENTS.md`) |
| `READ-9-13` | Stage 9: `egm-implementation-readiness` | 🟢 Nit | **ACTION-N2** (CLI Arguments & Evaluation JSON Artifacts) |
| `READ-9-14` | Stage 9: `egm-style` | 🟢 Nit | **ACTION-N3** (Synonym Redundancy & Wordy Intensifiers) |

---

## 4. Deliverable Files Summary

The following report artifacts have been written to the repository root:
* [zoom_out_report.md](file:///c:/Users/Rudra/ml-prep/GRPO%20with%20an%20SQL-execution%20reward/zoom_out_report.md): Stage 5 Initial Zoom Out & TRIZ Contradictions.
* [comprehension_report.md](file:///c:/Users/Rudra/ml-prep/GRPO%20with%20an%20SQL-execution%20reward/comprehension_report.md): Stage 6 Comprehension Test & Reader Paraphrase.
* [critic_report.md](file:///c:/Users/Rudra/ml-prep/GRPO%20with%20an%20SQL-execution%20reward/critic_report.md): Stage 7 Hostile Design, Structural, and Style Review on FeatureRange.
* [intermediate_zoom_out_report.md](file:///c:/Users/Rudra/ml-prep/GRPO%20with%20an%20SQL-execution%20reward/intermediate_zoom_out_report.md): Stage 8 Intermediate Zoom Out & Anti-Overengineering Audit.
* [implementation_readiness_report.md](file:///c:/Users/Rudra/ml-prep/GRPO%20with%20an%20SQL-execution%20reward/implementation_readiness_report.md): Stage 9 Implementation Readiness & Codebase Details on DetailedRange.
* [consolidated_review_report.md](file:///c:/Users/Rudra/ml-prep/GRPO%20with%20an%20SQL-execution%20reward/consolidated_review_report.md): Master deliverable synthesizing all 45 findings into 16 actionable tasks with a full traceability matrix.
