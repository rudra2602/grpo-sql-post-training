# Implementation Readiness Review Report (Stage 9)

**Target:** `PLAN.md` (DetailedRange, Lines 35–173)  
**Standard:** `resources/document_standards/implementation-readiness-standard.md`  
**Reviewers:** `egm-implementation-readiness` (Pragmatic Tech Lead), `egm-codebase-details` (Technical Quality Auditor), `egm-style` (Technical Copy Editor)  
**Date:** 2026-11-14  

---

## Reviewer Status

| Reviewer | Lens | Status |
|---|---|---|
| `egm-implementation-readiness` | Zero-Decision Rule, Actionability, Execution Details | Completed |
| `egm-codebase-details` | File Boundaries, Codebase Parity, Test Infrastructure | Completed |
| `egm-style` | Precision, Conciseness, Calibration, Objective Tone | Completed |

---

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. Omission of `src/data_loader.py` and Prompt Formatting Contract
* **Severity:** 🔴 Blocker
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** Phase 1 (Lines 40–56), Phase 3 (Lines 80–89); `AGENTS.md` Line 166
* **Hurdle:** `AGENTS.md` Section 5 defines `src/data_loader.py` as a primary component responsible for formatting prompts with SQLite DDL schemas and user questions. `PLAN.md` omits `src/data_loader.py` entirely from the Phase 1 checklist and does not define the prompt template for `Qwen2.5-Coder-1.5B-Instruct` (system instruction, DDL schema presentation, user turn formatting, code fence delimiters). An implementer cannot write tokenization, evaluation, or training pipelines without making independent design decisions.
* **Remediation:** Add `[Create] src/data_loader.py` to the Phase 1 checklist. Specify the exact prompt layout, tokenizer chat template invocation, and function signature:
  ```python
  def format_sql_prompt(schema_ddl: str, question: str) -> str: ...
  ```

---

### 2. Missing Dataset Acquisition Source and JSON Schema Contract
* **Severity:** 🔴 Blocker
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** Phase 1 (Lines 41–43); `data/train_logistics.json`, `data/test_logistics.json`, `data/databases/`
* **Hurdle:** The task "Download/curate 6 logistics SQLite databases and JSON splits... (~450 train, ~100 test)" provides neither the upstream download mechanism (e.g., Spider benchmark subset or pre-filtered fixture) nor the JSON schema contract for the split files. Downstream components (`data_loader.py`, `executor.py`, `reward.py`, `evaluate.py`, `train_grpo.py`, `test_leakage.py`) depend directly on exact field keys.
* **Remediation:** Document the exact curation script or source command (`scripts/prepare_data.py` or Spider download script) and define the JSON record schema:
  ```json
  {
    "db_id": "warehouse_management",
    "question": "What is the total quantity of items in warehouse W1?",
    "gold_sql": "SELECT SUM(quantity) FROM inventory WHERE warehouse_id = 'W1';",
    "difficulty": "medium"
  }
  ```

---

### 3. Underspecified Error Classification Taxonomy in `src/evaluate.py`
* **Severity:** 🔴 Blocker
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** Phase 2 (Lines 66–70), Phase 4 (Lines 105–106), Definition of Done (Line 171)
* **Hurdle:** The plan mandates that `src/evaluate.py` "automatically classifies all failures into the 8-class taxonomy", but does not define the decision tree or algorithmic rules. Distinguishing runtime errors is trivial, but programmatically diagnosing intent failures (`JOIN_ERROR` vs `CONDITION_ERROR` vs `AGGREGATION_ERROR` vs `VALUE_MISMATCH`) requires complex heuristics or an AST diff engine.
* **Remediation:** Align with the Stage 8 Zoom Out architecture: scope `src/evaluate.py` to 5 deterministic execution categories: `SYNTAX_ERROR`, `SCHEMA_HALLUCINATION`, `TIMEOUT_ABORT`, `EMPTY_RESULT`, and `VALUE_MISMATCH`. Enumerate the exact classification decision tree in Phase 2, and delegate relational root-cause diagnoses to the qualitative case studies in Phase 4.

---

### 4. Missing Explicit File Action Tags (`[Create]`, `[Modify]`, `[Delete]`)
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Lines 35–153 (Task Checklists across Phases 1–6)
* **Hurdle:** File paths throughout the checklists (`src/executor.py`, `src/reward.py`, `src/evaluate.py`, `notebooks/colab_train.ipynb`, `docs/RESULTS.md`, etc.) lack explicit operation tags. This violates the Implementation Readiness Standard's Target Scope & File Boundary rule, which requires explicitly marking each file to enforce write allowlists and prevent opportunistic scope creep.
* **Remediation:** Prefix every file reference with `[Create]`, `[Modify]`, or `[Delete]` (e.g., `[Create] src/executor.py`, `[Modify] docs/RESULTS.md`).

---

### 5. Underspecified Class and Function API Signatures
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Phase 1 (Lines 44–54); `src/executor.py`, `src/reward.py`
* **Hurdle:** Core function contracts are missing:
  1. `SafeSQLExecutor` mentions both `:memory:` and `?mode=ro` without declaring the method signature or return tuple.
  2. `compute_sql_reward` anti-hacking scoring rules for empty sets and trivial queries are not mathematically specified.
* **Remediation:** Document explicit class and function contracts:
  ```python
  class SafeSQLExecutor:
      def execute(self, db_path: str, sql: str, timeout: float = 1.0) -> Tuple[bool, List[Tuple], Optional[str]]: ...

  def compute_sql_reward(predicted_sql: str, gold_sql: str, db_path: str) -> float: ...
  ```
  Specify that $V_{\text{pred}} = []$ against non-empty $V_{\text{gold}}$ returns 0.0 match credit, and queries lacking table references receive 0.0.

---

### 6. Missing Test Plan Assertions and Test Infrastructure
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** Phase 1 (Lines 55–56); `tests/`
* **Hurdle:** The plan lists unit test filenames without specifying test cases or shared test infrastructure (`tests/conftest.py`, shared in-memory DB fixtures, sample schemas).
* **Remediation:** Specify `[Create] tests/conftest.py` with shared in-memory SQLite fixtures, and enumerate required test scenarios:
  - `test_executor.py`: (1) Valid SELECT returns rows; (2) Blocked commands (`DROP`, `DELETE`, `UPDATE`, `ATTACH`, `PRAGMA`, `ALTER`, `DETACH`) raise security error; (3) Runaway query (`WITH RECURSIVE cnt(x)...`) terminates within 1.0s; (4) Output truncates at 100 rows.
  - `test_reward.py`: (1) Exact multiset match scores 1.0; (2) Format adherence with syntax error scores 0.1; (3) Clean execution with wrong rows scores 0.3; (4) Unordered row equivalence scores 1.0; (5) Empty result against non-empty gold scores 0.0.
  - `test_leakage.py`: Asserts `set(train_db_ids).isdisjoint(set(test_db_ids))`.

---

### 7. Incomplete Colab T4 Hardware & Training Configuration
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Phase 2 (Line 67) & Phase 3 (Lines 80–92); `src/train_grpo.py`
* **Hurdle:** The plan specifies "16-bit", but Colab T4 GPUs lack native `bfloat16` support (`torch.float16` must be used). Furthermore, `GRPOTrainer` often defaults to `use_vllm=True`, which causes an immediate CUDA OOM on a single 15GB T4 when co-located with PyTorch. LoRA `target_modules` are also omitted.
* **Remediation:** Explicitly mandate in `src/train_grpo.py`:
  - `torch_dtype=torch.float16`
  - `GRPOConfig(..., use_vllm=False)`
  - `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05)`
  - Rollout sampling: `temperature=0.7`, `top_p=0.9`, `max_new_tokens=96`.

---

### 8. Underspecified TRL `GRPOTrainer` Reward Callable Interface
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** Phase 3 (Lines 80–85); `src/train_grpo.py` and `src/reward.py`
* **Hurdle:** In TRL, `GRPOTrainer` calls reward functions by passing batch completions and dataset columns via kwargs (`prompts, completions, gold_sql, db_id, **kwargs`). The dataset column schema and wrapper callable signature are not defined.
* **Remediation:** Document the dataset columns fed into `GRPOTrainer` (`prompt`, `gold_sql`, `db_path`) and the batch reward wrapper signature:
  ```python
  def grpo_sql_reward_func(completions, gold_sql, db_path, **kwargs) -> List[float]: ...
  ```

---

### 9. Uncalibrated Claim and Technical Inaccuracy in Risk Mitigations
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Section 3 (Lines 159, 160)
* **Style Rationale:** 
  - Line 159 claims static tiers "ensure gradients". In GRPO, intra-group normalization zeroes out advantages if all completions get identical partial scores. Action: `CALIBRATE` to: *"Tiered composite reward awards +0.1 for format and +0.2 for valid SQL, providing partial reward signals to promote non-zero gradient variance when execution fails early."*
  - Line 160 refers to cartesian joins "hanging GPU". SQLite executes in the host CPU process. Action: `CALIBRATE` to: *"Runaway cartesian joins blocking rollout execution thread"*.

---

### 10. Subjective Operator-State Criteria in Phase 5 Checklist
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Section 2, Phase 5 (Line 129)
* **Style Rationale:** Phrasing such as *"If time runs short or fatigue sets in, skip this phase cleanly"* introduces subjective personal-state criteria.
* **Recommended Revision:** *"If elapsed time exceeds schedule allocation, omit Phase 5."*

---

### 11. Emotive Framing & Redundant Defensive Caveats
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Section 2, The Cut-Line (Lines 115, 119); Phase 6 (Line 133); Section 4 (Line 172)
* **Style Rationale:** Terms like `"Honest Fallback"`, `"Real Resume Bullets"`, and `"Repo Polish"` use moralistic qualifiers and colloquialisms. Line 119 (`"4. Never fabricate numbers."`) is a redundant negative defensive caveat in an execution checklist.
* **Recommended Revision:** Change to `"Fallback Protocol"`, `"Phase 6: Documentation, Resume Impact Bullets & Repository Finalization"`, drop line 119, and calibrate line 172 to *"empirical evaluation metrics (no synthetic values or unpopulated placeholders)"*.

---

### 12. AST Blocklist Discrepancy with `AGENTS.md`
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** Phase 1 (Line 47) vs `AGENTS.md` Rule 3
* **Discrepancy:** Line 47 omits `ALTER`, `DETACH`, and `WRITEFILE` which are explicitly mandated by `AGENTS.md` Rule 3.
* **Remediation:** Update the blocked command list to: `DROP, DELETE, UPDATE, INSERT, ALTER, ATTACH, DETACH, PRAGMA, load_extension, WRITEFILE`.

---

### 13. Ambiguous CLI Flags and Evaluation Artifact Protocol
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** Phase 2 (Lines 66–72), Phase 4 (Lines 100–106)
* **Remediation:** Document CLI arguments for `src/evaluate.py`:
  ```bash
  python src/evaluate.py --model_path <path> [--adapter_path <path>] --split test --output_file docs/baseline_eval.json
  ```
  Specify that structured JSON artifacts (`docs/baseline_eval.json`, `docs/grpo_eval.json`) are generated before transcription into `docs/RESULTS.md`.

---

### 14. Redundant Phrasing and Wordy Intensifiers
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Phase 1 (Lines 45, 49), Phase 4 (Line 100), Section 4 (Line 170)
* **Style Rationale:** 
  - `"In-memory ephemeral DB loading"` ➔ `"In-memory database loading (`:memory:`)"`.
  - `"100-row limit truncation"` ➔ `"Truncation at 100 rows (`fetchmany(100)`)"`.
  - `"exact same 100 test questions"` ➔ `"same 100 test questions"`.
