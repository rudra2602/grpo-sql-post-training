# Initial Zoom Out Review Report (Stage 5)

**Target:** `PLAN.md` (Full File, Lines 1–173)  
**Standard:** `resources/document_standards/zoom-out-standard.md` & TRIZ Inventive Software Patterns  
**Date:** 2026-11-14  

---

## Reviewer Status

| Reviewer | Lens | Status |
|---|---|---|
| `egm-zoom-out` | Skeptical Architect (Simplicity, YAGNI, Anti-Weeds) | Completed |
| `egm-inventive-catalyst` | Inventive Catalyst (TRIZ Contradiction Analysis) | Completed |

---

## Prioritized Findings

### 1. Unrealistic Scope & Over-Engineering in Automated Error Taxonomy
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Section 2, Phase 2 (lines 66–70); Phase 4 (lines 105–108); Section 4, Definition of Done item 4 (line 171).
* **Simplicity Rationale:** Mandating that `src/evaluate.py` automatically classifies all failures into an 8-class taxonomy requires diagnosing semantic intent (e.g. distinguishing `JOIN_ERROR`, `CONDITION_ERROR`, and `AGGREGATION_ERROR` on queries that executed without runtime errors). Building a reliable SQL semantic AST differential engine in a 3-hour window risks producing fragile heuristics or stalling the timeline.
* **Recommendation:** Simplify the automated classifier in `src/evaluate.py` to 5 deterministic execution categories: `SYNTAX_ERROR`, `SCHEMA_HALLUCINATION`, `TIMEOUT_ABORT`, `EMPTY_RESULT`, and `VALUE_MISMATCH`. Reserve fine-grained relational root-cause categorization for the qualitative case studies in Phase 4.

---

### 2. Synchronous In-Loop SQL Execution Bottleneck in GRPO Rollout
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** Section 2, Phase 1 & Phase 3; Section 3 (Risk 3) / `SafeSQLExecutor` & `train_grpo.py` In-Loop Pipeline
* **False Compromise:** Assumes safe SQL execution requires synchronously copying databases to `:memory:` and executing queries on the main training thread. Across 200 steps with $G=4$ and gradient accumulation 8 (6,400 query executions), sequential 1.0s timeouts for cartesian joins starve GPU utilization.
* **Proposed Resolution:** Preliminary Action & Extraction. Pre-warm shared read-only database connections at trainer initialization rather than copying per query. Screen queries with sub-millisecond static AST checks to reject unjoined multi-table queries before dispatch.
* **Secondary Liabilities:** Memory consumption for pre-warmed connection pools alongside PyTorch VRAM.

---

### 3. Intra-Group Zero-Variance Gradient Starvation in GRPO Reward
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** Section 2, Phase 1 & Phase 3; Section 3 (Risk 2) / Reward Engine (`compute_sql_reward`)
* **False Compromise:** Assumes static reward tiers (+0.1, +0.2, +0.7) prevent gradient collapse. GRPO normalizes advantages within the group ($G=4$): $A_i = (R_i - \text{mean}(R)) / (\text{std}(R) + \epsilon)$. If all 4 completions achieve valid syntax but fail execution on a tough query, all receive $R_i = 0.3$. Intra-group variance becomes zero ($\text{std}(R)=0$), collapsing advantages to zero and starving policy gradients.
* **Proposed Resolution:** Dynamics / Continuous Shaping. Incorporate fine-grained differential signals (e.g., partial column/table Jaccard overlap from AST analysis) alongside execution outcome to maintain non-zero intra-group reward variance even when no query matches ground truth.
* **Secondary Liabilities:** Slightly increased reward calculation complexity; requires tuning so auxiliary signals do not incentivize column dumping over execution correctness.

---

### 4. Triplicated Sandboxing & Altitude Drop into Implementation Weeds
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Section 2, Phase 1 (lines 44–49, `src/executor.py`)
* **Simplicity Rationale:** The plan specifies redundant sandboxing layers (in-memory `:memory:`, read-only `?mode=ro`, and custom AST blocklist). In-memory or read-only mode already prevents disk mutation. A custom AST parser adds unnecessary dependencies (`sqlglot`/`sqlparse`), ignoring that standard Python `sqlite3` natively provides `conn.set_authorizer()` to block mutating opcodes at the C level. Listing specific function names and URI params drops into low-level implementation weeds.
* **Recommendation:** Define the sandbox by its architectural contract (ephemeral in-memory execution, read-only enforcement, timeout, row limit). Use built-in `set_authorizer` or keyword rejection instead of external AST parser dependencies.

---

### 5. YAGNI Violation in Anti-Hacking Guard for Empty Results
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Section 2, Phase 1 (line 54, `src/reward.py`)
* **Simplicity Rationale:** The plan proposes schema intersection checks to guard against trivial queries returning empty results. However, dataset curation guarantees all gold references return $\ge 1$ row ($V_{\text{gold}} \neq []$). Multiset comparison between $V_{\text{pred}} = []$ and $V_{\text{gold}} \neq []$ already naturally awards 0 points. Implementing schema intersection logic is speculative dead code.
* **Recommendation:** Return 0.0 execution match points whenever $V_{\text{pred}}$ is empty against a non-empty gold reference. Omit complex schema intersection algorithms.

---

### 6. SQLite Threading Preemption Limitations
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** Section 2, Phase 1 / `SafeSQLExecutor` Timeout Implementation
* **False Compromise:** Python threads cannot interrupt blocking SQLite C-routine calls, and `set_progress_handler` tracks bytecode instructions rather than wall-clock time.
* **Proposed Resolution:** Use disposable subprocess workers (`concurrent.futures.ProcessPoolExecutor`) with OS-level signals or strictly calibrated `set_progress_handler` instruction counts to enforce termination without locking the main thread.
* **Secondary Liabilities:** Inter-process overhead.

---

### 7. Premature Cut-Line All-or-Nothing Dichotomy
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** Section 2, The Cut-Line & Phase 4
* **False Compromise:** The cut-line suggests that failing full 200-step training by Sunday noon means abandoning post-training evaluation and falling back to zero-shot baseline analysis only, discarding intermediate 50-step checkpoints saved to Google Drive.
* **Proposed Resolution:** Beforehand Cushioning. Include an evaluation protocol for the latest healthy intermediate checkpoint (e.g. step 50, 100, 150) before declaring total training failure.
* **Secondary Liabilities:** Slight time allocation during failure triage.

---

### 8. Premature CLI Abstraction for Conditional Stretch Ablation
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Section 2, Phase 3 (line 80) and Phase 5 (lines 123–130)
* **Simplicity Rationale:** Introducing CLI flags like `--reward_mode binary` in Phase 3 before the primary tiered reward pipeline is validated adds unneeded branching logic.
* **Recommendation:** Keep `train_grpo.py` focused strictly on the tiered reward. Introduce ablation flags only if Phase 5 is triggered at the Cut-Line.
