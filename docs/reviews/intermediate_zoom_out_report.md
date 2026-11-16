# Intermediate Zoom Out Review Report (Stage 8)

**Target:** `PLAN.md` (Full File, Lines 1–173)  
**Standard:** `resources/document_standards/zoom-out-standard.md`  
**Auditor:** `egm-zoom-out` (Skeptical Architect Persona)  
**Date:** 2026-11-14  

---

## Executive Summary

The foundational strategy of `PLAN.md` (hybrid local Windows development/testing + Google Colab T4 execution, pure standard library SQLite sandboxing, and a 3-tier scalar execution reward) is sound, lean, and appropriately sized for a 60-hour sprint.

Auditing `PLAN.md` against prior reports from Stage 5 (TRIZ Inventive Catalyst), Stage 6 (Comprehension), and Stage 7 (Critic) establishes two key takeaways:
1. **Critical Operational Constraint**: Autoregressive rollout generation across 200 steps with 192 max tokens on a single Colab T4 GPU will severely violate the 4-hour Phase 3 window.
2. **Rejection of Secondary Over-Engineering**: Several mitigations proposed by prior review lenses (e.g., adding AST Jaccard reward shaping, introducing `ProcessPoolExecutor` IPC sandboxing, introducing 4-bit QLoRA dependencies, or duplicating multi-page architectural trade-off matrices into an operational task checklist) violate YAGNI and introduce unnecessary points of failure.

---

## Prioritized Findings & Strategic Directives

### 1. Schedule Feasibility: Right-Sizing Rollout Budget on Colab T4
* **Severity:** 🔴 Blocker
* **Affected Scope:** Section 1 (Lines 18–19); Section 2, Phase 3 (Lines 77–95)
* **Simplicity Rationale:** At $G=4$, batch size 1 with gradient accumulation 8 (32 completions per step) and up to 192 tokens per completion, each step generates up to 6,144 tokens. On a standard Colab T4 without vLLM (~20–30 tokens/sec decoding), 200 steps requires 8.5 to 13 hours of generation alone.
* **Minimalist Resolution (Rejecting 4-bit QLoRA):** Rather than introducing external quantization dependencies (`bitsandbytes`) or dynamic latency gates, solve this directly at the parameter level:
  1. **Strict Token Ceiling:** Cap `max_new_tokens` strictly to 96 (SQL queries on logistics schemas average 30–50 tokens; 192 is wasted compute).
  2. **Step Count Right-Sizing:** Set total training steps to 100 steps (3,200 rollouts is sufficient for initial policy adaptation).
  3. **Batch Generation:** Enforce batched prompt generation with left-padding.
  This reduces rollout tokens by ~70%, bringing Phase 3 comfortably within a 90–120 minute window on native 16-bit LoRA without external quantization hacks.

---

### 2. Specification Modularity: Rejecting Duplication of Architecture Matrices
* **Severity:** 🟡 Major
* **Affected Scope:** Section 1 / Specification Preamble (Lines 1–8); Critic Report Finding 1
* **Simplicity Rationale:** The Critic recommended adding "Problem Statement", "Goals & Non-Goals", and an "Alternatives Considered Matrix" into `PLAN.md`. However, `PLAN.md` is strictly an **operational timeline, task checklist, and risk contingency register**. The complete system architecture and problem overview already reside in `README.md`, and all 12 architectural decisions (including explicit trade-offs and rejected alternatives for Compute, Model, Framework, and Reward) are exhaustively documented in `docs/DECISIONS.md`. Duplicating multi-page architectural prose into `PLAN.md` violates Separation of Concerns.
* **Minimalist Resolution:** Maintain document modularity. Insert a concise context banner at the top of `PLAN.md` linking directly to `README.md` and `docs/DECISIONS.md`.

---

### 3. Complexity Control: Rejecting AST Reward Shaping in the Training Loop
* **Severity:** 🟡 Major
* **Affected Scope:** Section 2, Phase 1 (Lines 50–54, `src/reward.py`); Section 3 (Line 159, Risk 2); Stage 5 Inventive Catalyst Finding 3
* **Simplicity Rationale:** Stage 5 proposed adding partial AST component-matching (table/column Jaccard overlap) to `compute_sql_reward` to prevent intra-group zero-variance gradient starvation. This introduces third-party parser dependencies (`sqlglot`/`sqlparse`), creates reward hacking vulnerabilities (the model dumps column names into comments to harvest partial reward), and directly contradicts `docs/DECISIONS.md` Decision 5 where AST matching was evaluated and rejected.
* **Minimalist Resolution:** Retain the lean 3-tier scalar reward (Format +0.1, Syntax +0.2, Execution +0.7). Intra-group zero-variance on difficult prompts is handled by gradient accumulation across 8 prompts.

---

### 4. Sandboxing Simplicity: Rejecting Multi-Process Sandbox Pool
* **Severity:** 🟡 Major
* **Affected Scope:** Section 2, Phase 1 (Lines 44–49, `src/executor.py`); Stage 5 Inventive Catalyst Finding 6
* **Simplicity Rationale:** Stage 5 suggested replacing `threading` with disposable `ProcessPoolExecutor` subprocess workers. In a hybrid Windows/Linux workflow interacting with PyTorch, spawning subprocesses incurs 100–300ms startup latency on Windows, cannot cleanly serialize unpicklable SQLite connections, and introduces CUDA re-initialization crashes in PyTorch on Colab Linux.
* **Minimalist Resolution:** Keep `src/executor.py` pure standard library. Using standard `sqlite3` in-memory connections with `set_authorizer` and `threading.Timer` with `conn.interrupt()` operates with sub-millisecond overhead, zero external dependencies, and complete thread safety.

---

### 5. Automated Failure Taxonomy Scoping in src/evaluate.py
* **Severity:** 🟡 Major
* **Affected Scope:** Section 2, Phase 2 (Lines 66–70); Phase 4 (Lines 105–108); Section 4, Item 4 (Line 171)
* **Simplicity Rationale:** Mandating that `src/evaluate.py` automatically classify all query failures into an 8-class taxonomy during a 3-hour Saturday morning window is an over-engineering trap. Distinguishing `JOIN_ERROR`, `CONDITION_ERROR`, and `AGGREGATION_ERROR` on queries that executed without runtime errors requires a complex AST differential engine.
* **Minimalist Resolution:** Scope `src/evaluate.py` to 5 deterministic execution categories: `SYNTAX_ERROR`, `SCHEMA_HALLUCINATION`, `TIMEOUT_ABORT`, `EMPTY_RESULT`, and `VALUE_MISMATCH`. Reserve fine-grained relational root-cause diagnoses for the 3 qualitative case studies in Phase 4.

---

### 6. Missing Data Provisioning Path in Phase 1
* **Severity:** 🟡 Major
* **Affected Scope:** Section 2, Phase 1 (Lines 41–43)
* **Simplicity Rationale:** Task 2 vaguely instructs the developer to "Download/curate 6 logistics SQLite databases and JSON splits... (~450 train, ~100 test)". Curating these from scratch would consume the entire Friday night window.
* **Minimalist Resolution:** Explicitly specify the data origin in `PLAN.md` (e.g. referencing a pre-filtered repository fixture or a lightweight extraction script `scripts/prepare_data.py`) so data provisioning takes <15 minutes.

---

### 7. Missing Git Synchronization Checkpoint Before Colab Launch
* **Severity:** 🟢 Nit
* **Affected Scope:** Workflow transition between Phase 1 (Line 57) and Phase 2 (Line 65)
* **Simplicity Rationale:** Phase 2 begins by cloning the repository in Colab, but git commit and push is currently deferred until Phase 6. A local-to-cloud handoff checkpoint (`git add/commit/push`) must be placed at the conclusion of Phase 1.

---

### 8. Operational Cut-Line Schedule Alignment
* **Severity:** 🟢 Nit
* **Affected Scope:** Section 1 (Lines 22–25); Section 2 (Lines 112–120)
* **Simplicity Rationale:** Phase 4 concludes at 9:00 PM Saturday, but the Cut-Line decision trigger is set for 12:00 PM Sunday (a 15-hour vacuum). Moving the operational Cut-Line review to Saturday 10:00 PM IST ensures Sunday morning is actively used for either the Phase 5 ablation or the diagnostic fallback report.
