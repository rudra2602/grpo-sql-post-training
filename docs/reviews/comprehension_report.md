# Comprehension Test Report (Stage 6)

**Target:** `PLAN.md` (Lines 1–173)  
**Auditor:** `egm-comprehension` (Naive Reader Persona)  
**Date:** 2026-11-14  

---

### What this proposes
This document outlines a time-budgeted execution plan to post-train `Qwen2.5-Coder-1.5B-Instruct` on logistics text-to-SQL tasks using Group Relative Policy Optimization (GRPO) with sandboxed SQLite execution rewards. It schedules work across seven sequential phases spanning Friday evening to Monday morning, progressing from local Windows environment setup and test-driven reward engine implementation to Google Colab T4 baseline evaluation, smoke testing, primary LoRA training, and error taxonomy analysis. It also establishes a strict cut-line mechanism to fall back to an empirical diagnostic evaluation report if training stalls before Sunday noon, ensuring honest and verifiable outcomes prior to final application submission.

### How the existing system works, per this document
Per this document, the starting state is an untuned base model (`Qwen2.5-Coder-1.5B-Instruct`) without recorded text-to-SQL performance benchmarks on logistics schemas. The core technical components—including the safe SQLite execution sandbox, the multi-tier reward scoring logic, the test suite, and the training pipeline—do not yet exist and must be implemented from scratch. The system initially lacks curated database pairs, an automated error classifier, and measured metrics, requiring all infrastructure to be built locally and validated through unit tests before any GPU deployment.

---

## Reviewer Findings

### 1. Undefined 8-Class Failure Taxonomy Specification
* **Severity:** 🟡 Major
* **Affected Scope:** Phase 2 (Line 70), Phase 4 (Line 105), The Cut-Line (Line 116), Definition of Done (Line 171) — `"8-class taxonomy"` / `"8-class automated failure taxonomy"`
* **Discrepancy Description & Rationale:** The plan repeatedly mandates that `src/evaluate.py` automatically classify all query errors into an "8-class taxonomy", and lists this taxonomy breakdown as a mandatory completion criterion in the Definition of Done. However, the document never enumerates, defines, or links to the specific categories comprising this 8-class taxonomy. A newly onboarded developer cannot implement or validate `src/evaluate.py` without knowing the expected error classes.

---

### 2. Missing Ingestion / Origin Source for Logistics Databases and Splits
* **Severity:** 🟡 Major
* **Affected Scope:** Phase 1 (Lines 41–43) — `"Download/curate 6 logistics SQLite databases and JSON splits"`
* **Discrepancy Description & Rationale:** The plan directs the developer to "download/curate" six specific logistics databases (`warehouse_management`, `store_inventory`, `customers_and_invoices`, `products_catalog`, `shipping_and_containers`, `freight_tracking`) and ~550 query-SQL pairs across train and test splits. However, it provides no source repository, URL, benchmark reference (such as Spider or BIRD subsets), or synthetic curation script. An engineer starting this task has no specification for where these datasets reside or how they should be assembled.

---

### 3. Missing Git Synchronization Between Phase 1 (Local) and Phase 2 (Colab)
* **Severity:** 🟡 Major
* **Affected Scope:** Workflow sequence between Phase 1 (Lines 37–57) and Phase 2 (Line 65)
* **Discrepancy Description & Rationale:** Phase 2 instructs the developer to set up `notebooks/colab_train.ipynb` on Google Colab to "clone repository", but Phase 1 only covers local development and testing on Windows without any task to initialize git, commit, or push the code to a remote repository (e.g., GitHub). Pushing to GitHub is only scheduled in Phase 6 (Line 140). A developer following this schedule sequentially would find Colab unable to clone the uncommitted and unpushed local codebase.

---

### 4. Ambiguous Definition of Trivial Query Rejection
* **Severity:** 🟢 Nit
* **Affected Scope:** Phase 1 (Line 54) — `"trivial query rejection"`
* **Discrepancy Description & Rationale:** Under `compute_sql_reward`, the plan lists "Anti-hacking logic (empty-set protection, trivial query rejection)". While empty-set protection is intuitive, "trivial query rejection" is ambiguous. The document does not specify whether this rejects constant projections (`SELECT 1`), queries without `FROM` clauses, or unconditional table scans, leaving the implementation rules open to subjective interpretation.

---

### 5. Acronym Expansion on First Mention
* **Severity:** 🟢 Nit
* **Affected Scope:** Header / Section 1 (Line 3) — `"GRPO"`
* **Discrepancy Description & Rationale:** The core training method acronym "GRPO" is introduced in the target goal without being spelled out or linked to reference documentation (e.g., Group Relative Policy Optimization). While later sections mention "group relative advantage normalization", expanding the acronym or linking it on first mention ensures complete self-containment for fresh readers.
