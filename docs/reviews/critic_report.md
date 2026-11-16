# Critic Review Report (Stage 7)

**Target:** `PLAN.md` (FeatureRange, Lines 1–34)  
**Standard:** `resources/document_standards/feature-spec-standard.md`  
**Reviewers:** `egm-design` (Design Critic), `egm-structural` (Structural Auditor), `egm-style` (Technical Copy Editor)  
**Date:** 2026-11-14  

---

## Reviewer Status

| Reviewer | Lens | Status |
|---|---|---|
| `egm-design` | Architecture, Infrastructure Economics, Alternatives, Risks | Completed |
| `egm-structural` | Layout Hierarchy, Symmetry, Template Compliance | Completed |
| `egm-style` | Precision, Tone, Brevity, Claim Calibration | Completed |

---

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. Missing Mandatory Architecture Sections (Alternatives Considered, Problem Statement, Goals/Non-Goals)
* **Severity:** 🔴 Blocker
* **Reviewer Lens:** `egm-design`, `egm-structural`
* **Affected Scope:** Lines 1–8 (Specification Preamble)
* **Discrepancy Description:** The document fails the **Feature Specification Standard** on multiple required components:
  1. *The Alternatives Rule Violation:* Specifies a complex post-training stack (`Qwen2.5-Coder-1.5B-Instruct` + GRPO + SQLite execution rewards) without documenting or evaluating rejected alternatives (e.g., SFT, DPO, PPO, or model size choices such as 0.5B vs 1.5B vs 3B).
  2. *Missing Problem Statement & Goals/Non-Goals:* The business/technical problem is unstated, and scope boundaries (cross-database generalization, multi-turn handling, token budgets) are not explicitly framed.
* **Recommendation:** Insert architectural sections before the timeline detailing:
  - **Problem Statement:** Why standard SFT falls short on logistics Text-to-SQL (syntax memorization vs. schema-grounded semantic execution).
  - **Goals & Non-Goals:** Explicitly establish in-scope deliverables (LoRA fine-tuning for exact execution match on unseen logistics schemas) vs out-of-scope tasks (full parameter fine-tuning, web UI, multi-turn conversation).
  - **Alternatives Evaluation Matrix:** Document concrete trade-offs comparing GRPO vs SFT vs DPO, justifying GRPO for reward-driven self-exploration without pairwise human preference curation.

---

### 2. Unsubstantiated Rollout Generation Throughput & Training Window Bottleneck
* **Severity:** 🔴 Blocker
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Lines 18–19 (Phase 3 Milestone Timeline)
* **Discrepancy Description:** Phase 3 allocates 4.0 hours (12:00 PM – 4:00 PM IST) for both a 0.5B smoke test and a 200-step 1.5B GRPO training run on a free-tier Colab T4 GPU, assuming 200 steps will take ~65–75 minutes. Unlike SFT, each GRPO step autoregressively generates $G=4$ completions across an effective batch size of 8 (~4,800 tokens per step). On a standard T4 GPU without vLLM (~20–30 tokens/sec decoding speed), 200 steps of rollout generation requires 8.5 to 13 hours—exceeding the budgeted 4-hour window by 200%–300%.
* **Recommendation:** Apply concrete generation constraints:
  - Cap maximum completion tokens strictly to $\le 128$ tokens (SQL queries rarely exceed 80 tokens).
  - Adopt 4-bit / 8-bit quantized base weights (QLoRA) to maximize memory bandwidth and generation speed on a single T4.
  - Implement a throughput gate during the 0.5B smoke test: if measured step latency exceeds 45s, scale steps down (e.g., from 200 to 100) or adjust group size $G$ to guarantee execution within the allotted window.

---

### 3. Structural Section Inversion
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-structural`
* **Affected Scope:** Line 9 (`## 1. Backwards Timeline & Milestones`)
* **Discrepancy Description:** The document immediately leads with chronological execution milestones (`## 1. Backwards Timeline & Milestones`) ahead of feature definition, problem statement, and architectural context. This disrupts logical reading flow and violates standard document hierarchy.
* **Recommendation:** Place execution timelines after specification and architecture sections (`Problem Statement`, `Goals & Non-Goals`, `Alternatives Considered`).

---

### 4. Single Point of Failure (SPOF) on Colab T4 Quotas & Preemption
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Lines 16–20 (Phase 2 & Phase 3 Scheduling)
* **Discrepancy Description:** Chaining Phase 2 (3 hrs) and Phase 3 (4 hrs) sequentially on Google Colab free-tier risks preemption or GPU quota exhaustion (Colab free tier frequently cuts sessions after 3–4 hours of heavy compute). If locked out mid-run, the project has no identified secondary execution infrastructure.
* **Recommendation:** Specify compute failover options (e.g. Kaggle Notebooks free dual T4 GPUs or on-demand spot instance on RunPod/Lambda Labs) and ensure checkpoints are saved to external persistent storage (Google Drive) every 25–50 steps.

---

### 5. Delayed Cut-Line Trigger & 15-Hour Operational Vacuum
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Lines 20–25 (Phase 4 & The Cut Line)
* **Discrepancy Description:** Phase 4 ends at Saturday 9:00 PM IST, but the Cut-Line decision trigger is set to Sunday 12:00 PM IST (Noon), creating an unallocated 15-hour gap in a 60-hour total project timeline. A training stall on Saturday afternoon would waste critical hours before enacting fallback diagnosis.
* **Recommendation:** Move the Cut-Line trigger immediately to the conclusion of Phase 4 (Saturday 15 Nov, 10:00 PM IST) to govern overnight actions and enable immediate Sunday morning diagnostic reporting if blocked.

---

### 6. Subjective and Moralizing Terminology in Project Goal
* **Severity:** 🟡 Major
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Line 3 (`Target Goal`)
* **Style Rationale:** The phrase `"honest baseline comparison"` introduces subjective moralizing rather than dry, verifiable engineering framing.
* **Editorial Action:** `CALIBRATE`. Replace with `"empirical baseline comparison"` or `"controlled baseline comparison"`.

---

### 7. Document Title Redundancy
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Line 1 (`# Master Project Plan & Execution Schedule`)
* **Style Rationale:** Contains filler word `"Master"` and synonym bloat.
* **Editorial Action:** Simplify to `# Project Execution Plan`.

---

### 8. Temporal Inconsistency & Pleonasm in Milestone Descriptions
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Line 14 (`Phase 1`)
* **Style Rationale:** Uses conversational `"Tonight"` instead of uniform absolute date (`Fri 14 Nov`). `"Safe Sandbox"` is pleonastic.
* **Editorial Action:** Replace with `Fri 14 Nov: 9:00 PM - 11:45 PM IST` and `"SQLite Sandbox"`.

---

### 9. Colloquial Terminology in Later Milestones
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Lines 26, 28
* **Style Rationale:** Phrases such as `"Repo Polish"`, `"Resume Bullets"`, and `"Sanity Review"` are informal colloquialisms.
* **Editorial Action:** Replace with `"Repository Finalization"`, `"Empirical Metric Summary"`, and `"Final Validation"`.

---

### 10. Bare Code Fence Syntax
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-structural`
* **Affected Scope:** Line 11
* **Discrepancy Description:** Milestone ASCII diagram uses a bare triple backtick fence without language identifier.
* **Recommendation:** Specify explicit language identifier (```` ```text ````).

---

### 11. Cross-Platform Local-to-Cloud Environment Parity
* **Severity:** 🟢 Nit
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Lines 14–17
* **Discrepancy Description:** Developing on Windows local (`grpo-sql`) and running on Colab Linux risks path and dependency mismatches.
* **Recommendation:** Mandate `pathlib.Path` usage and schedule a pre-flight notebook environment check on Friday night.
