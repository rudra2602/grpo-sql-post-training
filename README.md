# Post-Training Small Open LLMs with GRPO via Verifiable SQL Execution Rewards

## Project Overview
This project post-trains a small open-weight language model (**`Qwen/Qwen2.5-Coder-1.5B-Instruct`**) using **Group Relative Policy Optimization (GRPO)** with a verifiable, execution-based reward signal. 

Instead of relying on human feedback (RLHF) or an imperfect LLM judge, the model's generated SQL queries are executed against a safe, in-memory SQLite database environment. The reward is determined deterministically by:
1. **Format compliance** (parseable SQL blocks).
2. **Syntactic validity** (zero runtime SQL execution errors).
3. **Execution correctness** (multiset equivalence between predicted query results and ground-truth query results).

The project specifically evaluates post-training gains on multi-table **logistics, inventory, shipment tracking, and port terminal management** schemas (adapted from Spider benchmarks) to reflect real-world enterprise data challenges at global supply chain scale (DP World Group AI/ML).

> [!NOTE]
> **Data Contamination & Baseline Disclosure:** Because Spider is a public academic benchmark, its schemas and query patterns may exist within the pre-training or instruction-tuning corpora of `Qwen2.5-Coder-1.5B-Instruct`. Baseline zero-shot performance may consequently be elevated, and post-training GRPO improvements are targeted at domain-grounded execution consistency and edge-case syntax/timeout elimination rather than de novo syntax acquisition.

---

## Current Status: Planning & Specification Complete
* **Status Date:** 15 November 2026
* **Current Phase:** Phase 0 (Planning & Documentation Complete). No training code or weights have been executed yet.
* **Environment:** Hybrid development (Local Windows PC for development & testing; Google Colab T4 15GB for training).
* **Next Immediate Step:** Phase 1 implementation (Local environment, SQLite logistics datasets, safe executor, reward engine, and `pytest` suite).

> [!NOTE]
> All quantitative metrics, execution accuracy numbers, and failure mode statistics in this repository are strictly set to `TBD`. Real measured numbers will only be populated after verified model evaluation passes.

---

## System Architecture

```
Prompt: (Schema DDL + Question)
         │
         ▼
┌───────────────────────────────────────────────┐
│  Qwen2.5-Coder-1.5B-Instruct (LoRA Policy)   │
└───────────────────────────────────────────────┘
         │  Samples G=4 rollouts per prompt
         ▼
┌───────────────────────────────────────────────┐
│       Safe SQLite Execution Sandbox           │
│  - Ephemeral :memory: DB                      │
│  - set_authorizer security blocking           │
│  - 1.0s timeout & 100-row truncation limit    │
└───────────────────────────────────────────────┘
         │  Executes query against SQLite
         ▼
┌───────────────────────────────────────────────┐
│           Tiered Composite Reward             │
│  - Format adherence: +0.1                     │
│  - Syntax validity:   +0.2                     │
│  - Execution match:   +0.7 (Bag equality)     │
│  - Anti-hacking: Empty-set & trivial guards   │
└───────────────────────────────────────────────┘
         │  Normalized advantages A_i
         ▼
┌───────────────────────────────────────────────┐
│           TRL GRPOTrainer Engine              │
│  - Group relative advantage normalization     │
│  - Policy ratio clipping (epsilon = 0.2)      │
│  - KL divergence penalty (beta = 0.005)       │
│  - LoRA backward pass (r=16, alpha=32)        │
└───────────────────────────────────────────────┘
```

---

## Repository Directory Map

```text
├── data/
│   ├── databases/             # In-repo lightweight SQLite database files (~50 KB each)
│   ├── train_logistics.json   # 406 question-SQL pairs across 8 schemas
│   └── test_logistics.json    # 108 held-out pairs across 2 unseen schemas (zero-leakage)
├── scripts/
│   └── prepare_data.py        # Downloads Spider logistics subset and filters valid pairs
├── src/
│   ├── __init__.py
│   ├── executor.py            # Safe in-memory SQLite sandbox & timeout handler
│   ├── reward.py              # Tiered composite reward & anti-hacking checks
│   ├── data_loader.py         # Prompt formatting & schema injection
│   ├── evaluate.py            # Deterministic greedy decoding & 5-class failure taxonomy
│   └── train_grpo.py          # TRL GRPOTrainer script (runs on Colab or local GPU)
├── tests/
│   ├── conftest.py            # Shared in-memory SQLite fixtures and sample schemas
│   ├── test_executor.py       # Unit tests: enforces read-only, blocks mutations, enforces timeout
│   ├── test_reward.py         # Unit tests: verifies format, syntax, execution, and anti-hacking
│   └── test_leakage.py        # Unit tests: asserts 0 database/schema overlap between train and test
├── notebooks/
│   └── colab_train.ipynb      # Standalone Colab runner (clones repo, runs train_grpo.py)
├── docs/
│   ├── DECISIONS.md           # Log of all 12 architectural decisions and rationales
│   ├── REWARD_DESIGN.md       # Exact mathematical reward specification & anti-hacking rules
│   ├── EVAL_PROTOCOL.md       # Metrics, leakage prevention, and 5-class failure taxonomy
│   ├── ENVIRONMENT.md         # Local Windows and Colab environment setup steps
│   ├── EXPERIMENT_LOG.md      # Run tracking table (configs, seeds, runtimes, results)
│   ├── RESULTS.md             # Empty results template and resume bullet templates
│   └── LEARNING_NOTES.md      # Plain-language concept explanations for technical interviews
├── PLAN.md                    # Detailed backwards schedule, cut line, and task checklists
├── AGENTS.md                  # Autonomous agent instructions and engineering constraints
└── requirements.txt           # Python dependencies
```

---

## Core Specifications

* **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Smoke test: `Qwen2.5-Coder-0.5B-Instruct`).
* **Training Method:** Group Relative Policy Optimization (GRPO) via Hugging Face `trl`.
* **Parameter Efficiency:** 16-bit LoRA ($r=16, \alpha=32$, targeting linear projections `q_proj`, `k_proj`, `v_proj`, `o_proj`).
* **Group Size ($G$):** 4 completions per prompt.
* **Effective Batch Size:** 8 prompts (32 completions per step), ~100 calibrated steps (~90–120 minutes on Colab T4).
* **Generation Constraints:** `max_new_tokens=96`, `torch_dtype=torch.float16`, `use_vllm=False`.
* **Evaluation Protocol:** Deterministic greedy decoding ($T=0.0$) evaluated on 108 held-out questions on completely unseen database schemas (`tracking_orders` and `ship_1`).
* **Primary Metrics:**
  * **Execution Accuracy (EX %)**: Result set bag/multiset equivalence with gold SQL.
  * **Valid SQL Rate (VSR %)**: Syntax error-free execution percentage.
  * **Format Adherence Rate (FAR %)**: Clean markdown SQL block extraction percentage.
* **Diagnostic Taxonomy:** 5-class automated failure categorization (`SYNTAX_ERROR`, `SCHEMA_HALLUCINATION`, `TIMEOUT_ABORT`, `EMPTY_RESULT`, `VALUE_MISMATCH`), supplemented by 3 qualitative case studies for relational root causes (`JOIN_ERROR`, `CONDITION_ERROR`, `AGGREGATION_ERROR`).

---

## References & Documentation
* [Project Plan & Schedule](PLAN.md)
* [Architectural Decisions Log](docs/DECISIONS.md)
* [Reward Function Design](docs/REWARD_DESIGN.md)
* [Evaluation Protocol](docs/EVAL_PROTOCOL.md)
* [Agent Guidelines](AGENTS.md)
