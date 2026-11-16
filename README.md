# Post-Training Small Open LLMs with GRPO via Verifiable SQL Execution Rewards

## Project Overview
This project post-trains a small open-weight language model (**`Qwen/Qwen2.5-Coder-1.5B-Instruct`**) using **Group Relative Policy Optimization (GRPO)** with a verifiable, execution-based reward signal. 

Instead of relying on human feedback (RLHF) or an imperfect LLM judge, the model's generated SQL queries are executed against a safe, in-memory SQLite database environment. The reward is determined deterministically by:
1. **Format compliance** (parseable SQL markdown tags).
2. **Syntactic validity** (zero runtime SQL execution errors).
3. **Execution correctness** (multiset equivalence between predicted query results and ground-truth query results).

The project specifically evaluates post-training gains on multi-table **logistics, inventory, shipment tracking, and port terminal management** schemas (adapted from Spider benchmarks) to reflect real-world enterprise data challenges at global supply chain scale (DP World Group AI/ML).

---

## Benchmark Results (Cross-Database Held-Out Test Set - Harness v2)

Evaluated on `data/test_logistics.json` (108 held-out questions on strictly unseen Spider schemas: `ship_1` and `tracking_orders`) under **deterministic greedy decoding ($T=0.0$)**, `max_new_tokens=512`, 16-bit (`float16`) precision on an NVIDIA GeForce RTX 4050 Laptop GPU, hard 1.0s sandbox timeout, and multiset row matching.

| Metric | Base Model (`Qwen2.5-Coder-1.5B-Instruct`) | Post-Trained Model (GRPO LoRA Adapter) | Absolute Delta ($\Delta$) | Wilson 95% Confidence Intervals | Statistical Significance |
|---|---|---|---|---|---|
| **Execution Accuracy (EX %)** | **58.33 %** (63 / 108) | **60.19 %** (65 / 108) | **+1.85 %** (+2 queries) | Base: [48.9%, 67.2%]<br>GRPO: [50.8%, 68.9%] | $p = 0.5000$ (McNemar exact; within noise) |
| **Valid SQL Rate (VSR %)** | **87.04 %** (94 / 108) | **87.04 %** (94 / 108) | **0.00 %** (exact parity) | Base: [79.4%, 92.1%]<br>GRPO: [79.4%, 92.1%] | Identical valid SQL rates |
| **Format Adherence Rate (FAR %)** | **100.00 %** (108 / 108) | **100.00 %** (108 / 108) | **0.00 %** (exact parity) | Base: [96.6%, 100.0%]<br>GRPO: [96.6%, 100.0%] | Identical format adherence |

* **Maritime Domain (`ship_1`):** Execution accuracy improved from **50.00%** (24/48) to **54.17%** (26/48) (+2 queries).
* **Order Fulfillment (`tracking_orders`):** Execution accuracy was **65.00%** (39/60) for both models (0 delta across all 60 queries).
* **Paired Contingency Analysis:** 63 queries were solved by both models; 2 queries were flipped from incorrect to correct by GRPO; **0 regressions occurred** (no question solved by Base was missed by GRPO). The exact two-sided McNemar test yields $p = 0.5000$, indicating that the +2 query gain is within expected sampling variance.
* **Pre-Training Data Caveat:** Because Spider schemas are public, query patterns and table structures may be present in the pre-training data of `Qwen2.5-Coder-1.5B-Instruct`, accounting for the strong baseline performance.
* **Automated Failure Taxonomy:** Real errors are semantic rather than syntactic: `VALUE_MISMATCH` (21.3%) and `SCHEMA_HALLUCINATION` (12.0%), while `SYNTAX_ERROR` was only 0.9% (1 query) for both models.

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
│   ├── databases/             # Clean SQLite databases (10 multi-table logistics schemas)
│   ├── train_logistics.json   # 406 question-SQL pairs across 8 schemas
│   └── test_logistics.json    # 108 held-out pairs across 2 unseen schemas (zero-leakage)
├── checkpoints/
│   └── grpo_final/            # Saved LoRA adapter weights (checkpoint-10, 20, 30, final)
├── src/
│   ├── __init__.py
│   ├── executor.py            # Safe in-memory SQLite sandbox & timeout handler
│   ├── reward.py              # Tiered composite reward & anti-hacking checks
│   ├── data_loader.py         # Prompt formatting & schema injection
│   ├── evaluate.py            # Deterministic greedy decoding & 6-class failure taxonomy (Harness v2)
│   ├── statistics.py          # Wilson score CIs and exact McNemar hypothesis testing
│   └── train_grpo.py          # TRL GRPOTrainer script (with 6GB VRAM memory safeguards)
├── tests/
│   ├── conftest.py            # Shared in-memory SQLite fixtures and sample schemas
│   ├── test_executor.py       # Unit tests: enforces read-only, blocks mutations, enforces timeout
│   ├── test_reward.py         # Unit tests: verifies format, syntax, execution, and anti-hacking
│   ├── test_leakage.py        # Unit tests: asserts 0 database/schema overlap between train and test
│   ├── test_evaluate.py       # Unit tests: 6-class failure taxonomy and token truncation checks
│   └── test_statistics.py     # Unit tests: Wilson score intervals and McNemar binomial probabilities
├── notebooks/
│   └── colab_train.ipynb      # Standalone Colab runner
├── scripts/
│   ├── compare_evals.py       # Compares evaluation runs, computes CIs and McNemar test
│   └── prepare_data.py        # Data curation and schema validation pipeline
├── docs/
│   ├── RESULTS.md             # Full experimental metrics, deltas, and qualitative case studies
│   ├── EXPERIMENT_LOG.md      # Detailed logs of all baseline and post-training runs
│   ├── LEARNING_NOTES.md      # Technical deep-dives & interview handbook for DP World ML track
│   ├── DECISIONS.md           # Architectural decision records
│   ├── REWARD_DESIGN.md       # Reward function mathematical formulation
│   ├── EVAL_PROTOCOL.md       # Evaluation protocol and zero-leakage guidelines
│   └── reviews/               # Archived stage 5-9 review audits and reports
├── PLAN.md                    # Project roadmap and definition of done
└── AGENTS.md                  # Autonomous agent directives and engineering constraints
```

---

## Quickstart & Replication

```bash
# 1. Activate conda environment
conda activate grpo-sql

# 2. Run unit tests (42/42 passing)
pytest tests/ -v

# 3. Run baseline evaluation under Harness v2 (512 tokens, float16)
python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --split test --precision float16 --max_new_tokens 512 --output_file docs/eval_v2_base.json

# 4. Evaluate GRPO post-trained model under Harness v2 (512 tokens, float16)
python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter_path checkpoints/grpo_final --split test --precision float16 --max_new_tokens 512 --output_file docs/eval_v2_grpo.json

# 5. Generate comparative statistical report (Wilson CIs, McNemar exact p-value)
python scripts/compare_evals.py
```

---

## Documentation & Deep-Dives
* [Experimental Results & Qualitative Analysis](docs/RESULTS.md)
* [Run Tracking & Hyperparameters Log](docs/EXPERIMENT_LOG.md)
* [Interview Preparation Handbook](docs/LEARNING_NOTES.md)
* [Reward Design Specification](docs/REWARD_DESIGN.md)
* [Evaluation Protocol & Taxonomy](docs/EVAL_PROTOCOL.md)
