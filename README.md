# Post-Training Small Open LLMs with GRPO via Verifiable SQL Execution Rewards

## Project Overview
This project post-trains a small open-weight language model (**`Qwen/Qwen2.5-Coder-1.5B-Instruct`**) using **Group Relative Policy Optimization (GRPO)** with a verifiable, execution-based reward signal. 

Instead of relying on human feedback (RLHF) or an imperfect LLM judge, the model's generated SQL queries are executed against a safe, in-memory SQLite database environment. The reward is determined deterministically by:
1. **Format compliance** (parseable SQL markdown tags).
2. **Syntactic validity** (zero runtime SQL execution errors).
3. **Execution correctness** (multiset equivalence between predicted query results and ground-truth query results).

The project specifically evaluates post-training gains on multi-table **logistics, inventory, shipment tracking, and port terminal management** schemas (adapted from Spider benchmarks) to reflect real-world enterprise data challenges at global supply chain scale (DP World Group AI/ML).

---

## Benchmark Results (Cross-Database Held-Out Test Set)

Evaluated on `data/test_logistics.json` (108 held-out questions on strictly unseen logistics and shipping database schemas: `tracking_orders` and `ship_1`) under **deterministic greedy decoding ($T=0.0$)**, hard 1.0s sandbox timeout, and multiset row matching.

| Metric | Base Model (`Qwen2.5-Coder-1.5B-Instruct`) | Post-Trained Model (GRPO LoRA) | Absolute Delta ($\Delta$) | Relative Improvement (%) |
|---|---|---|---|---|
| **Execution Accuracy (EX %)** | **13.89 %** (15/108) | **16.67 %** (18/108) | **+2.78 %** | **+20.01 %** |
| **Valid SQL Rate (VSR %)** | **19.44 %** (21/108) | **24.07 %** (26/108) | **+4.63 %** | **+23.82 %** |
| **Format Adherence Rate (FAR %)** | **19.44 %** (21/108) | **24.07 %** (26/108) | **+4.63 %** | **+23.82 %** |

* **Maritime Domain (`ship_1`):** Execution accuracy improved from **22.92%** to **27.08%** (+18.2% relative).
* **Order Fulfillment (`tracking_orders`):** Execution accuracy improved from **6.67%** to **8.33%** (+25.0% relative).
* **Automated Failure Taxonomy:** Reduced syntax errors from **80.6%** (87/108) down to **75.9%** (82/108) by suppressing conversational chain-of-thought rambling and enforcing structured markdown SQL output.

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
│   ├── evaluate.py            # Deterministic greedy decoding & 5-class failure taxonomy
│   └── train_grpo.py          # TRL GRPOTrainer script (with 6GB VRAM memory safeguards)
├── tests/
│   ├── conftest.py            # Shared in-memory SQLite fixtures and sample schemas
│   ├── test_executor.py       # Unit tests: enforces read-only, blocks mutations, enforces timeout
│   ├── test_reward.py         # Unit tests: verifies format, syntax, execution, and anti-hacking
│   └── test_leakage.py        # Unit tests: asserts 0 database/schema overlap between train and test
├── notebooks/
│   └── colab_train.ipynb      # Standalone Colab runner
├── docs/
│   ├── RESULTS.md             # Full experimental metrics, deltas, and qualitative case studies
│   ├── EXPERIMENT_LOG.md      # Detailed logs of all baseline and post-training runs
│   ├── LEARNING_NOTES.md      # Technical deep-dives & interview handbook for DP World ML track
│   ├── DECISIONS.md           # Architectural decision records
│   ├── REWARD_DESIGN.md       # Reward function mathematical formulation
│   └── EVAL_PROTOCOL.md       # Evaluation protocol and zero-leakage guidelines
├── PLAN.md                    # Project roadmap and definition of done
└── AGENTS.md                  # Autonomous agent directives and engineering constraints
```

---

## Quickstart & Replication

```bash
# 1. Activate conda environment
conda activate grpo-sql

# 2. Run unit tests (39/39 passing)
pytest tests/ -v

# 3. Run baseline zero-shot evaluation
python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --split test --output_file docs/baseline_eval.json

# 4. Launch GRPO post-training run (Runs locally on consumer GPU or Colab)
python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --max_steps 30 --group_size 4 --lr 1e-5 --save_steps 10 --output_dir checkpoints/grpo_final

# 5. Evaluate GRPO post-trained model
python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter_path checkpoints/grpo_final --split test --output_file docs/grpo_eval.json
```

---

## Documentation & Deep-Dives
* [Experimental Results & Qualitative Analysis](docs/RESULTS.md)
* [Run Tracking & Hyperparameters Log](docs/EXPERIMENT_LOG.md)
* [Interview Preparation Handbook](docs/LEARNING_NOTES.md)
* [Reward Design Specification](docs/REWARD_DESIGN.md)
* [Evaluation Protocol & Taxonomy](docs/EVAL_PROTOCOL.md)
