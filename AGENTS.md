# AGENTS.md - Autonomous Agent Directives

> **Notice for Antigravity & AI Assistants:**  
> Antigravity discovers and reads `AGENTS.md` located in the root of the workspace. If needed, a copy or symlink can also be placed in `.gemini/rules/project.md`. Any AI agent or developer operating within this codebase must strictly adhere to the following directives without exception.

---

## 1. Project Goal & Scope
The objective of this project is to post-train `Qwen/Qwen2.5-Coder-1.5B-Instruct` using Group Relative Policy Optimization (GRPO) with a verifiable SQLite execution reward on logistics, supply chain, and container tracking schemas.

This is a portfolio and resume project for an ML internship at DP World ("LLM Post-Training and Evaluation" track). Every experiment, metric, and failure diagnosis must be mathematically rigorous, reproducible, and completely truthful.

---

## 2. Chosen Technology Stack & Versions
* **Language & Runtime:** Python 3.10+ in Conda environment `grpo-sql`.
* **Primary Deep Learning Framework:** PyTorch >= 2.2.0, CUDA 12.1.
* **RL & Post-Training Library:** Hugging Face `trl` (version >= 0.14.0) using `GRPOTrainer` and `GRPOConfig`.
* **Parameter-Efficient Fine-Tuning:** Hugging Face `peft` (version >= 0.10.0) for 16-bit LoRA.
* **Model Transformers:** Hugging Face `transformers` (version >= 4.40.0), `accelerate` (version >= 0.28.0).
* **Database & Sandboxing:** Standard Python `sqlite3`, `threading`.
* **Experiment Tracking:** `wandb` (Weights & Biases) with offline CSV fallback.
* **Testing:** `pytest` >= 8.0.

---

## 3. Strict Engineering Constraints (Non-Negotiable)

### Rule 1: Zero Fabrication of Metrics or Numbers
* **Never invent, hallucinate, simulate, or guess execution accuracy numbers, percentages, runtimes, loss values, or failure taxonomy distributions.**
* If an experiment has not been executed yet, the value in all documentation and code comments must strictly read `TBD`.
* All resume bullets in templates must use explicit placeholders like `[X]%` with the explicit instruction: `"fill with real measured results only"`.

### Rule 2: Immutable Evaluation Protocol & Zero Leakage
* **Never modify the test dataset (`data/test_logistics.json`) or evaluation logic (`src/evaluate.py`) once the baseline evaluation has been recorded.**
* The test set must remain 100% cross-database held-out. The database schemas in `test_logistics.json` must never exist in `train_logistics.json`.
* Both the base model and the GRPO-trained model must be evaluated using identical prompt templates, identical temperature ($T=0.0$), identical timeout limits (1.0s), and identical multiset matching criteria.

### Rule 3: Database & SQL Execution Safety
* **Never execute model-generated SQL directly against a persistent disk database without the sandbox.**
* All executions must pass through `src/executor.py` (`SafeSQLExecutor`):
  * Ephemeral in-memory copy (`:memory:`) or read-only URI mode (`?mode=ro`).
  * Strict regex/AST blocking of mutating commands: `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `DETACH`, `PRAGMA`, `load_extension`, `WRITEFILE`.
  * Hard execution timeout: 1.0 second per query.
  * Maximum row return limit: 100 rows (`cursor.fetchmany(100)`).

### Rule 4: Explicit User Confirmation for Major Changes
* Before changing model architecture, modifying reward tier weights, altering training hyperparameters, or switching execution platforms, you **must ask the user for explicit confirmation**.

### Rule 5: Update Learning Notes After Every Phase
* At the conclusion of each implementation phase, you must update `docs/LEARNING_NOTES.md` with a clear, plain-language explanation of what was built, why it was designed that way, and how to explain it in a technical interview.

---

## 4. Setup & Common Run Commands

```bash
# 1. Activate conda environment
conda activate grpo-sql

# 2. Run local unit tests (Must pass 100% before any training launch)
pytest tests/ -v

# 3. Verify zero schema leakage between train and test
pytest tests/test_leakage.py -v

# 4. Run baseline zero-shot evaluation (Local or Colab)
python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --split test --output_file docs/baseline_eval.json

# 5. Run 10-minute dry-run / smoke test on 0.5B model (Colab)
python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --max_steps 15 --group_size 4

# 6. Launch primary GRPO training run (Colab T4)
python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --max_steps 200 --group_size 4 --lr 1e-5 --output_dir checkpoints/grpo_final

# 7. Evaluate GRPO post-trained model
python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter_path checkpoints/grpo_final --split test --output_file docs/grpo_eval.json
```

---

## 5. Directory Map & Component Responsibilities

```text
├── data/
│   ├── databases/             # Clean SQLite databases (read-only)
│   ├── train_logistics.json   # 406 pairs across 8 schemas
│   └── test_logistics.json    # 108 pairs across 2 held-out schemas
├── src/
│   ├── executor.py            # Sandboxed SQLite execution & timeout logic
│   ├── reward.py              # 3-tier composite reward + anti-hacking checks
│   ├── data_loader.py         # Formats prompts with DDL schemas & user questions
│   ├── evaluate.py            # Greedy decoding & 5-class automated failure taxonomy
│   └── train_grpo.py          # TRL GRPOTrainer script with WandB logging
├── tests/
│   ├── conftest.py            # Shared in-memory SQLite fixtures
│   ├── test_executor.py       # Unit tests for security, sandboxing, and timeouts
│   ├── test_reward.py         # Unit tests for reward math and edge cases
│   └── test_leakage.py        # Enforces zero train-test database overlap
├── notebooks/
│   └── colab_train.ipynb      # Standalone Colab training runner
├── docs/                      # Specification, logs, and interview learning notes
├── README.md                  # Project overview and honest status
├── PLAN.md                    # Schedule, checklists, cut line, definition of done
└── AGENTS.md                  # These guidelines
```

---

## 6. Coding Conventions
* **Type Hints & Docstrings:** Every function in `src/` must include type hints (`def func(query: str) -> float:`) and a concise docstring explaining arguments, return types, and exceptions.
* **Pure Python Standard Library for Safety:** Keep `src/executor.py` free of unnecessary C-extensions; use standard `sqlite3` and `threading`.
* **No Hardcoded Paths:** Always use relative paths or `pathlib.Path(__file__).parent` so code works seamlessly across both Windows and Colab Linux.
