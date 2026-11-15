# Experiment Tracking Log

> **Strict Honesty Rule:** All cells in this document must remain `TBD` until a real experiment run is executed and measured. Never write placeholder numbers or estimated values.

---

## 1. Master Experiment Summary Table

| Run ID | Date & Time | Model Name | Training Method | Group Size ($G$) | LR / Schedule | Steps | Precision | Hardware | Runtime (min) | Test EX (%) | Test VSR (%) | WandB Run URL | Status / Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `RUN-00-BASE` | TBD | `Qwen2.5-Coder-1.5B-Instruct` | Zero-Shot Greedy ($T=0.0$) | N/A | None | 0 | 16-bit | Colab T4 | TBD | TBD | TBD | N/A | Ground-truth baseline run |
| `RUN-01-SMOKE` | TBD | `Qwen2.5-Coder-0.5B-Instruct` | GRPO (16-bit LoRA) | 4 | 1e-5 / Cosine | 15 | 16-bit | Colab T4 | TBD | TBD | TBD | TBD | 10-minute pipeline smoke test |
| `RUN-02-PRIMARY` | TBD | `Qwen2.5-Coder-1.5B-Instruct` | GRPO (Tiered Composite) | 4 | 1e-5 / Cosine | 200 | 16-bit | Colab T4 | TBD | TBD | TBD | TBD | Primary post-training run |
| `RUN-03-STRETCH` | TBD | `Qwen2.5-Coder-1.5B-Instruct` | GRPO (Pure Binary Reward) | 4 | 1e-5 / Cosine | 200 | 16-bit | Colab T4 | TBD | TBD | TBD | TBD | Conditional Sunday ablation run |

---

## 2. Detailed Run Sheets

### Run `RUN-00-BASE` (Baseline Zero-Shot)
* **Execution Date:** TBD
* **Command:** `python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --split test`
* **Decoding:** Temperature = 0.0, Greedy, max_tokens = 192
* **Measured Metrics:**
  * Execution Accuracy (EX): `TBD %`
  * Valid SQL Rate (VSR): `TBD %`
  * Format Adherence Rate (FAR): `TBD %`
* **Top Failure Modes Diagnosed:**
  1. TBD
  2. TBD
  3. TBD
* **Notes:** TBD

---

### Run `RUN-01-SMOKE` (Dry-Run / Smoke Test)
* **Execution Date:** TBD
* **Command:** `python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-0.5B-Instruct --max_steps 15 --group_size 4`
* **Objective:** Verify loss computation, reward function callable, advantage normalization, and WandB logging without crashing.
* **Peak VRAM:** `TBD GB`
* **Average Step Time:** `TBD s`
* **Outcome:** TBD

---

### Run `RUN-02-PRIMARY` (Primary GRPO Post-Training)
* **Execution Date:** TBD
* **Command:** `python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --max_steps 200 --group_size 4 --lr 1e-5 --beta 0.005 --reward_mode tiered`
* **Hyperparameters:**
  * LoRA: $r=16, \alpha=32$, target: all-linear
  * Optimizer: AdamW, LR = $1\times 10^{-5}$, Cosine schedule, Warmup = 10%
  * KL Beta: $0.005$
  * Effective Batch Size: 8 prompts (32 completions per step)
* **Training Dynamics:**
  * Initial Mean Reward (Step 1): `TBD`
  * Final Mean Reward (Step 200): `TBD`
  * Peak KL Divergence: `TBD`
  * Total Runtime: `TBD minutes`
* **Evaluation Results (Test Set, Greedy $T=0.0$):**
  * Execution Accuracy (EX): `TBD %` (Delta: `+TBD %`)
  * Valid SQL Rate (VSR): `TBD %` (Delta: `+TBD %`)
  * Format Adherence (FAR): `TBD %` (Delta: `+TBD %`)
* **Notes:** TBD

---

### Run `RUN-03-STRETCH` (Ablation: Pure Binary Reward)
* **Execution Date:** TBD
* **Command:** `python src/train_grpo.py --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct --max_steps 200 --group_size 4 --lr 1e-5 --reward_mode binary`
* **Research Question:** Does removing the intermediate syntax reward (+0.2) slow down exploration or cause zero-gradient stagnation?
* **Evaluation Results:**
  * Execution Accuracy: `TBD %`
  * Valid SQL Rate: `TBD %`
* **Notes:** TBD (Executed only if time permits on Sunday).
