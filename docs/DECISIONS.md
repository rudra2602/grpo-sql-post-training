# Architecture Decisions Log

This document records every major design decision made during the planning phase of the GRPO text-to-SQL post-training project, including the options considered, the selected path, and the engineering rationale.

---

### **Decision 1: Compute Execution Strategy**
* **Context:** The local machine is Windows with an RTX 4050 (6GB VRAM, no WSL2); free Google Colab provides an NVIDIA T4 (15GB VRAM, Ubuntu Linux).
* **Options Considered:**
  * *Option A (Hybrid):* Local development & testing on Windows; training and generation on Google Colab T4.
  * *Option B (100% Local Windows):* Run 4-bit QLoRA on the local RTX 4050 6GB GPU.
  * *Option C (100% Google Colab):* Develop entirely inside Colab notebooks.
* **Selected Choice:** **Option A (Hybrid Workflow)**
* **Rationale:** Local 6GB VRAM is insufficient for GRPO group rollout generation ($G=4$ or $8$) without high risk of OOM crashes. Developing in Windows native environment introduces severe C++ compilation traps for modern LLM libraries. Option A provides 15GB VRAM and stable Linux for training while maintaining a clean, modular repository locally for technical interviews.

---

### **Decision 2: Base Model Selection**
* **Context:** We require an open-weights model between 0.5B and 1.5B parameters that can learn text-to-SQL without succumbing to the RL "zero-gradient" cold-start trap.
* **Options Considered:**
  * *Option A:* `Qwen/Qwen2.5-Coder-1.5B-Instruct` as primary target, with `Qwen2.5-Coder-0.5B-Instruct` for 10-minute smoke testing.
  * *Option B:* `Qwen2.5-Coder-0.5B-Instruct` only.
  * *Option C:* `meta-llama/Llama-3.2-1B-Instruct`.
* **Selected Choice:** **Option A (`Qwen/Qwen2.5-Coder-1.5B-Instruct` with 0.5B smoke test)**
* **Rationale:** Alibaba’s Qwen2.5-Coder series is the current state-of-the-art for sub-2B code reasoning. It has pre-existing SQL knowledge, ensuring initial completions occasionally execute successfully and provide non-zero advantage gradients. The 1.5B size is impressive for an enterprise internship portfolio, while the 0.5B model guarantees fast pipeline debugging.

---

### **Decision 3: Training Library & Framework**
* **Context:** We need a robust, well-documented implementation of Group Relative Policy Optimization (GRPO) that supports custom Python execution rewards and LoRA.
* **Options Considered:**
  * *Option A:* Hugging Face `trl` (`GRPOTrainer`) with `peft`.
  * *Option B:* Unsloth (Fast GRPO).
  * *Option C:* Custom PyTorch GRPO loop from scratch.
* **Selected Choice:** **Option A (Hugging Face TRL + PEFT)**
* **Rationale:** TRL is the industry-standard post-training library maintained by Hugging Face. Its APIs are stable, natively support custom Python reward callables, and are immediately recognized by interviewers. Unsloth has zero Windows support and frequently experiences dependency breaks on Colab. Writing raw PyTorch risks silent tensor bugs under a 60-hour deadline.

---

### **Decision 4: Domain Scope & Dataset Selection**
* **Context:** The project targets DP World (Group AI/ML), a global container port, terminal operations, and logistics conglomerate.
* **Options Considered:**
  * *Option A:* Curated Supply-Chain & Logistics Suite from Spider schemas (~450–500 train, ~100 held-out test).
  * *Option B:* Random general academic Spider benchmark (sports, cinema, schools).
  * *Option C:* Simple single-table datasets (WikiSQL / `sql-create-context`).
* **Selected Choice:** **Option A (Logistics & Supply Chain Suite with Zero Leakage)**
* **Rationale:** Direct business alignment with DP World. Multi-table schemas (warehouses, shipping, containers, orders) require genuine relational reasoning (JOINs, aggregations). Cross-database evaluation guarantees zero schema leakage.

---

### **Decision 5: Reward Function Design**
* **Context:** GRPO requires a reward signal to score completions. Pure binary matching causes cold-start starvation, while unconstrained rewards encourage reward hacking.
* **Options Considered:**
  * *Option A:* Tiered Composite Reward (0.1 Format + 0.2 Syntax + 0.7 Execution Match with Anti-Hacking).
  * *Option B:* Pure Binary 0/1 Outcome Reward.
  * *Option C:* AST (Abstract Syntax Tree) Component-Matching Reward.
* **Selected Choice:** **Option A (Tiered Composite with Anti-Hacking Guards)**
* **Rationale:** Solves the sparse-reward exploration failure by awarding partial points for valid syntax and formatting. Firm anti-hacking rules (empty result protections, trivial query blocks) prevent exploitation. AST matching was rejected because it penalizes valid alternative SQL queries.

---

### **Decision 6: SQL Execution Safety & Sandboxing**
* **Context:** Model-generated SQL is untrusted code that could modify data, corrupt files, or trigger infinite loops.
* **Options Considered:**
  * *Option A:* Ephemeral In-Memory Copy (`:memory:`) + Read-Only Connection + Static Guard + 1.0s Timeout + 100-Row Truncation.
  * *Option B:* Subprocess Docker Containerization per query.
  * *Option C:* Naive `sqlite3.connect()` with `try/except`.
* **Selected Choice:** **Option A (Safe In-Memory Sandbox)**
* **Rationale:** 100% pure Python standard library (`sqlite3`, `threading`), zero external dependencies, sub-millisecond execution latency. Prevents file modification, blocks cartesian joins from hanging GPU training, and ensures 100% data immutability.

---

### **Decision 7: Baseline Comparison Strategy**
* **Context:** To prove RL lift, we must compare the post-trained model against a fair starting point.
* **Options Considered:**
  * *Option A:* Pure Zero-Shot Base Model (Greedy decoding, $T=0.0$) as primary; SFT as conditional stretch.
  * *Option B:* Mandatory Dual Baseline (Zero-Shot + SFT from start).
  * *Option C:* Few-Shot In-Context Prompting Baseline.
* **Selected Choice:** **Option A (Zero-Shot Base Primary)**
* **Rationale:** Follows the DeepSeek-R1-Zero paradigm of evaluating direct RL reasoning emergence. Keeps execution simple and guarantees baseline numbers are locked in early on Saturday without eating training buffers.

---

### **Decision 8: Evaluation Protocol & Failure Taxonomy**
* **Context:** In an LLM evaluation track, diagnostic granularity is essential.
* **Options Considered:**
  * *Option A:* Multi-Metric Suite (EX%, VSR%, FAR%) + Automated 8-Class Failure Taxonomy using Greedy Decoding ($T=0.0$).
  * *Option B:* Pass@1 Sampling across 3 Random Seeds ($T=0.7$).
  * *Option C:* Flat Binary Execution Accuracy only.
* **Selected Choice:** **Option A (Multi-Metric Suite + Automated 8-Class Taxonomy)**
* **Rationale:** Greedy decoding is completely deterministic and matches Spider's official evaluation protocol. The 8-class taxonomy automatically diagnoses whether errors stem from syntax, joins, hallucinated columns, or empty results, providing high-signal interview talking points.

---

### **Decision 9: Training Hyperparameters & PEFT / LoRA**
* **Context:** Configuration must fit comfortably on a 15GB T4 GPU and complete in under 2 hours without session timeouts.
* **Options Considered:**
  * *Option A:* 16-bit LoRA ($r=16, \alpha=32$), Group Size $G=4$, effective batch size = 8, 200 steps (~70 mins).
  * *Option B:* 4-bit QLoRA with Group Size $G=8$ and 400 steps (~4 hours).
  * *Option C:* Full parameter fine-tuning.
* **Selected Choice:** **Option A (16-bit LoRA, G=4, 200 Steps)**
* **Rationale:** Uses ~8.5 GB VRAM out of 15 GB. Avoids 4-bit quantization latency and numeric instability. Finishes cleanly in ~70 minutes on Colab T4. Full fine-tuning is mathematically impossible on 15GB VRAM.

---

### **Decision 10: Experiment Tracking & Logging**
* **Context:** Need live monitoring during training and permanent preservation of loss/reward curves.
* **Options Considered:**
  * *Option A:* Weights & Biases (W&B) + Local CSV backup.
  * *Option B:* TensorBoard + Local CSV.
  * *Option C:* Plain CSV + Matplotlib scripts.
* **Selected Choice:** **Option A (Weights & Biases + Local CSV)**
* **Rationale:** Built natively into TRL. Protects against Colab disconnects by streaming to the cloud. Allows 1-click export of high-resolution reward curves for the README.

---

### **Decision 11: Reward Ablation & Stretch Goal Strategy**
* **Context:** Testing tiered vs binary reward is high-signal, but could jeopardize the 2.5-day deadline if mandatory.
* **Options Considered:**
  * *Option A:* Conditional Stretch Goal (Sunday afternoon window only).
  * *Option B:* Mandatory Ablation in core Saturday schedule.
  * *Option C:* Drop ablations entirely.
* **Selected Choice:** **Option A (Conditional Stretch Goal)**
* **Rationale:** Script supports `--reward_mode binary`. Only run if primary Base vs. GRPO results are locked in by Sunday noon. Protects the deadline while leaving upside open.

---

### **Decision 12: Repository Architecture & Testing**
* **Context:** Software engineering quality is a key hiring criterion. Untested reward logic can silently ruin RL runs.
* **Options Considered:**
  * *Option A:* Clean Modular Python Package with Automated `pytest` suite.
  * *Option B:* Monolithic single-file script (`train_and_eval.py`).
  * *Option C:* Notebook-only development.
* **Selected Choice:** **Option A (Modular Package + Local `pytest` Suite)**
* **Rationale:** 100% verifiable bug-free reward logic before touching GPU compute. Professional directory layout demonstrates engineering rigor to DP World interviewers.

---

### **Decision 13: Authorized Evaluation Harness v2 Update & Artifact Freezing**
* **Context:** Analysis of initial v1 evaluations revealed that `max_new_tokens=128` caused 82/108 completions to be truncated mid-thought before emitting fenced SQL blocks. The resulting English prose was executed as SQL and incorrectly classified as `SYNTAX_ERROR`. Furthermore, `docs/baseline_eval.json` was missing from the repository, and hardware/precision varied across runs. While `AGENTS.md` normally freezes evaluation code, the user explicitly authorized a limited v2 harness upgrade.
* **Options Considered:**
  * *Option A:* Keep v1 results and attempt to explain the 128-token truncation verbally in interviews without changing code.
  * *Option B:* Freeze v1 evaluation artifacts (`eval_v1_grpo.json`, `eval_v1_5step.json`) and build Evaluation Harness v2 with:
    1. Default `max_new_tokens=512`.
    2. Per-question generated token count and `hit_token_limit` tracking.
    3. A 6th error category `NO_SQL_EXTRACTED` for non-SQL conversational outputs.
    4. Explicit precision/quantization CLI option and metadata tracking.
* **Selected Choice:** **Option B (Freeze v1 as baseline, introduce Evaluation Harness v2)**
* **Rationale:** A fair, rigorous re-evaluation requires identical hardware, identical precision, and sufficient generation tokens (512) so the model is evaluated on its SQL capability rather than token truncation. Freezing v1 preserves historical reproducibility while v2 provides honest, scientifically valid metrics for DP World review.

---

### **Decision 14: Decision Gate — Report Verified v2 Results (Option A) vs. New Training Run (Option B)**
* **Context:** The v2 evaluation revealed that the true base model capability is 58.33% EX (63/108) with 100% format adherence and 87.04% valid SQL. The post-trained GRPO model achieved 60.19% EX (65/108), yielding a +1.85% net gain (+2 queries on `ship_1`) with zero regressions ($b = 0$). An exact McNemar test indicated $p = 0.5000$, showing that this difference is within statistical sampling noise. We must decide whether to report these verified results honestly or attempt an overnight training run (RUN-04).
* **Options Considered:**
  * *Option A (Report Verified Results Honestly):* Document the verified v2 numbers with full statistical rigor (Wilson CIs, exact McNemar test, and failure analysis of the 2 repaired queries).
  * *Option B (Launch RUN-04):* Run 100+ steps of GRPO on the local RTX 4050 (~2+ hours of GPU compute).
* **Selected Choice:** **Option A (Report Verified Results Honestly)**
* **Rationale:**
  1. **Interview Alignment:** In an "LLM Post-Training and Evaluation" track for DP World, diagnosing an evaluation artifact (the 128-token limit), eliminating false claims, computing Wilson confidence intervals, and demonstrating zero regressions ($b=0$) showcases senior-level evaluation engineering and scientific integrity.
  2. **Technical Constraints:** During training, `max_completion_length = 96` and the reward heavily weights format (+0.1) and syntax (+0.2). Because the base model already achieves 100% format adherence and 87% valid SQL, further training without redesigning the reward risks policy degradation or reward hacking.
  3. **Deadline Safety:** Avoids thermal or preemption risks before the hard 7:30 AM IST resume deadline.
