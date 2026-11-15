# Project Execution Plan

**Target Goal:** Post-train `Qwen2.5-Coder-1.5B-Instruct` with Group Relative Policy Optimization (GRPO) using SQLite verifiable execution rewards on logistics schemas, with empirical baseline comparison and failure analysis.  
**Hard Deadline:** Tuesday, 17 Nov 2026, 9:00 AM IST.  
**Start Date:** Sunday, 15 Nov 2026, 6:53 AM IST (~50 hours remaining).  

---

## 1. Backwards Timeline & Milestones

```text
[Now] Sun 15 Nov, 6:53 AM IST
  │
  ├─► Phase 1: Local Setup, Data Curation (Spider Subset), Sandbox, Conftest & Unit Tests 
  │            (Sun 15 Nov: 7:00 AM - 10:30 AM IST) -> Ends with git commit + push checkpoint
  │
  ├─► Phase 2: Colab T4 Baseline Zero-Shot Evaluation on 108 Test Pairs & 5-Class Error Taxonomy
  │            (Sun 15 Nov: 11:00 AM - 2:00 PM IST)
  │
  ├─► Phase 3: Smoke Test (0.5B) & Primary 1.5B GRPO Training (100 steps, max_tokens=96, fp16)
  │            (Sun 15 Nov: 2:30 PM - 6:30 PM IST) -> LoRA checkpoints saved to Google Drive
  │
  ├─► Phase 4: Post-Training Evaluation on 108 Test Pairs & 3 Manual Qualitative Case Studies
  │            (Sun 15 Nov: 7:00 PM - 10:00 PM IST)
  │
  ├─► Buffer / Training Triage / Checkpoint Fallback Evaluation
  │            (Sun 15 Nov: 10:00 PM - Mon 16 Nov: 5:00 PM IST)
  │
  ├─► [THE CUT LINE] Mon 16 Nov, 6:00 PM IST (15 hours remaining before deadline)
  │     ├── IF SUCCESSFUL: Lock metrics; optionally run Phase 5 stretch ablation
  │     └── IF BLOCKED: Enact Fallback Protocol using latest valid checkpoint or baseline diagnosis
  │
  ├─► Phase 6: Final Documentation, Results Table, Empirical Resume Bullets & Repo Finalization
  │            (Mon 16 Nov: 6:00 PM - 11:30 PM IST)
  │
  ├─► Phase 7: Buffer, Final Validation & Application Form Submission
  │            (Tue 17 Nov: 7:00 AM - 8:30 AM IST)
  │
[DEADLINE] Tue 17 Nov, 9:00 AM IST
```

---

## 2. Detailed Phase Task Checklists

### **Phase 1: Environment, Safe Sandbox, Reward Engine & Unit Tests**
* **Time Window:** Sunday, 15 Nov, 7:00 AM – 10:30 AM IST
* **Goal:** Build the complete local infrastructure on Windows and verify 100% passing tests before touching a GPU.
- [ ] Create clean local conda environment `grpo-sql` with Python 3.10.
- [ ] `[Create] scripts/prepare_data.py`:
  - Downloads 10 verified Spider SQLite databases from Hugging Face (`Chinastark/spider_datasets`) and query pairs from `xlangai/spider`.
  - Filters out queries returning 0 rows or syntax errors (retaining only queries returning $\ge 1$ rows).
  - Generates `[Create] data/train_logistics.json` (406 pairs across 8 schemas: `customers_and_invoices`, `department_store`, `customers_and_addresses`, `product_catalog`, `store_product`, `ship_mission`, `products_for_hire`, `customer_deliveries`).
  - Generates `[Create] data/test_logistics.json` (108 pairs across 2 held-out schemas: `tracking_orders`, `ship_1`).
  - JSON record contract: `{"db_id": str, "question": str, "gold_sql": str, "difficulty": str}`.
- [ ] `[Create] src/data_loader.py`:
  - Implements `format_sql_prompt(schema_ddl: str, question: str) -> str`.
  - Formats prompt with system instructions, SQLite DDL schema, question, and markdown code fence (```` ```sql\n...\n``` ````).
- [ ] `[Create] src/executor.py` (`SafeSQLExecutor`):
  - In-memory database loading (`:memory:`).
  - `set_authorizer` callback blocking mutating commands: `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `DETACH`, `PRAGMA`, `load_extension`, `WRITEFILE`.
  - 1.0s timeout mechanism via `threading.Timer` with `conn.interrupt()` and `set_progress_handler`.
  - Truncation at 100 rows (`cursor.fetchmany(100)`).
  - Method signature: `execute(db_path: str, sql: str, timeout: float = 1.0) -> Tuple[bool, List[Tuple], Optional[str]]`.
- [ ] `[Create] src/reward.py` (`compute_sql_reward`):
  - Format score (+0.1 for markdown SQL tags).
  - Syntax/execution score (+0.2 for error-free run).
  - Execution match score (+0.7 for multiset row equivalence).
  - Anti-hacking logic: if gold result returns $\ge 1$ row, empty predicted result returns 0.0 match credit; queries lacking table references return 0.0.
  - Function signature: `compute_sql_reward(predicted_sql: str, gold_sql: str, db_path: str) -> float`.
- [ ] `[Create] tests/conftest.py`:
  - Shared in-memory SQLite fixtures with sample logistics tables and mock question-SQL pairs.
- [ ] `[Create] tests/test_executor.py`:
  - (1) Valid SELECT returns rows; (2) Blocked mutations raise security exception; (3) Runaway query terminates within 1.0s; (4) Truncates at 100 rows.
- [ ] `[Create] tests/test_reward.py`:
  - (1) Exact multiset match scores 1.0; (2) Format only scores 0.1; (3) Clean execution with wrong rows scores 0.3; (4) Unordered row equivalence scores 1.0; (5) Empty result against non-empty gold scores 0.0.
- [ ] `[Create] tests/test_leakage.py`:
  - Asserts `set(train_db_ids).isdisjoint(set(test_db_ids))`.
- [ ] Run `pytest tests/ -v` locally on Windows and achieve **100% pass rate**.
- [ ] **Local-to-Cloud Git Checkpoint:** Initialize git, commit all Phase 1 files, and push to GitHub repository (`git add .`, `git commit -m "feat: complete phase 1 safe sandbox and reward engine"`, `git push origin main`).
* **Interview Checkpoint:** *"Can I explain why we need an in-memory copy, set_authorizer security guard, and anti-hacking reward logic?"*

---

### **Phase 2: Colab Setup & Baseline Zero-Shot Evaluation**
* **Time Window:** Sunday, 15 Nov, 11:00 AM – 2:00 PM IST
* **Goal:** Establish the ground-truth baseline on the 108 held-out test questions using deterministic greedy decoding.
- [ ] Set up Google Colab session with free T4 GPU (15 GB VRAM).
- [ ] `[Create] notebooks/colab_train.ipynb`:
  - Clones repository from GitHub and installs dependencies (`trl`, `peft`, `transformers`, `wandb`, `accelerate`).
- [ ] `[Create] src/evaluate.py`:
  - CLI arguments: `--model_path <path> [--adapter_path <path>] --split test --output_file <path>`.
  - Evaluates greedy decoding ($T=0.0$) on `data/test_logistics.json` (108 pairs) in `torch.float16`.
  - Computes Execution Accuracy (EX), Valid SQL Rate (VSR), Format Adherence Rate (FAR).
  - Automatically classifies all failures into the **5-class deterministic taxonomy**:
    1. `SYNTAX_ERROR`: SQLite parser rejection / ill-formed SQL.
    2. `SCHEMA_HALLUCINATION`: References non-existent tables or columns.
    3. `TIMEOUT_ABORT`: Execution exceeded 1.0s limit.
    4. `EMPTY_RESULT`: Query executed cleanly but returned 0 rows against non-empty gold.
    5. `VALUE_MISMATCH`: Clean execution with non-empty rows differing from gold multiset.
  - Outputs structured artifact `[Create] docs/baseline_eval.json`.
- [ ] Run baseline evaluation for `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
- [ ] `[Modify] docs/RESULTS.md`: Populate Baseline column with **real measured numbers**.
* **Interview Checkpoint:** *"Can I explain the baseline performance and the top failure modes of the un-tuned model?"*

---

### **Phase 3: Pipeline Smoke Test & Primary GRPO Training**
* **Time Window:** Sunday, 15 Nov, 2:30 PM – 6:30 PM IST
* **Goal:** Verify end-to-end training stability on a tiny model, then complete the primary 1.5B GRPO run.
- [ ] `[Create] src/train_grpo.py`:
  - `torch_dtype=torch.float16`, `GRPOConfig(..., use_vllm=False)`.
  - Rollout generation bounds: `max_new_tokens=96`, start at `max_steps=100`.
  - 16-bit LoRA: $r=16, \alpha=32$, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`, `lora_dropout=0.05`.
  - Group size $G=4$, per_device_batch_size=1, gradient_accumulation=8 (effective batch size=8 prompts, 32 completions per step).
  - Learning rate = $1\times 10^{-5}$, KL $\beta = 0.005$.
  - Rollout exploration sampling: `temperature=0.7`, `top_p=0.9`.
  - Dataset schema: `Dataset.from_dict({"prompt": prompts, "gold_sql": gold_sqls, "db_path": db_paths})`.
  - Reward callable wrapper signature:
    ```python
    def grpo_sql_reward_func(completions, gold_sql, db_path, **kwargs) -> List[float]: ...
    ```
  - Logs the proportion of rollout groups with zero reward variance ($\text{std}(R)=0$) to WandB.
  - Configures LoRA checkpoint saving every 25–50 steps directly to Google Drive.
- [ ] **10-Minute Smoke Test:** Run 15 steps on `Qwen2.5-Coder-0.5B-Instruct` in Colab.
  - Verify that completions generate without OOM in `float16`.
  - Verify that reward function returns valid floats.
  - Measure empirical seconds/step: calibrate final 1.5B step count to guarantee completion in ~90–120 minutes.
  - Verify that WandB logs loss, mean reward, and KL divergence.
- [ ] **Launch Primary GRPO Run:** `Qwen2.5-Coder-1.5B-Instruct`
  - Run for calibrated step count (~100 steps).
  - Monitor live reward curves, advantage distributions, and KL divergence on Weights & Biases.
  - Save final LoRA adapter checkpoint to Google Drive and local machine (`checkpoints/grpo_final/`).
* **Interview Checkpoint:** *"Can I explain group relative advantage normalization, policy clipping, and why KL beta prevents collapse?"*

---

### **Phase 4: Post-Training Evaluation & Error Taxonomy Analysis**
* **Time Window:** Sunday, 15 Nov, 7:00 PM – 10:00 PM IST
* **Goal:** Measure post-training gains on the held-out test set and analyze error shifts.
- [ ] Run `src/evaluate.py` on the trained GRPO model on the same 108 test questions ($T=0.0$):
  ```bash
  python src/evaluate.py --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter_path checkpoints/grpo_final --split test --output_file docs/grpo_eval.json
  ```
- [ ] Calculate real metric deltas:
  - $\Delta \text{EX} = \text{EX}_{\text{GRPO}} - \text{EX}_{\text{Base}}$
  - $\Delta \text{VSR} = \text{VSR}_{\text{GRPO}} - \text{VSR}_{\text{Base}}$
  - $\Delta \text{FAR} = \text{FAR}_{\text{GRPO}} - \text{FAR}_{\text{Base}}$
- [ ] Classify all failures using the 5-class automated failure taxonomy.
- [ ] `[Modify] docs/RESULTS.md`: Log before-and-after failure breakdown with real measured numbers.
- [ ] Conduct **3 manual qualitative case studies** analyzing relational root causes (`JOIN_ERROR`, `CONDITION_ERROR`, `AGGREGATION_ERROR`) on queries where base and GRPO predictions diverged.
- [ ] **Checkpoint Recovery Rule:** If primary training was interrupted or sub-converged, evaluate the latest healthy intermediate checkpoint (e.g. step 50, 75) before declaring training failure.
* **Interview Checkpoint:** *"Can I explain which specific failure modes GRPO eliminated and which ones remain unsolved?"*

---

### **The Cut-Line: Monday, 16 Nov, 6:00 PM IST**
* **Success Criteria:** Verified Base and GRPO evaluation results are recorded in `docs/RESULTS.md`.
* **If Successful:** Proceed to Phase 5 (Optional Stretch Ablation) or transition directly into final report drafting.
* **If Blocked / Training Failed:** Stop training attempts immediately. Switch to **Fallback Protocol**:
  1. Freeze the baseline evaluation results and the 5-class error taxonomy.
  2. If an intermediate checkpoint exists (e.g. step 25 or 50), evaluate it and record partial adaptation metrics.
  3. Document the exact training failure dynamics (e.g. reward collapse, KL explosion, or VRAM bottleneck) using W&B diagnostic charts.
  4. Frame the project as a comprehensive evaluation and RL post-training diagnostic study for enterprise text-to-SQL.
  5. Never fabricate numbers.

---

### **Phase 5: Conditional Stretch Goal (Reward Ablation)**
* **Time Window:** Monday, 16 Nov, 12:00 PM – 4:00 PM IST
* **Trigger:** Only executed if primary Phase 4 results are locked and documented before Monday noon.
- [ ] Launch ablation run: GRPO with binary outcome reward (`--reward_mode binary`).
- [ ] Run for 100 steps on Colab T4.
- [ ] Evaluate on test set; compare learning speed and final accuracy against tiered composite reward.
- [ ] If elapsed time exceeds schedule allocation, omit Phase 5.

---

### **Phase 6: Documentation, Real Resume Bullets & Repo Finalization**
* **Time Window:** Monday, 16 Nov, 6:00 PM – 11:30 PM IST
* **Goal:** Convert all measured results into professional documentation and resume bullets.
- [ ] Export high-resolution training curves from Weights & Biases (Reward growth, Loss, KL divergence) into `docs/assets/`.
- [ ] `[Modify] README.md`: Populate with real measured numbers, comparison tables, and failure analysis charts.
- [ ] `[Modify] docs/RESULTS.md`: Populate final summary and draft 3 measured resume bullet points using only verified numbers.
- [ ] `[Modify] docs/LEARNING_NOTES.md`: Complete with interview question answers.
- [ ] Perform a clean git commit and push to GitHub.
* **Interview Checkpoint:** *"Can I walk through my entire repository and explain every file from top to bottom?"*

---

### **Phase 7: Buffer, Final Validation & Application Form Submission**
* **Time Window:** Tuesday, 17 Nov, 7:00 AM – 8:30 AM IST (Deadline is 9:00 AM IST)
* **Goal:** Final review and verification.
- [ ] Verify GitHub repository is public, clean, and all relative links work.
- [ ] Verify no placeholder text or `[X]%` remains in submission text.
- [ ] Submit internship application form before 8:30 AM IST (30-minute safety buffer).

---

## 3. Risks & Contingency Plan

| Potential Risk | Severity | Mitigation / Contingency Plan |
|---|---|---|
| Colab session disconnects during training | Medium | Training script saves checkpoints every 25–50 steps directly to Google Drive; W&B preserves all metrics in the cloud. |
| Zero-gradient reward collapse in GRPO | High | Tiered composite reward awards +0.1 for format and +0.2 for valid SQL, providing partial reward signals to promote non-zero gradient variance when execution fails early; share of zero-variance groups is logged. |
| Runaway cartesian joins blocking rollout thread | High | `SafeSQLExecutor` strictly aborts any query taking >1.0s via `conn.interrupt()` and progress handler; rows truncated at 100. |
| Schema data leakage between train & test | High | `test_leakage.py` unit test strictly enforces zero database name or table overlap between training (8 schemas) and test sets (2 schemas). |
| Time runs out before training finishes | Critical | The Monday 6:00 PM Cut-Line guarantees pivot to checkpoint recovery or diagnostic evaluation report if training stalls. |

---

## 4. Definition of Done
The project is officially complete when:
1. All local unit tests (`pytest tests/`) pass with 0 failures.
2. Baseline model (`Qwen2.5-Coder-1.5B-Instruct` zero-shot) has been evaluated on the held-out 108 test questions ($T=0.0$) with real numbers recorded.
3. GRPO model has been trained and evaluated on the same 108 test questions with real numbers recorded.
4. A 5-class failure taxonomy breakdown is documented comparing Base vs. GRPO, supplemented by 3 qualitative case studies.
5. All markdown files (`README.md`, `PLAN.md`, `docs/*`) are fully populated with real numbers (no fake numbers, no unpopulated placeholders).
6. Application form is submitted before Tuesday, 17 Nov 2026, 9:00 AM IST.
