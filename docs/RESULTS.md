# Experimental Results & Performance Analysis (Evaluation Harness v2)

> **Strict Honesty & Scientific Integrity Protocol:**  
> All metrics, counts, confidence intervals, and statistical tests in this report are verified, real measured values from completed benchmark evaluation runs on `data/test_logistics.json` saved in `docs/eval_v2_base.json`, `docs/eval_v2_grpo.json`, and `docs/comparison_v2.json`. Zero numbers were fabricated, estimated, or extrapolated.

---

## 1. Primary Benchmark Results

Evaluated on `data/test_logistics.json` (108 held-out test questions across Spider schemas `ship_1` and `tracking_orders`) on an **NVIDIA GeForce RTX 4050 Laptop GPU** in **16-bit (`float16`) precision** using **deterministic greedy decoding ($T=0.0$, `do_sample=False`)**, `max_new_tokens = 512`, hard 1.0s sandbox timeout, and multiset row matching.

| Metric | Base Model (`Qwen2.5-Coder-1.5B-Instruct`) | Post-Trained Model (GRPO LoRA Adapter) | Absolute Delta ($\Delta$) | Wilson 95% Confidence Intervals | Statistical Significance |
|---|---|---|---|---|---|
| **Execution Accuracy (EX %)** | **58.33 %** (63 / 108) | **60.19 %** (65 / 108) | **+1.85 %** (+2 queries) | Base: [48.9%, 67.2%]<br>GRPO: [50.8%, 68.9%] | $p = 0.5000$ (McNemar exact; within noise) |
| **Valid SQL Rate (VSR %)** | **87.04 %** (94 / 108) | **87.04 %** (94 / 108) | **0.00 %** (exact parity) | Base: [79.4%, 92.1%]<br>GRPO: [79.4%, 92.1%] | Identical valid SQL rates |
| **Format Adherence Rate (FAR %)** | **100.00 %** (108 / 108) | **100.00 %** (108 / 108) | **0.00 %** (exact parity) | Base: [96.6%, 100.0%]<br>GRPO: [96.6%, 100.0%] | Identical format adherence |

### Generation & Token Limit Profile
* **Average Completion Length:** 246.0 tokens (Base) vs. 238.9 tokens (GRPO).
* **Token Limit Truncation ($\ge 512$ tokens):** 1 / 108 (0.93%) for Base vs. 2 / 108 (1.85%) for GRPO.
* **Accuracy Conditional on Emitted SQL:** 58.33% (Base: 63/108) vs. 60.19% (GRPO: 65/108), as both models emitted markdown SQL blocks for 100% of test prompts.

### Per-Database Performance
* **`ship_1` (Maritime vessels & captains, 48 test questions):**
  * Base Model: **50.00 %** (24 / 48 correct)
  * GRPO Model: **54.17 %** (26 / 48 correct) $\to$ **+4.17 % absolute gain** (+2 queries)
* **`tracking_orders` (Shipment fulfillment & orders, 60 test questions):**
  * Base Model: **65.00 %** (39 / 60 correct)
  * GRPO Model: **65.00 %** (39 / 60 correct) $\to$ **0.00 % delta** (exact parity across all 60 questions)

---

## 2. Automated 6-Class Failure Taxonomy Breakdown

Breakdown of error causes across all failed queries on the 108 held-out test questions:

| Failure Taxonomy Category | Base Model Error Count | GRPO Model Error Count | Shift | Category Diagnosis |
|---|---|---|---|---|
| `NO_SQL_EXTRACTED` | **0** (0.0%) | **0** (0.0%) | **0** | Conversational prose with no fenced SQL and no SQL starting keyword |
| `SYNTAX_ERROR` | **1** (0.9%) | **1** (0.9%) | **0** | Malformed SQLite syntax or parser failure |
| `SCHEMA_HALLUCINATION` | **13** (12.0%) | **13** (12.0%) | **0** | References to non-existent tables or columns in the schema |
| `TIMEOUT_ABORT` | **0** (0.0%) | **0** (0.0%) | **0** | Runaway executions exceeding the 1.0s sandbox limit |
| `EMPTY_RESULT` | **8** (7.4%) | **6** (5.6%) | **-2** | Logically over-constrained queries returning 0 rows against non-empty ground truth |
| `VALUE_MISMATCH` | **23** (21.3%) | **23** (21.3%) | **0** | Executed cleanly, but row values or column projections differed from gold multiset |
| **Total Failures** | **45 / 108** (41.67%) | **43 / 108** (39.81%) | **-2** | Net reduction of 2 failed queries |

> [!NOTE]
> **Key Finding on Failure Causes:**
> At 512 tokens, `SYNTAX_ERROR` accounts for only **0.9%** (1 query) of outcomes for both models. The pre-trained base model already has high SQL syntax capability. The real failure modes are semantic: **`VALUE_MISMATCH` (21.3%)** (e.g., incorrect filter thresholds or projection ordering) and **`SCHEMA_HALLUCINATION` (12.0%)** (referencing non-existent columns).

---

## 3. Paired 2x2 Contingency Matrix & Statistical Significance

Because both models were evaluated greedily ($T=0.0$) on the exact same 108 test questions on identical hardware, we performed a paired contingency analysis:

| Contingency Outcome | Count | Percentage | Description |
|---|---|---|---|
| **Both Models Correct** | **63** | 58.33 % | Identical correct outputs on identical questions |
| **Only Base Correct ($b$)** | **0** | 0.00 % | Regressions introduced by GRPO post-training |
| **Only GRPO Correct ($c$)** | **2** | 1.85 % | Queries repaired by GRPO post-training |
| **Neither Model Correct** | **43** | 39.81 % | Queries failed by both models |
| **Total Evaluated** | **108** | 100.00 % | Total held-out test questions |

### Statistical Test (Exact McNemar's Test):
* **Discordant Pairs ($b + c$):** 2 queries (Index 40 and Index 42).
* **Discordance Asymmetry:** $b = 0$, $c = 2$.
* **Exact Two-Sided McNemar $p$-value:**
  $$p = 2 \times \binom{2}{0} (0.5)^2 = 0.5000$$
* **Conclusion:** Because $p = 0.5000 > 0.05$ and the Wilson 95% confidence intervals overlap heavily ([48.9%, 67.2%] vs. [50.8%, 68.9%]), **the +1.85% net gain cannot be claimed as statistically significant**. It is within expected sampling noise for a 108-question test set. However, the GRPO model achieved this gain with zero regressions ($b = 0$).

---

## 4. Deep-Dive: The Two Shifted Queries (Index 40 and Index 42)

The 2 queries where GRPO successfully corrected base model errors both flipped from `EMPTY_RESULT` to exact execution match:

### Shift 1: Subquery Negation vs. Invalid Inner Join on Null (Index 40)
* **Database:** `ship_1` (Maritime vessels)
* **Question:** *"what are the names and classes of the ships that do not have any captain yet?"*
* **Gold SQL:**
  ```sql
  SELECT name , CLASS FROM ship WHERE ship_id NOT IN (SELECT ship_id FROM captain)
  ```
* **Base Model Output (Failed with `EMPTY_RESULT`):**
  ```sql
  SELECT T1.Name , T1.Class FROM Ship AS T1 INNER JOIN Captain AS T2 ON T1.Ship_ID = T2.Ship_ID WHERE T2.Captain_ID IS NULL
  ```
  * *Failure Diagnosis:* The base model performed an `INNER JOIN` between `Ship` and `Captain`, then checked `WHERE T2.Captain_ID IS NULL`. Because inner joins discard unmatched rows, the result was 0 rows.
* **GRPO Model Output (Correct - 1.0 Reward):**
  ```sql
  SELECT Name, Class FROM Ship WHERE Ship_ID NOT IN (SELECT Ship_ID FROM Captain)
  ```
  * *Analysis:* The policy formulated an uncorrelated `NOT IN` subquery that correctly handled unassigned relationships, returning the exact gold multiset.

---

### Shift 2: Order By Limit vs. Impossible Minimum Comparison (Index 42)
* **Database:** `ship_1` (Maritime vessels)
* **Question:** *"Find the name of the ship that is steered by the youngest captain."*
* **Gold SQL:**
  ```sql
  SELECT t1.name FROM ship AS t1 JOIN captain AS t2 ON t1.ship_id = t2.ship_id ORDER BY t2.age LIMIT 1
  ```
* **Base Model Output (Failed with `EMPTY_RESULT`):**
  ```sql
  SELECT T2.Name FROM captain AS T1 INNER JOIN ship AS T2 ON T1.Ship_ID = T2.Ship_ID WHERE T1.age < (SELECT MIN(age) FROM captain);
  ```
  * *Failure Diagnosis:* The base model wrote `WHERE age < MIN(age)`. Because no value can be strictly less than the minimum value in the table, the query returned 0 rows.
* **GRPO Model Output (Correct - 1.0 Reward):**
  ```sql
  SELECT T2.Name FROM captain AS T1 JOIN Ship AS T2 ON T1.Ship_ID = T2.Ship_ID ORDER BY T1.age ASC LIMIT 1
  ```
  * *Analysis:* The policy formulated `ORDER BY age ASC LIMIT 1`, perfectly resolving the extreme value query.

---

## 5. Verified Numbers Table (Honest Resume Reference)

The following table provides verified, unexaggerated metrics for documentation and technical interviews:

| Parameter | Base Model | GRPO Post-Trained Model | Delta / Shift |
|---|---|---|---|
| **Base Model Backbone** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Same architecture |
| **Evaluation Precision** | 16-bit (`float16`) | 16-bit (`float16`) | Identical |
| **Evaluation Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (6GB) | NVIDIA GeForce RTX 4050 Laptop GPU (6GB) | Identical |
| **Decoding Strategy** | Greedy ($T=0.0$, `max_new_tokens=512`) | Greedy ($T=0.0$, `max_new_tokens=512`) | Identical |
| **Test Questions Evaluated** | 108 (Spider `ship_1` & `tracking_orders`) | 108 (Spider `ship_1` & `tracking_orders`) | Cross-database held-out |
| **Execution Accuracy (EX)** | **58.33 %** (63 / 108) | **60.19 %** (65 / 108) | **+1.85 %** (+2 queries) |
| **Wilson 95% Confidence Interval** | **[48.9 %, 67.2 %]** | **[50.8 %, 68.9 %]** | Overlapping intervals |
| **Valid SQL Rate (VSR)** | **87.04 %** (94 / 108) | **87.04 %** (94 / 108) | 0.00 % (exact parity) |
| **Format Adherence Rate (FAR)** | **100.00 %** (108 / 108) | **100.00 %** (108 / 108) | 0.00 % (exact parity) |
| **Discordant Pairs ($b, c$)** | N/A | $b = 0$, $c = 2$ | 0 regressions, +2 repaired |
| **Exact McNemar $p$-value** | N/A | $p = 0.5000$ | Not statistically significant |
| **RL Rollouts in Training** | N/A | 480 completions (30 steps $\times$ 16 comp/step) | Primary run (`RUN-03`) |
| **Peak Training VRAM** | N/A | 5.1 GB (4-bit NF4 QLoRA, chunked logprobs 512) | Local consumer GPU |
