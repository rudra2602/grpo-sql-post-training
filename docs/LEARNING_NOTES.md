# Interview Learning Notes & Technical Deep-Dives

This document is your personal technical handbook for interviews with DP World (Group AI/ML). It breaks down every component of this project into simple, plain-language concepts, explaining **what** was built, **why** it was designed that way, and **how** to answer interview questions about it.

---

## Phase 1: Environment, Safe Sandbox & Reward Design

### 1. What is an Execution-Based Verifiable Reward?
* **Plain Language:** Instead of asking a human or another AI model (LLM-as-a-judge) if a SQL query looks good, we physically run the query on a real SQLite database. If the query runs and returns the exact same data as the reference answer, we give it a high reward.
* **Why it matters:** LLM-as-a-judge is slow, expensive, and can be fooled by confident-sounding queries. Execution against a database is instantaneous, deterministic, and 100% objective ground truth.

### 2. Why do we need an In-Memory Sandbox with Progress Timeouts?
* **Plain Language:** During RL training, the model generates thousands of wild, experimental queries. If a query contains `DROP TABLE`, it could erase the database. If it accidentally does a runaway join (`SELECT * FROM A, B, C, D`), it could freeze the computer forever trying to combine billions of rows.
* **Interview Talking Point:**  
  *"We built a zero-mutation sandbox. We load the database into an ephemeral in-memory (`:memory:`) SQLite instance or strict read-only URI mode. We use a static pre-execution regex guard to block mutating keywords like `DROP` or `PRAGMA`. Crucially, we attach SQLite's `set_progress_handler` to abort any query exceeding 1.0 second, completely preventing runaway cartesian joins from hanging GPU training."*

### 2b. How is `src/executor.py` implemented and why are there 5 layers of defense?
* **Plain Language:** Model-generated SQL is untrusted executable code. We built `SafeSQLExecutor` with 5 concentric defense layers:
  1. **Static Regex Guard:** Instantly rejects dangerous keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `DETACH`, `PRAGMA`, `load_extension`, `WRITEFILE`) before SQLite even touches the string.
  2. **Ephemeral In-Memory Cloning:** The source database is opened in strict read-only URI mode (`?mode=ro`) and cloned into an isolated RAM copy (`:memory:`) using SQLite's native `backup()` API. The underlying disk database is never modified or locked.
  3. **SQLite AST Authorizer Hook (`set_authorizer`):** Even if a dangerous command slips past regex (e.g. through obfuscated SQL or subquery injection), SQLite's internal AST compiler invokes our authorizer callback before executing any statement. We explicitly return `SQLITE_DENY` for 24 mutating action codes.
  4. **Dual Timeout Watchdog (1.0s limit):** Runaway cartesian products (e.g. `SELECT * FROM A, B, C, D`) can stall GPU training forever. We pair SQLite's `set_progress_handler` (which counts opcode instructions and interrupts long executions) with a concurrent `threading.Timer` calling `conn.interrupt()`.
  5. **Result Truncation (`fetchmany(100)`):** We strictly cap returned rows at 100 to prevent multi-gigabyte memory spikes from exploding process RAM.
* **Interview Talking Point:**
  *"In production RL post-training, unconstrained query execution can hang GPU worker threads or exhaust host memory. Our `SafeSQLExecutor` implements defense-in-depth: static regex pre-filtering, ephemeral in-memory cloning, a C-level SQLite authorizer hook, dual opcode/thread timeouts, and 100-row result truncation. In unit tests, this stopped runaway recursive queries within 0.5s and verified 100% disk immutability."*

### 3. What is Reward Hacking and how did we prevent it?
* **Plain Language:** An RL agent will always exploit any mathematical loophole to get points without doing the real work. For example, if a question has an empty answer (`0 rows found`), a lazy model that outputs `SELECT * WHERE 1=0;` would match and get full points!
* **Interview Talking Point:**  
  *"We implemented three guards: first, empty-set matches do not get full accuracy credit unless the predicted query references legitimate prompt tables. Second, trivial queries like `SELECT 1;` are rejected unless they query real tables. Third, our tiered reward (+0.1 format, +0.2 syntax, +0.7 execution match) ensures that learning syntax provides early guidance without diluting the primary incentive for exact execution correctness."*

### 4. What does `scripts/prepare_data.py` do and why is data curation critical?
* **Plain Language:** It downloads 10 logistics and supply-chain SQLite databases and query pairs from Spider. It tests every single gold SQL query against the real database. If any query returns zero rows or causes an error, it is thrown out. Then, it classifies difficulty and splits the dataset into 406 training pairs (8 databases) and 108 test pairs (2 databases), ensuring the test databases are 100% unseen.
* **Why it matters:**
  1. **Zero-Row Elimination:** If a reference query returns 0 rows, a model could output a completely bogus query with an unsatisfiable filter (`WHERE 1=0`) and accidentally match the empty set! Retaining only queries returning $\ge 1$ rows eliminates this trivial exploitation vector. In our run, exactly 2 queries in `store_product` returned 0 rows and were pruned (reducing 44 to 42 valid pairs).
  2. **Cross-Database Generalization:** In enterprise logistics (e.g. DP World container terminals vs. freight invoices), schemas change across business units. The test databases (`tracking_orders`, `ship_1`) share zero tables with the training set. The model must learn generalized relational reasoning from the DDL schema prompt rather than memorizing schema names.

---

## Phase 2: Baselines & Evaluation Rigor

### 1. What is Cross-Database Zero-Leakage Evaluation?
* **Plain Language:** If you test a model on the same database tables it trained on, it might just memorize the column names instead of learning how to write SQL. We strictly separate the databases so the test set schemas were never seen during training.
* **Interview Talking Point:**  
  *"We enforce a strict cross-database split adapted from the Spider benchmark. The test databases (`tracking_orders`, `ship_1`) have zero table or schema overlap with our training databases. This proves the model acquired generalized relational reasoning rather than database-specific memorization."*

### 2. Why use Greedy Decoding ($T=0.0$) instead of Sampling ($T=0.7$)?
* **Plain Language:** Setting temperature to 0 means the model always picks the single most probable token. There is zero randomness.
* **Why it matters:** If you evaluate with random sampling, your accuracy can fluctuate by $\pm 4\%$ just from lucky rolls of the dice. Greedy decoding makes every evaluation 100% deterministic and reproducible, which is the official standard for benchmarks like Spider.

### 3. How does the Automated 5-Class Failure Taxonomy work in `src/evaluate.py`?
* **Plain Language:** In real-world enterprise ML, saying *"the model failed 35 questions"* is useless feedback. We need to know *why* it failed. Our diagnostic evaluation script (`src/evaluate.py`) inspects SQLite execution states and assigns every incorrect query to one of 5 deterministic failure buckets:
  1. `TIMEOUT_ABORT`: The query exceeded our 1.0s sandbox limit (triggered by runaway Cartesian cross-joins like `SELECT * FROM A, B, C, D` with missing `ON` conditions).
  2. `SCHEMA_HALLUCINATION`: SQLite raised `no such table` or `no such column`, proving the model invented table or column names not in the prompt DDL.
  3. `SYNTAX_ERROR`: SQLite parser rejection (unclosed quotes, missing keywords, broken SQL, conversational chain-of-thought preambles).
  4. `EMPTY_RESULT`: Query executed without syntax error, but returned 0 rows against a non-empty reference result (indicating over-constrained filters, like `WHERE age > 100 AND age < 20`).
  5. `VALUE_MISMATCH`: Query executed cleanly and returned data, but the multiset of rows differed from gold ground truth (wrong projection, bad join condition, or wrong aggregate function).

---

## Phase 3: GRPO & Consumer GPU Optimization (6GB VRAM)

### 1. How does GRPO work, and how is it different from PPO?
* **Traditional PPO (Proximal Policy Optimization):** Requires training two separate large neural networks: the **Actor** (the LLM writing text) and the **Critic / Value Model** (another LLM trying to predict expected reward at every token). Storing two models in VRAM doubles memory consumption and complicates training.
* **GRPO (Group Relative Policy Optimization):** Eliminates the Critic model entirely!
  1. For each prompt, the model samples a small group of $G$ candidate answers (e.g., $G = 4$).
  2. Each answer is scored by our verifiable SQLite reward function: $\{R_1, R_2, R_3, R_4\}$.
  3. We compute the mean and standard deviation of those scores.
  4. The **advantage** of each answer is simply how much better or worse it is compared to the group average:
     $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
  5. If an answer scored higher than the group average, we increase the probability of its tokens; if lower, we decrease it.

### 2. How did we solve the Consumer GPU Memory Bottleneck (6GB VRAM)?
* **Plain Language:** In standard cloud settings (A100/H100 with 80GB), libraries use huge default buffer sizes. When running on an NVIDIA RTX 4050 Laptop GPU (6.0 GB VRAM), standard GRPO crashed instantly with OOM. We solved this with 4 engineering interventions:
  1. **NF4 4-bit QLoRA Quantization:** Loaded the 1.5B base model in 4-bit NormalFloat with bfloat16 compute, shrinking base model memory from 3.1 GB to **1.53 GB**.
  2. **Chunked Logprob Tile Reduction:** TRL's internal `_CHUNKED_LOGPROB_CHUNK_SIZE` defaults to 32,768 tokens (designed for datacenter GPUs), which caused a 4GB CUDA allocation. We overrode this to **512 tokens**, reducing logprob memory by **64x**.
  3. **Prompt-by-Prompt Rollout Generation:** We created `SafeGRPOTrainer` to override `_generate_single_turn()`, generating 1 prompt ($1 \times 4 = 4$ completions) at a time rather than batching all 4 prompts ($16$ completions) concurrently into the KV cache.
  4. **DDL Schema Sanitization:** Companion `schema.sql` dumps contained thousands of redundant `INSERT INTO` row statements, blowing prompt sequences up to **11,474 tokens** and causing MLP activation recomputation OOMs in backward passes. We sanitized training prompts to pure `CREATE TABLE` DDL, capping all prompts under **987 tokens** (an 11.7x sequence reduction that cut attention memory by **138x**).

---

## Phase 4: Final Results & Analysis

### 1. Verified Benchmark Metrics (Cross-Database Held-Out Test Set)
* **Execution Accuracy (EX):** Improved from **13.89%** (15/108) to **16.67%** (18/108) — a **+2.78% absolute gain** (**+20.01% relative improvement**).
* **Valid SQL Rate (VSR):** Improved from **19.44%** (21/108) to **24.07%** (26/108) — a **+4.63% absolute gain** (**+23.82% relative improvement**).
* **Format Adherence (FAR):** Improved from **19.44%** to **24.07%** (**+4.63% absolute gain**).
* **Maritime Generalization (`ship_1`):** Improved from **22.92%** to **27.08%** correct.
* **Order Tracking Generalization (`tracking_orders`):** Improved from **6.67%** to **8.33%** correct.

### 2. Failure Taxonomy Shift & Qualitative Case Studies
* **Syntax Error Reduction:** `SYNTAX_ERROR` dropped from **87** (80.6%) down to **82** (75.9%).
* **Suppression of Conversational Rambling (Index 40):** The base model consistently failed by generating conversational preambles (`"Let's walk through the question..."`), causing SQLite parser errors. Under GRPO reward shaping, the policy learned to directly generate clean markdown SQL blocks.
* **Equivalence of Correlated Subqueries (Index 64):** In supply chain order queries, the GRPO model formulated correlated subqueries (`WHERE customer_id = (SELECT customer_id FROM Customers WHERE customer_name = 'Jeramie')`) rather than explicit `INNER JOIN`s. Because our reward engine evaluated actual SQLite execution output rather than brittle string exact match, this query received full execution reward 1.0!

---

## Phase 5: Rigorous Re-Evaluation & Artifact Freezing (Harness v2)

### Step 1: Why did we freeze v1 and what did we discover about the 128-token wall?
* **Plain Language:** When we inspected the raw model outputs from the first evaluation (v1), we discovered that the model was trying to think step-by-step before writing code (e.g. *"Let's walk through the question..."*). But because generation was cut off at 128 tokens, it stopped in the middle of a sentence before ever writing the SQL code. The evaluation script grabbed that incomplete English sentence, tried to run it in SQLite, and flagged it as a `SYNTAX_ERROR`.
* **Why freezing matter:** Instead of deleting or overwriting the original files, we renamed them to `eval_v1_grpo.json` and `eval_v1_5step.json`. In real-world data science and research, you never erase historical experiment records; you preserve them transparently so that any auditor or interviewer can see the progression and understand why the methodology was refined.
* **Interview Talking Point:**
  *"When auditing our initial benchmark results, we noticed an evaluation artifact: a 128-token cap truncated chain-of-thought completions before fenced SQL blocks could be emitted. Rather than papering over this, we froze the initial results as v1, documented the missing baseline artifact honestly, and designed an authorized v2 evaluation protocol with extended token limits, truncation detection, and explicit precision tracking."*

### Step 2: What changed in Evaluation Harness v2 and why does it matter?
* **Plain Language:**
  1. **512 Tokens:** We increased the generation ceiling from 128 to 512 tokens so the model has enough room to reason and finish its SQL block.
  2. **Truncation Flag (`hit_token_limit`):** We now record whether a query was cut off at the limit or completed naturally, allowing us to separate real model mistakes from cut-off answers.
  3. **6th Category (`NO_SQL_EXTRACTED`):** If the model outputs plain English conversational text without a fenced SQL block and without starting with `SELECT` or `WITH`, it is categorized as `NO_SQL_EXTRACTED` rather than pretending it was broken SQL syntax.
  4. **Strict Precision Matching:** We added an explicit `--precision` flag (`float16` vs. `4bit`) and logged it in the JSON artifact so both base and GRPO models are evaluated under identical mathematical precision on the same GPU.
* **Interview Talking Point:**
  *"In our v2 evaluation harness, we disentangled format/truncation failures from SQL semantic errors. By expanding the token budget to 512, tracking token truncation flags, and introducing `NO_SQL_EXTRACTED` for non-SQL conversational completions, we ensure our benchmark measures true relational reasoning rather than prompt formatting artifacts."*

### Step 3: What did the fair v2 re-evaluation reveal?
* **Plain Language:**
  When we gave both models 512 tokens and evaluated them on the exact same GPU in 16-bit precision:
  1. **The base model is already quite strong at SQL:** Instead of 13.89% (15/108), the base model actually scored **58.33% (63/108)** execution accuracy! It emitted markdown fenced SQL blocks 100% of the time (108/108) once it had room to finish its thoughts.
  2. **GRPO post-training improved accuracy from 63 to 65 (+1.85% absolute):** The GRPO post-trained model achieved **60.19% (65/108)** execution accuracy.
  3. **Syntax errors were never the real problem:** At 512 tokens, syntax errors dropped to 1 out of 108 queries for both models! The real errors are semantic: `VALUE_MISMATCH` (23 queries where the query ran but selected the wrong rows/columns) and `SCHEMA_HALLUCINATION` (13 queries referencing non-existent columns).
* **Interview Talking Point:**
  *"Conducting a fair re-evaluation under identical hardware and a 512-token budget revealed the true baseline capability of Qwen2.5-Coder-1.5B (58.3% EX, 100% format adherence). The initial 13.9% baseline was an evaluation artifact caused by mid-thought truncation at 128 tokens. Under identical 16-bit evaluation, GRPO achieved 60.19% EX (+1.85% absolute, 63 -> 65 of 108), proving that honest benchmarking is essential to avoid claiming credit for format adherence that the base model already possessed."*

### Step 4: What is a Paired McNemar Test and Wilson Confidence Interval?
* **Plain Language:**
  - **Wilson 95% Confidence Interval:** Instead of giving a single percentage (e.g. 58.3%), a confidence interval shows the range where the true accuracy lies given our sample size of 108 questions. For Base, it is [48.9%, 67.2%]; for GRPO, it is [50.8%, 68.9%]. Because these ranges overlap heavily, we cannot say one model is definitively better from raw percentages alone.
  - **Paired McNemar's Test:** When you test two models on the exact same questions, questions where *both* succeed (63) or *both* fail (43) tell you nothing about the difference between them. You only look at the **discordant pairs**: questions where one model succeeded and the other failed. Here, Base won 0 questions ($b=0$) and GRPO won 2 questions ($c=2$). McNemar's test asks: *"If the two models were identical in ability, what is the chance of flipping a coin twice and getting 2 heads?"* The answer is $p = 0.5000$ (50%). Since $p > 0.05$, the difference is not statistically significant.
  - **Zero Regressions ($b=0$):** Although the lift is small, the GRPO model introduced **zero regressions**—it never broke a question that the base model answered correctly.
* **Interview Talking Point:**
  *"In rigorous LLM evaluation, claiming victory over small percentage shifts without hypothesis testing is an amateur mistake. We implemented a paired McNemar test on the 108 test questions ($b=0, c=2, p=0.5000$) and computed Wilson 95% confidence intervals. This demonstrated that while GRPO fixed two complex relational logic bugs with zero regressions, the net +1.85% gain is within expected statistical sampling variance."*

### Step 5 & 6: Why is Documentation Honesty your Biggest Interview Advantage?
* **Plain Language:** Many candidates copy-paste inflated metrics like *"+20% accuracy boost"* or *"thousands of rollouts"*. When an experienced interviewer asks to see the code or drill down into the numbers, these claims collapse. By voluntarily diagnosing the 128-token truncation artifact, reporting exact counts (63 -> 65), showing zero regressions, and noting that public Spider schemas may exist in pre-training data, you demonstrate the scientific maturity of a senior evaluation engineer.
* **Interview Talking Point:**
  *"What sets this project apart is rigorous evaluation hygiene: when my initial audit revealed that a 128-token cap artificially depressed the baseline to 13.9% by cutting off chain-of-thought outputs mid-sentence, I didn't hide it. I upgraded the harness, established a 58.3% baseline, and proved that GRPO repaired relational filter bugs without regressions, all backed by committed JSON artifacts and paired statistical tests."*

---

## 30-Second Interview Elevator Pitch (DP World AI/ML Track)
> *"For this project, I post-trained `Qwen2.5-Coder-1.5B` using Group Relative Policy Optimization (GRPO) with a verifiable in-memory SQLite execution reward on multi-table shipping and order fulfillment schemas. I built a defense-in-depth sandbox with dual opcode and thread timeouts, optimized rollout memory to run locally in 6GB VRAM, and authored an automated 6-class failure taxonomy. When auditing initial evaluations, I diagnosed a 128-token truncation artifact that artificially depressed the baseline; under an authorized 512-token re-evaluation, the base model achieved 58.33% execution accuracy, while GRPO reached 60.19% (63 to 65 of 108 correct) with zero regressions ($b=0, p=0.5000$). This experience demonstrates not just post-training mechanics, but the rigorous statistical integrity essential for enterprise evaluation."*
