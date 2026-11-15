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

## 30-Second Interview Elevator Pitch (DP World AI/ML Track)
> *"For this project, I post-trained `Qwen2.5-Coder-1.5B` using Group Relative Policy Optimization (GRPO) with a verifiable in-memory SQLite execution reward on multi-table logistics and container tracking schemas. I engineered a defense-in-depth sandbox with dual opcode and thread timeouts, designed a tiered composite reward function (+0.1 format, +0.2 syntax, +0.7 execution match) with anti-hacking guards, and optimized the rollout engine to train locally within 6GB VRAM. Under deterministic zero-leakage greedy decoding on 108 unseen supply-chain test questions, the GRPO post-trained model achieved a **+20.01% relative improvement in execution accuracy** (13.89% to 16.67%) and boosted valid SQL generation by **+23.82%** (19.44% to 24.07%), while our 5-class failure taxonomy demonstrated measurable reduction in conversational syntax errors."*
