# Reward Function Design & Specification

## 1. Overview & Objective
In Group Relative Policy Optimization (GRPO), the reward function serves as the sole loss feedback signal. The model samples a group of candidate SQL completions $\{y_1, y_2, \dots, y_G\}$ for a given schema and question prompt $x$. The completions are evaluated, producing scalar rewards $\{R_1, R_2, \dots, R_G\}$.

The advantage for each completion $y_i$ is computed by standardizing rewards relative to the group:
$$A_i = \frac{R_i - \text{mean}(\{R_1, \dots, R_G\})}{\text{std}(\{R_1, \dots, R_G\}) + \epsilon}$$

If the reward function is a binary all-or-nothing metric ($0.0$ or $1.0$), early training batches suffer from zero-variance groups ($\text{std} = 0$), providing zero gradient update. To overcome this **sparse reward exploration failure**, we employ a **Tiered Composite Reward** that provides dense learning milestones while remaining mathematically robust against exploitation.

---

## 2. Mathematical Reward Specification

Each candidate completion $y$ is assigned a score $R(y) \in [0.0, 1.0]$ defined as:

$$R(y) = R_{\text{format}}(y) + R_{\text{syntax}}(y) + R_{\text{exec\_match}}(y)$$

### Component Breakdown

| Tier | Component | Weight | Condition / Trigger |
|---|---|---|---|
| **Tier 1** | **Format Adherence ($R_{\text{format}}$)** | **+0.1** | The completion contains a cleanly closed ````sql ... ```` markdown code block with extractable, non-empty SQL content. |
| **Tier 2** | **Syntax & Executability ($R_{\text{syntax}}$)** | **+0.2** | The extracted SQL query passes static safety checks and executes against the in-memory SQLite database without runtime errors. |
| **Tier 3** | **Execution Match ($R_{\text{exec\_match}}$)** | **+0.7** | The executed query results match the gold-standard reference query results under **multiset (bag) equivalence**. |

$$\text{Maximum Reward} = 0.1 + 0.2 + 0.7 = \mathbf{1.0}$$
$$\text{Syntax-Only Reward} = 0.1 + 0.2 + 0.0 = \mathbf{0.3}$$
$$\text{Format-Only Reward} = 0.1 + 0.0 + 0.0 = \mathbf{0.1}$$
$$\text{Complete Failure Reward} = \mathbf{0.0}$$

---

## 3. Multiset (Bag) Equivalence
In SQL relational databases, unless an explicit `ORDER BY` clause is requested in the natural language question, the order of returned rows is non-deterministic.

* **Definition:** Let $V_{\text{pred}}$ be the list of result rows returned by the candidate query, and $V_{\text{gold}}$ be the list of result rows returned by the reference query.
* **Tuple Matching:** Each row is converted to a tuple of stringified, lowercased, whitespace-trimmed values.
* **Multiset Equality:** 
  $$\text{Counter}(V_{\text{pred}}) == \text{Counter}(V_{\text{gold}})$$
* If `ORDER BY` is present in the gold query, strict ordered list equivalence ($V_{\text{pred}} == V_{\text{gold}}$) is enforced.

---

## 4. Reward Hacking Risks & Defensive Guards

RL agents are notorious for discovering mathematical shortcuts that harvest reward without solving the task. We institute three defensive checks:

### Guard 1: Empty Result Hacking
* **Vulnerability:** If a question yields zero matching records (empty table `[]`), a model generating `SELECT * FROM table WHERE 1=0;` or an over-filtered query would achieve an accidental match. If exploited, the policy could collapse into always returning empty sets.
* **Defense:**
  1. If $V_{\text{pred}} == []$ and $V_{\text{gold}} == []$, full execution credit ($+0.7$) is **NOT** awarded automatically.
  2. Full credit is granted only if the candidate query's referenced tables and columns intersect with the gold query's schema elements.
  3. Furthermore, our dataset curation intentionally filters out empty-return queries from both train and test splits so that legitimate result sets always contain $\ge 1$ row.

### Guard 2: Trivial Query Rejection
* **Vulnerability:** If the reference query returns a single scalar integer (e.g. `COUNT(*)` returning `5`), a query like `SELECT 5;` returns the identical row without interacting with the database tables.
* **Defense:**
  * Any query that does not reference at least one valid table present in the active database schema (`FROM <table>`) receives $R_{\text{syntax}} = 0.0$ and $R_{\text{exec\_match}} = 0.0$.

### Guard 3: Syntax Gaming
* **Vulnerability:** Generating `SELECT * FROM containers;` for every question to easily collect the $+0.3$ format and syntax points.
* **Defense:**
  * In GRPO, advantages are relative within the group ($A_i = \frac{R_i - \bar{R}}{\sigma}$). If all 4 completions collect the trivial $+0.3$ syntax reward, their advantage is $0.0$, yielding no gradient benefit! 
  * Only completions that achieve the execution match ($1.0$) obtain positive relative advantage, driving the policy toward genuine correctness.

---

## 5. Required Unit Test Suite (`tests/test_reward.py`)

The reward implementation must pass the following synthetic unit test cases:

1. **`test_perfect_query()`**: Perfectly formatted SQL matching ground truth $\to$ Returns exactly `1.0`.
2. **`test_syntax_only()`**: Valid SQL executing on the table, but returning wrong rows $\to$ Returns exactly `0.3`.
3. **`test_syntax_error()`**: Malformed SQL (`SELECT FROM WHERE`) inside markdown tags $\to$ Returns exactly `0.1`.
4. **`test_no_markdown_tags()`**: Raw SQL text without ````sql ... ```` tags:
   - If raw SQL runs and matches: loses format point $\to$ Returns `0.9`.
5. **`test_trivial_scalar_hacking()`**: Query is `SELECT 1;` when gold output is `1` $\to$ Returns `0.0` (caught by table reference guard).
6. **`test_multiset_ordering()`**: Candidate returns `[('A', 1), ('B', 2)]` while gold returns `[('B', 2), ('A', 1)]` $\to$ Returns `1.0` (order-invariant).
7. **`test_dangerous_mutation_attempt()`**: Query contains `DROP TABLE shipments;` $\to$ Execution aborted by safety sandbox; returns `0.1` format score, `0.0` for syntax and match.
8. **`test_empty_result_prevention()`**: Both candidate and gold return `[]` on mismatched schema $\to$ Zero execution match points awarded.
