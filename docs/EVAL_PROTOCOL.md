# Evaluation Protocol & Diagnostic Error Taxonomy

## 1. Overview
In evaluating post-trained models for the DP World "LLM Post-Training and Evaluation" track, evaluation must be:
1. **Deterministic and Reproducible:** Eliminating random sampling variance.
2. **Strictly Isolated (Zero-Leakage):** Evaluating cross-database schema generalization.
3. **Diagnostically Granular:** Distinguishing between syntax errors, schema hallucinations, timeouts, and result set mismatches.

> [!NOTE]
> **Data Contamination & Baseline Note:** Because Spider is a public academic benchmark, its schemas and query patterns may exist within the pre-training or instruction-tuning corpora of `Qwen2.5-Coder-1.5B-Instruct`. Baseline zero-shot performance may consequently be elevated, and post-training GRPO improvements are targeted at domain-grounded execution consistency and edge-case syntax/timeout elimination rather than de novo syntax acquisition.

---

## 2. Dataset Splits & Leakage Prevention

The benchmark suite is curated from relational logistics, shipping, warehouse storage, order tracking, and product fulfillment schemas from the Spider corpus.

### Dataset Distribution

| Split | Database Identifier (`db_id`) | Description | Verified Valid Pairs ($\ge 1$ Row) |
|---|---|---|:---:|
| **Train** | `customers_and_invoices` | Enterprise billing, invoices, payment accounts | 82 |
| **Train** | `department_store` | Large retail logistics, inventory, supplier deliveries | 88 |
| **Train** | `customers_and_addresses` | Customer shipping locations, delivery contact channels | 88 |
| **Train** | `product_catalog` | Product structures, attribute catalogs | 42 |
| **Train** | `store_product` | Store locations, district inventories | 42 |
| **Train** | `ship_mission` | Vessel operations and mission assignments | 30 |
| **Train** | `products_for_hire` | Equipment rental inventory, payment records | 18 |
| **Train** | `customer_deliveries` | Truck delivery routes, driver assignments, delivery logs | 16 |
| **Total Train** | *8 Distinct Schemas* | *Cross-Domain Logistics & Supply Chain* | **406** |
| **Test (Held-Out)** | `tracking_orders` | Shipment fulfillment, carrier tracking, order line items | 60 |
| **Test (Held-Out)** | `ship_1` | Maritime vessels, ship dimensions, captain rosters | 48 |
| **Total Test** | *2 Held-Out Schemas* | *Unseen Logistics & Marine Operations* | **108** |

### Leakage Prevention Guarantees
* **Cross-Database Split:** The databases in `test_logistics.json` **never appear** in `train_logistics.json`.
* **Zero Schema Overlap:** $\text{Schemas}_{\text{train}} \cap \text{Schemas}_{\text{test}} = \emptyset$.
* **Automated Unit Test Verification:** `tests/test_leakage.py` loads both datasets and asserts that the set intersection of database IDs and table names is completely empty.

---

## 3. Evaluation Metrics

Both the un-tuned Base Model (`Qwen2.5-Coder-1.5B-Instruct`) and the GRPO post-trained model are evaluated across the exact same 108 held-out test questions.

### Primary Metrics

1. **Execution Accuracy (EX %)**
   $$\text{EX} = \frac{\sum_{i=1}^{N} \mathbb{I}(V_{\text{pred}}^{(i)} \equiv_{\text{multiset}} V_{\text{gold}}^{(i)})}{N} \times 100$$
   Measures the percentage of test questions where the candidate query executed successfully and returned the exact same table of rows and values as the gold reference query. Reported with **Wilson score 95% confidence intervals**.

2. **Valid SQL Rate (VSR %)**
   $$\text{VSR} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Candidate query executes without SQLite runtime error})}{N} \times 100$$
   Measures syntactic competence and basic schema awareness.

3. **Format Adherence Rate (FAR %)**
   $$\text{FAR} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Output contains valid closed markdown sql block})}{N} \times 100$$
   Measures whether the model follows system formatting instructions without conversational drift.

4. **Paired Exact McNemar's Test**
   Because both models are evaluated on the exact same 108 test queries, we compute exact binomial two-sided $p$-values on discordant pairs ($b$ = only base correct, $c$ = only GRPO correct) to determine whether accuracy differences are statistically significant ($p < 0.05$) or within sampling noise.

---

## 4. Fair Comparison Protocol & Evolution to Harness v2

To ensure absolute scientific fairness, all models must be evaluated under identical environmental constraints:
* **Decoding Parameters:** Greedy decoding strictly enforced (`temperature = 0.0`, `do_sample = False`, `num_beams = 1`).
* **Prompt Template:** Exact same system prompt and DDL representation via `src/data_loader.py`.
* **Execution Environment:** Same timeout limit (1.0 second), same memory constraints, and same SQLite binaries.
* **Hardware & Precision Consistency:** Both models are evaluated on the same local GPU (NVIDIA GeForce RTX 4050 6GB) under the exact same precision (`float16`).

### Harness v2 Evolution (Addressing the 128-Token Truncation Artifact)
In initial v1 testing, generation was constrained to `max_new_tokens = 128`. Detailed audit revealed that `Qwen2.5-Coder-1.5B-Instruct` frequently begins completions with step-by-step reasoning (*"Let's walk through the question..."*), causing outputs to be cut off mid-thought before emitting fenced SQL blocks. The evaluation script previously executed this truncated English prose, resulting in 82/108 queries failing as false `SYNTAX_ERROR`s.

Under authorized **Harness v2**:
1. **`max_new_tokens = 512`:** Allocates sufficient token budget so reasoning and SQL blocks complete naturally. (Average completion length is ~240 tokens).
2. **Token Limit Diagnostics:** Each question records exact generated tokens and `hit_token_limit` boolean ($\ge 512$).
3. **6-Class Failure Taxonomy:** Adds `NO_SQL_EXTRACTED` for outputs lacking fenced blocks and not starting with SQL keywords (`SELECT`/`WITH`), clearly distinguishing formatting/prose from broken SQLite syntax.
4. **Explicit Precision Tracking:** `--precision` CLI flag (`float16`) logged in JSON artifacts for auditability.

---

## 5. Automated 6-Class Failure Taxonomy

For every test question where $V_{\text{pred}} \not\equiv V_{\text{gold}}$, our diagnostic evaluation harness (`src/evaluate.py`) automatically inspects the execution result to assign one of 6 deterministic categories:

| # | Error Category Code | Detection Logic / Trigger |
|---|---|---|
| 1 | `NO_SQL_EXTRACTED` | Output contains no fenced ` ```sql ` block and the text does not begin with `SELECT` or `WITH`. |
| 2 | `SYNTAX_ERROR` | SQLite parser failure on extracted SQL (`sqlite3.OperationalError: near "...": syntax error`, unclosed quotes, mismatched parentheses). |
| 3 | `SCHEMA_HALLUCINATION` | SQLite throws `no such table: ...` or `no such column: ...`, indicating references to non-existent schema entities. |
| 4 | `TIMEOUT_ABORT` | Query execution exceeds the 1.0s sandbox limit (indicating runaway cartesian joins from missing join conditions). |
| 5 | `EMPTY_RESULT` | Query executes cleanly without error, but returns 0 rows while the gold query returns data (over-constrained filters or impossible conditions). |
| 6 | `VALUE_MISMATCH` | Query executes cleanly and returns non-empty rows, but multiset row contents or column projections differ from gold reference output. |

### Qualitative Relational Error Analysis (Case Studies)
Fine-grained relational root causes—distinguishing whether a `VALUE_MISMATCH` was caused by:
* `JOIN_ERROR` (incorrect `ON` keys or missing intermediate junction tables),
* `CONDITION_ERROR` (inverted boolean logic, `>=` vs `>`, or wrong literal constant), or
* `AGGREGATION_ERROR` (missing `GROUP BY`, wrong aggregate function, or omitted `DISTINCT`),

are evaluated through **3 manual qualitative case studies** in Phase 4 comparing base model vs. GRPO predictions.

### Comparative Error Reporting
The evaluation script generates a comparative distribution table:

$$\Delta \text{Error}_k = \text{Count}_{\text{Base}}(k) - \text{Count}_{\text{GRPO}}(k)$$

This demonstrates precisely which failure modes reinforcement learning eliminated (e.g. "GRPO eliminated 100% of Timeout Aborts and reduced Schema Hallucinations by [X]%").
