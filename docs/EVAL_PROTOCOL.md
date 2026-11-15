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
   Measures the percentage of test questions where the candidate query executed successfully and returned the exact same table of rows and values as the gold reference query.

2. **Valid SQL Rate (VSR %)**
   $$\text{VSR} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Candidate query executes without SQLite runtime error})}{N} \times 100$$
   Measures syntactic competence and basic schema awareness.

3. **Format Adherence Rate (FAR %)**
   $$\text{FAR} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Output contains valid closed markdown sql block})}{N} \times 100$$
   Measures whether the model follows system formatting instructions without conversational drift.

---

## 4. Fair Comparison Protocol

To ensure absolute scientific fairness, all models must be evaluated under identical environmental constraints:
* **Decoding Parameters:** Greedy decoding strictly enforced (`temperature = 0.0`, `do_sample = False`, `num_beams = 1`).
* **Prompt Template:** Exact same system prompt and DDL representation via `src/data_loader.py`:
  ```text
  You are an expert SQL assistant. Given the following SQLite database schema, write a valid SQL query that answers the user question.
  Return ONLY the SQL query enclosed in ```sql ... ``` markdown tags.
  
  ### Schema:
  {schema_ddl}
  
  ### Question:
  {question}
  ```
* **Execution Environment:** Same timeout limit (1.0 second), same memory constraints, and same SQLite binaries.
* **Immutability Rule:** No changes to `data/test_logistics.json` or `src/evaluate.py` are permitted once the baseline numbers are recorded.

---

## 5. Automated 5-Class Failure Taxonomy & Qualitative Case Studies

For every test question where $V_{\text{pred}} \not\equiv V_{\text{gold}}$, our diagnostic evaluation harness (`src/evaluate.py`) automatically inspects the SQLite execution result to assign one of 5 deterministic execution failure categories:

```
                          Query Evaluation
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          Execution Error?               Executed Cleanly?
                 │                               │
        ┌────────┼────────┐              ┌───────┴───────┐
        ▼        ▼        ▼              ▼               ▼
     Syntax    Schema    Timeout      Empty Set?       Value
     Error  Hallucination Abort       (0 rows vs gold) Mismatch
```

### Taxonomy Category Definitions

| # | Error Category Code | Detection Logic / Trigger |
|---|---|---|
| 1 | `SYNTAX_ERROR` | SQLite throws a syntax error (e.g., `sqlite3.OperationalError: near "...": syntax error`, unclosed quotes, mismatched parentheses). |
| 2 | `SCHEMA_HALLUCINATION` | SQLite throws `no such table: ...` or `no such column: ...`, indicating the model hallucinated schema entities not in the prompt DDL. |
| 3 | `TIMEOUT_ABORT` | Query execution exceeds 1.0 second and is aborted by the sandbox (indicates runaway cartesian joins from missing join conditions). |
| 4 | `EMPTY_RESULT` | Query executes cleanly without runtime error, but returns 0 rows while the gold query returns data (indicates over-constrained or conflicting filter conditions). |
| 5 | `VALUE_MISMATCH` | Query executes cleanly and returns non-empty rows, but multiset row contents or column projections differ from the gold reference output. |

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
