import json

with open("docs/eval_v2_base.json", "r", encoding="utf-8") as f:
    b = json.load(f)["detailed_results"]
with open("docs/eval_v2_grpo.json", "r", encoding="utf-8") as f:
    g = json.load(f)["detailed_results"]

print("=" * 60)
print("INSPECTING THE 2 FLIPPED QUERIES (Base Fail -> GRPO Correct)")
print("=" * 60)
for i in range(len(b)):
    if not b[i]["is_match"] and g[i]["is_match"]:
        print(f"\nIndex {i} (Database: {b[i]['db_id']}):")
        print(f"Question:    {b[i]['question']}")
        print(f"Base Failure:{b[i]['error_category']}")
        print(f"Gold SQL:    {b[i]['gold_sql']}")
        print(f"Base SQL:    {b[i]['extracted_sql']}")
        print(f"GRPO SQL:    {g[i]['extracted_sql']}")
