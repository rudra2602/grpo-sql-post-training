import time
import sys
from pathlib import Path
import torch

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data_loader import load_logistics_dataset
from src.evaluate import run_evaluation

model_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Loading {model_path} in float16...")

t0_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)
model.eval()
print(f"Model loaded in {time.time() - t0_load:.2f}s")

dataset = load_logistics_dataset(split="test")[:10]
print(f"Evaluating first {len(dataset)} questions with max_new_tokens=512...")

t0_eval = time.time()
summary = run_evaluation(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    device=device,
    max_new_tokens=512,
)
eval_time = time.time() - t0_eval
sec_per_query = eval_time / len(dataset)
eta_total_sec = sec_per_query * 108
eta_total_min = eta_total_sec / 60.0

print("\n" + "=" * 50)
print("SPEED CALIBRATION RESULTS (10 queries)")
print("=" * 50)
print(f"Total time for 10 queries: {eval_time:.2f}s")
print(f"Average time per query:    {sec_per_query:.2f}s")
print(f"Estimated ETA for 108:     {eta_total_sec:.1f}s ({eta_total_min:.2f} minutes)")
print(f"Sample execution matches:  {summary['counts']['correct']}/{len(dataset)}")
print("=" * 50)
