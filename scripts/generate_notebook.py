"""
Generates notebooks/colab_train.ipynb notebook artifact.
"""
import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GRPO Post-Training with Verifiable SQLite Execution Rewards\n",
                "### Portfolio & Capstone: DP World (LLM Post-Training and Evaluation Track)\n",
                "\n",
                "This self-contained Google Colab notebook implements the full training and evaluation journey:\n",
                "1. **Environment Verification:** Runs `pytest tests/ -v` to ensure 100% pass rate on Linux.\n",
                "2. **Phase 2 (Baseline Evaluation):** Deterministic greedy evaluation ($T=0.0$) of `Qwen/Qwen2.5-Coder-1.5B-Instruct` across 108 held-out test questions.\n",
                "3. **Phase 3 (GRPO Training):** 10-minute smoke test (0.5B model) followed by primary 1.5B GRPO training (100 steps, $G=4$, 16-bit LoRA, KL penalty $\\beta=0.005$).\n",
                "4. **Phase 4 (Post-Training Evaluation):** Re-evaluation of the trained model and automated 5-class failure taxonomy analysis.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Verification & GPU Check\n",
                "Select a **T4 GPU** runtime (`Runtime -> Change runtime type -> T4 GPU`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Clone Repository & Install Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone the repository if not already in directory:\n",
                "# !git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git\n",
                "# %cd <YOUR_REPO_NAME>\n",
                "\n",
                "!pip install --quiet \\\n",
                "    \"transformers>=4.40.0\" \\\n",
                "    \"trl>=0.14.0\" \\\n",
                "    \"peft>=0.10.0\" \\\n",
                "    \"accelerate>=0.28.0\" \\\n",
                "    \"datasets>=2.18.0\" \\\n",
                "    \"wandb>=0.16.0\" \\\n",
                "    \"pytest>=8.0.0\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Mount Google Drive for Checkpoint Persistence\n",
                "Saves trained LoRA checkpoints to Google Drive so no weights are lost if Colab preempts."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "!mkdir -p /content/drive/MyDrive/grpo_sql_checkpoints\n",
                "!mkdir -p checkpoints"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Run Complete Verification Test Suite\n",
                "Verify that all sandbox, reward, and zero-leakage unit tests pass 100% on Colab Linux."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pytest tests/ -v"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Phase 2: Baseline Zero-Shot Evaluation\n",
                "Runs greedy decoding ($T=0.0$) with `Qwen/Qwen2.5-Coder-1.5B-Instruct` on the 108 held-out test questions.\n",
                "Generates `docs/baseline_eval.json` with execution metrics and 5-class failure breakdown."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python src/evaluate.py \\\n",
                "    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \\\n",
                "    --split test \\\n",
                "    --output_file docs/baseline_eval.json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Inspect Baseline Metrics & Failure Taxonomy"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "with open('docs/baseline_eval.json', 'r') as f:\n",
                "    base_data = json.load(f)\n",
                "\n",
                "m = base_data['metrics']\n",
                "print(f\"Baseline Execution Accuracy (EX):    {m['execution_accuracy_pct']}%\")\n",
                "print(f\"Baseline Valid SQL Rate (VSR):        {m['valid_sql_rate_pct']}%\")\n",
                "print(f\"Baseline Format Adherence Rate (FAR): {m['format_adherence_rate_pct']}%\")\n",
                "print('\\n5-Class Failure Taxonomy Breakdown:')\n",
                "for cat, count in base_data['failure_taxonomy'].items():\n",
                "    print(f\"  {cat:<25}: {count}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Phase 3: Smoke Test & Primary GRPO Training\n",
                "Run a 10-minute dry-run / smoke test on `Qwen2.5-Coder-0.5B-Instruct` first to calibrate runtime and verify VRAM stability."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 10-Minute Smoke Test on 0.5B model\n",
                "!python src/train_grpo.py \\\n",
                "    --model_name Qwen/Qwen2.5-Coder-0.5B-Instruct \\\n",
                "    --max_steps 15 \\\n",
                "    --group_size 4 \\\n",
                "    --output_dir checkpoints/grpo_smoke_test"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Launch Primary GRPO Training Run (1.5B Model)\n",
                "Trains `Qwen/Qwen2.5-Coder-1.5B-Instruct` with 16-bit LoRA for 100 steps on 406 logistics query pairs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python src/train_grpo.py \\\n",
                "    --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \\\n",
                "    --max_steps 100 \\\n",
                "    --group_size 4 \\\n",
                "    --lr 1e-5 \\\n",
                "    --output_dir checkpoints/grpo_final"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Phase 4: Post-Training Evaluation on Held-Out Test Set\n",
                "Evaluate the post-trained GRPO adapter on the exact same 108 test questions ($T=0.0$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python src/evaluate.py \\\n",
                "    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \\\n",
                "    --adapter_path checkpoints/grpo_final \\\n",
                "    --split test \\\n",
                "    --output_file docs/grpo_eval.json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Before-and-After Comparison & Taxonomy Shift Analysis\n",
                "Computes measured deltas between Base Model and Post-Trained GRPO Model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "with open('docs/baseline_eval.json') as f:\n",
                "    b = json.load(f)\n",
                "with open('docs/grpo_eval.json') as f:\n",
                "    g = json.load(f)\n",
                "\n",
                "b_m = b['metrics']\n",
                "g_m = g['metrics']\n",
                "\n",
                "print('=' * 60)\n",
                "print('METRIC COMPARISON (Held-Out Test Set: 108 pairs)')\n",
                "print('=' * 60)\n",
                "print(f\"Execution Accuracy (EX):     Base={b_m['execution_accuracy_pct']}%  ->  GRPO={g_m['execution_accuracy_pct']}%  (Delta: {g_m['execution_accuracy_pct'] - b_m['execution_accuracy_pct']:+.2f}%)\")\n",
                "print(f\"Valid SQL Rate (VSR):         Base={b_m['valid_sql_rate_pct']}%  ->  GRPO={g_m['valid_sql_rate_pct']}%  (Delta: {g_m['valid_sql_rate_pct'] - b_m['valid_sql_rate_pct']:+.2f}%)\")\n",
                "print(f\"Format Adherence Rate (FAR):  Base={b_m['format_adherence_rate_pct']}%  ->  GRPO={g_m['format_adherence_rate_pct']}%  (Delta: {g_m['format_adherence_rate_pct'] - b_m['format_adherence_rate_pct']:+.2f}%)\")\n",
                "\n",
                "print('\\nFAILURE TAXONOMY SHIFT:')\n",
                "print(f\"{'Category':<25} {'Base Count':<12} {'GRPO Count':<12} {'Shift':<10}\")\n",
                "print('-' * 58)\n",
                "for cat in b['failure_taxonomy']:\n",
                "    bc = b['failure_taxonomy'][cat]\n",
                "    gc = g['failure_taxonomy'][cat]\n",
                "    print(f\"{cat:<25} {bc:<12} {gc:<12} {gc - bc:+d}\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

output_path = Path("notebooks/colab_train.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created {output_path} successfully.")
