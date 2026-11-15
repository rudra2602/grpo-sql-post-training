"""
GRPO (Group Relative Policy Optimization) post-training script for text-to-SQL.

Features:
  - Hugging Face TRL GRPOTrainer with 16-bit LoRA (PEFT)
  - Group size G=4, effective batch size 8 prompts (32 completions per step)
  - Custom verifiable SQLite execution reward callable (grpo_sql_reward_func)
  - Rollout bound max_completion_length=96 tokens, KL beta=0.005
  - Logs zero-variance reward group fraction to WandB
  - Saves LoRA checkpoints every 25-50 steps
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List
import numpy as np

# Ensure project root is in sys.path when invoked directly as python src/train_grpo.py
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import prepare_grpo_dataset
from src.reward import grpo_sql_reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO Post-Training with SQLite Execution Rewards.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Base model ID or path.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/grpo_final", help="Directory to save LoRA checkpoints.")
    parser.add_argument("--max_steps", type=int, default=100, help="Total training steps.")
    parser.add_argument("--group_size", type=int, default=4, help="Number of completions per prompt group (G).")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device prompt batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--kl_beta", type=float, default=0.005, help="KL penalty coefficient (beta).")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--max_completion_length", type=int, default=96, help="Maximum completion generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Rollout sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Rollout top-p sampling.")
    parser.add_argument("--save_steps", type=int, default=25, help="Steps between checkpoint saves.")
    parser.add_argument("--reward_mode", type=str, default="tiered", choices=["tiered", "binary"], help="Reward shaping mode.")
    parser.add_argument("--wandb_project", type=str, default="grpo-sql-post-training", help="WandB project name.")
    return parser.parse_args()


def main() -> None:
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = parse_args()

    # Workaround for Colab pre-installed torchao (<0.16.0) triggering PEFT version mismatch error
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao
        peft.tuners.lora.torchao.is_torchao_available = lambda: False
        peft.tuners.lora.torchao.dispatch_torchao = lambda *args, **kwargs: None
    except Exception:
        pass

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    try:
        import wandb
    except ImportError:
        wandb = None

    class EmptyCacheCallback(TrainerCallback):
        """Clears CUDA memory cache after each optimization step to eliminate fragmentation."""
        def on_step_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 60)
    print("STARTING GRPO POST-TRAINING RUN")
    print(f"Model:                {args.model_name}")
    print(f"Max Steps:            {args.max_steps}")
    print(f"Group Size (G):       {args.group_size}")
    print(f"Effective Batch:      {args.batch_size * args.gradient_accumulation_steps} prompts ({args.batch_size * args.gradient_accumulation_steps * args.group_size} completions/step)")
    print(f"Learning Rate:        {args.lr}")
    print(f"KL Beta:              {args.kl_beta}")
    print(f"Max Completion Len:   {args.max_completion_length}")
    print(f"Reward Mode:          {args.reward_mode}")
    print(f"Output Directory:     {args.output_dir}")
    print("=" * 60)

    # 1. Prepare Training Dataset
    print("\nLoading curated logistics training dataset...")
    data_dict = prepare_grpo_dataset(split="train")
    train_dataset = Dataset.from_dict(data_dict)
    print(f"Loaded {len(train_dataset)} training prompt-SQL pairs.")

    # 2. Configure 16-bit LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. Define Reward Callable with Zero-Variance Tracking
    def grpo_reward_wrapper(completions: List[str], gold_sql: List[str], db_path: List[str], **kwargs: Any) -> List[float]:
        rewards = grpo_sql_reward_func(
            completions=completions,
            gold_sql=gold_sql,
            db_path=db_path,
            reward_mode=args.reward_mode,
            **kwargs,
        )

        # Track and log proportion of rollout groups with zero reward variance
        if len(rewards) >= args.group_size:
            groups = [rewards[i:i + args.group_size] for i in range(0, len(rewards), args.group_size)]
            zero_var_flags = [bool(np.std(g) == 0.0) for g in groups if len(g) == args.group_size]
            if zero_var_flags and wandb is not None and wandb.run is not None:
                frac = float(np.mean(zero_var_flags))
                wandb.log({"train/zero_reward_variance_fraction": frac})

        return rewards

    # 4. Configure GRPO Hyperparameters
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.kl_beta,
        save_steps=args.save_steps,
        logging_steps=1,
        save_total_limit=3,
        report_to="wandb" if wandb is not None and wandb.run is not None else "none",
        use_vllm=False,
        fp16=torch.cuda.is_available(),
    )

    # 5. Initialize GRPOTrainer
    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[grpo_reward_wrapper],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        callbacks=[EmptyCacheCallback()],
    )

    # 6. Execute Training Loop
    print("\nStarting GRPO Rollouts and Optimization...")
    trainer.train()

    # 7. Save Final Adapter Weights
    print(f"\nSaving final LoRA adapter to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    print("GRPO Post-Training Run Completed Successfully!")


if __name__ == "__main__":
    main()
