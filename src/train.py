"""
Training entrypoint.

Usage:
    python -m src.train --config config/config.yaml

Loads the processed dataset (run data_preparation.py first), attaches LoRA
adapters to the base model, trains with TRL's SFTTrainer, and saves the
final adapter to disk (and optionally pushes to the Hugging Face Hub).
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

from src.config_utils import load_config
from src.model_utils import load_model_for_training


def build_trainer(model, tokenizer, train_dataset, val_dataset, cfg: dict) -> SFTTrainer:
    train_cfg = cfg["training"]

    args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=cfg["data"]["seed"],
        report_to="none",
        dataset_text_field="text",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
    )

    # Only compute loss on the response tokens, not the instruction tokens —
    # documented in the QLoRA paper as a meaningful accuracy improvement,
    # especially for multi-turn data.
    if train_cfg["train_on_responses_only"]:
        # ChatML markers — must match the template applied in model_utils.py /
        # data_preparation.py. Only the assistant's SQL output contributes to
        # the loss, which is what the QLoRA paper recommends for instruction data.
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    return trainer


def run_training(cfg: dict, resume_from_checkpoint: str = None) -> None:
    model, tokenizer = load_model_for_training(cfg)

    train_dataset = load_dataset(
        "json", data_files=cfg["data"]["processed_train_path"], split="train"
    )
    val_dataset = load_dataset(
        "json", data_files=cfg["data"]["processed_val_path"], split="train"
    )

    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, cfg)

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = Path(cfg["training"]["final_adapter_dir"])
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final LoRA adapter to {final_dir}")

    if cfg["huggingface"]["push_to_hub"]:
        repo_id = cfg["huggingface"]["repo_id"]
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"Pushed adapter to Hugging Face Hub: {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None,
                         help="Path to a checkpoint dir to resume training from (survives Colab disconnects)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
