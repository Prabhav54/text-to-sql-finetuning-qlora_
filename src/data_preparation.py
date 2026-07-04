"""
Data preparation module.

Responsible for:
  1. Loading the raw text-to-SQL dataset (b-mc2/sql-create-context) from
     Hugging Face Hub, or a local file if one has been dropped into data/raw/
  2. Slicing to 10,000+ records (matches the resume's dataset size)
  3. Formatting each example as a ChatML conversation:
       system: instructions + schema context
       user:   natural language question
       assistant: target SQL query
  4. Splitting into train/val and saving processed splits to disk

Swap the dataset in config/config.yaml and adjust `format_example()` below
if you move to a different domain/task — everything downstream (train.py,
inference.py, evaluate.py) just consumes the resulting "text" field.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset, Dataset
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer

from src.config_utils import load_config


SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a "
    "natural language question, generate the correct SQL query."
)


def load_raw_dataset(cfg: dict):
    """Load dataset from Hugging Face Hub (falls back to local jsonl if configured)."""
    dataset_name = cfg["data"]["dataset_name"]
    raw_path = Path(cfg["data"]["raw_path"])

    if raw_path.exists():
        print(f"Loading local raw dataset from {raw_path}")
        dataset = load_dataset("json", data_files=str(raw_path), split="train")
    else:
        print(f"Downloading dataset '{dataset_name}' from Hugging Face Hub")
        dataset = load_dataset(dataset_name, split="train")

    num_records = cfg["data"].get("num_records")
    if num_records and len(dataset) > num_records:
        dataset = dataset.shuffle(seed=cfg["data"]["seed"]).select(range(num_records))
        print(f"Sliced dataset down to {num_records} records")

    return dataset


def get_tokenizer_with_chat_template(cfg: dict):
    """Load just the tokenizer (fast, no model weights) with the ChatML template
    applied, so formatting here exactly matches what model_utils.py uses for
    training/inference — no drift between the two.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    tokenizer = get_chat_template(tokenizer, chat_template=cfg["model"]["chat_template"])
    return tokenizer


def format_example(example: dict, cfg: dict, tokenizer=None) -> dict:
    """Convert a raw (question, context, answer) row into a single ChatML-formatted
    training string: system schema/instructions -> user question -> assistant SQL.
    """
    question_field = cfg["data"]["question_field"]
    context_field = cfg["data"]["context_field"]
    answer_field = cfg["data"]["answer_field"]

    question = example.get(question_field, "").strip()
    context = example.get(context_field, "").strip()
    answer = example.get(answer_field, "").strip()

    conversation = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nSchema:\n{context}"},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    if tokenizer is not None:
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    else:
        # Fallback plain-ChatML string if tokenizer isn't available (e.g. quick tests)
        text = (
            f"<|im_start|>system\n{conversation[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>"
        )

    return {"text": text}


def prepare_and_save(cfg: dict) -> None:
    raw_dataset = load_raw_dataset(cfg)
    tokenizer = get_tokenizer_with_chat_template(cfg)

    formatted = raw_dataset.map(
        lambda ex: format_example(ex, cfg, tokenizer),
        remove_columns=raw_dataset.column_names,
    )

    split = formatted.train_test_split(
        test_size=cfg["data"]["val_split_ratio"],
        seed=cfg["data"]["seed"],
    )
    train_data, val_data = split["train"], split["test"]

    train_path = Path(cfg["data"]["processed_train_path"])
    val_path = Path(cfg["data"]["processed_val_path"])
    train_path.parent.mkdir(parents=True, exist_ok=True)

    _save_jsonl(train_data, train_path)
    _save_jsonl(val_data, val_path)

    print(f"Saved {len(train_data)} train examples -> {train_path}")
    print(f"Saved {len(val_data)} val examples -> {val_path}")


def _save_jsonl(dataset: Dataset, path: Path) -> None:
    with open(path, "w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prepare_and_save(cfg)


if __name__ == "__main__":
    main()
