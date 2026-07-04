"""
Inference module.

Loads the fine-tuned adapter and exposes a single `generate()` function used
by both evaluate.py and the Streamlit app — one code path, no duplicated
prompt-formatting logic.

Task: given a database schema (context) + natural language question,
generate the SQL query, matching the ChatML conversation format used in
data_preparation.py / model_utils.py.
"""

import argparse

from src.config_utils import load_config
from src.model_utils import load_model_for_inference
from src.data_preparation import SYSTEM_PROMPT


def build_conversation(question: str, context: str = "") -> list:
    """Must match the ChatML conversation structure used in data_preparation.py."""
    system_content = f"{SYSTEM_PROMPT}\n\nSchema:\n{context}" if context else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


def generate(model, tokenizer, question: str, cfg: dict, context: str = "") -> str:
    conversation = build_conversation(question, context)
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    inf_cfg = cfg["inference"]
    outputs = model.generate(
        **inputs,
        max_new_tokens=inf_cfg["max_new_tokens"],
        temperature=inf_cfg["temperature"],
        top_p=inf_cfg["top_p"],
        use_cache=True,
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Strip everything up to the last assistant turn so only the SQL is returned
    return decoded.split("assistant")[-1].strip().lstrip("\n").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--context", default="", help="Database schema (CREATE TABLE statements)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter_path = cfg["training"]["final_adapter_dir"]

    model, tokenizer = load_model_for_inference(cfg, adapter_path)
    response = generate(model, tokenizer, args.question, cfg, args.context)

    print(f"\nQuestion: {args.question}\nSQL: {response}")


if __name__ == "__main__":
    main()
