"""
Model utilities.

Isolates all model/tokenizer/LoRA loading logic so that:
  - train.py just asks for a "ready to train" model
  - inference.py just asks for a "ready to generate" model
  - swapping base models (Llama -> Mistral -> Qwen) only requires editing config.yaml
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_base_model(cfg: dict):
    """Load the quantized base model + tokenizer as defined in config, and
    apply the ChatML template so prompts are formatted identically across
    data_preparation.py, train.py, and inference.py.

    base_model can be swapped between Llama-3 and Phi-3 (or any Unsloth
    4-bit model) with a one-line change in config.yaml — nothing else here
    needs to change.
    """
    model_cfg = cfg["model"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg["chat_template"])
    return model, tokenizer


def attach_lora(model, cfg: dict):
    """Wrap the base model with LoRA adapters for parameter-efficient fine-tuning."""
    lora_cfg = cfg["lora"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=cfg["data"]["seed"],
    )
    return model


def load_model_for_training(cfg: dict):
    """Convenience wrapper: base model + LoRA adapters, ready for SFTTrainer."""
    model, tokenizer = load_base_model(cfg)
    model = attach_lora(model, cfg)
    return model, tokenizer


def load_model_for_inference(cfg: dict, adapter_path: str):
    """Load the base model and attach a previously trained LoRA adapter for inference."""
    model_cfg = cfg["model"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,   # Unsloth auto-detects saved adapters here
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    FastLanguageModel.for_inference(model)  # enables Unsloth's 2x faster inference path
    return model, tokenizer
