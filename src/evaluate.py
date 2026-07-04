"""
Evaluation module.

Runs the held-out validation set through the fine-tuned model and reports:
  - SQL task accuracy: normalized exact-match between generated SQL and the
    reference query (whitespace/case-insensitive comparison — the standard
    proxy metric for text-to-SQL benchmarks like WikiSQL/Spider)
  - ROUGE-L as a softer text-similarity backstop
  - A base-model-vs-fine-tuned comparison, which produces the
    "% improvement in task-specific accuracy" number for the resume

Usage:
    python -m src.evaluate --config config/config.yaml
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from rouge_score import rouge_scorer

from src.config_utils import load_config
from src.model_utils import load_model_for_inference, load_base_model
from src.inference import generate, build_conversation


def normalize_sql(query: str) -> str:
    """Lightly normalize SQL so formatting differences (spacing, case,
    trailing semicolons) don't count as mismatches."""
    query = query.strip().rstrip(";").lower()
    query = re.sub(r"\s+", " ", query)
    return query


def compute_sql_accuracy(predictions: list, references: list) -> dict:
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_sql(pred) == normalize_sql(ref)
    )
    return {"sql_exact_match_accuracy": matches / len(predictions) if predictions else 0.0}


def compute_rouge(predictions: list, references: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return {"rougeL_avg": sum(scores) / len(scores) if scores else 0.0}


def _extract_qca(row: dict):
    """Pull (question, context, reference_sql) back out of a ChatML-formatted
    training row saved by data_preparation.py."""
    text = row["text"]
    system_part = text.split("<|im_start|>user")[0]
    context = system_part.split("Schema:\n", 1)[-1].split("<|im_end|>")[0].strip()
    question = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>")[0].strip()
    reference = text.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>")[0].strip()
    return question, context, reference


def _run_predictions(model, tokenizer, val_dataset, cfg: dict, limit: int = None):
    predictions, references, samples = [], [], []
    rows = val_dataset if limit is None else val_dataset.select(range(min(limit, len(val_dataset))))

    for i, row in enumerate(rows):
        question, context, reference = _extract_qca(row)
        prediction = generate(model, tokenizer, question, cfg, context)

        predictions.append(prediction)
        references.append(reference)

        if i < cfg["evaluation"]["num_qualitative_samples"]:
            samples.append({"question": question, "reference_sql": reference, "predicted_sql": prediction})

    return predictions, references, samples


def run_evaluation(cfg: dict) -> None:
    adapter_path = cfg["training"]["final_adapter_dir"]
    val_dataset = load_dataset("json", data_files=cfg["data"]["processed_val_path"], split="train")

    # --- Fine-tuned model ---
    model, tokenizer = load_model_for_inference(cfg, adapter_path)
    predictions, references, samples = _run_predictions(model, tokenizer, val_dataset, cfg)

    results = {
        **compute_sql_accuracy(predictions, references),
        **compute_rouge(predictions, references),
        "num_examples": len(predictions),
        "qualitative_samples": samples,
    }

    # --- Base model (no LoRA) for comparison, on a smaller subset to save time ---
    if cfg["evaluation"].get("compare_against_base_model"):
        del model
        base_model, base_tokenizer = load_base_model(cfg)
        base_preds, base_refs, _ = _run_predictions(base_model, base_tokenizer, val_dataset, cfg, limit=50)
        base_accuracy = compute_sql_accuracy(base_preds, base_refs)["sql_exact_match_accuracy"]

        ft_accuracy = results["sql_exact_match_accuracy"]
        improvement = ((ft_accuracy - base_accuracy) / base_accuracy * 100) if base_accuracy > 0 else float("inf")

        results["base_model_sql_accuracy"] = base_accuracy
        results["improvement_over_base_pct"] = round(improvement, 1)

    out_path = Path(cfg["evaluation"]["eval_output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Fine-tuned SQL accuracy: {results['sql_exact_match_accuracy']:.3f} | "
          f"ROUGE-L: {results['rougeL_avg']:.3f}")
    if "improvement_over_base_pct" in results:
        print(f"Base model accuracy: {results['base_model_sql_accuracy']:.3f} | "
              f"Improvement over base: {results['improvement_over_base_pct']}%")
    print(f"Full results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
