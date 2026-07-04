# Domain-Specific LLM Fine-Tuning (Text-to-SQL)

Fine-tunes Qwen2.5-1.5B-Instruct on a text-to-SQL instruction dataset formatted in
**ChatML**, using **QLoRA via Unsloth**, then evaluates it against the base model.

Scope: fine-tuning and evaluation only. No orchestration framework (Flyte), no API
deployment. The `app/streamlit_app.py` is a local demo for trying the model
interactively — not a hosted service.

Matches this resume entry:

> **Domain-Specific LLM Fine-Tuning (Phi-3 / Llama-3)** — Hugging Face, Unsloth, PyTorch, QLoRA
> - Engineered a custom instruction-tuning dataset of 10,000+ domain-specific records, formatting raw data into standard ChatML conversational templates.
> - Fine-tuned Llama-3-8B using QLoRA and Unsloth via Google Colab, reducing memory footprint by 70% and accelerating training time.
> - Published quantized LoRA adapters to Hugging Face, achieving a 40% improvement in task-specific accuracy (e.g., SQL generation) compared to the base model.

## Project Structure

```
llm-finetune-project/
├── config/
│   └── config.yaml          # model choice, ChatML template, dataset, hyperparameters
├── data/
│   ├── raw/                 # optional local dataset override
│   └── processed/           # ChatML-formatted train/val splits
├── src/
│   ├── data_preparation.py  # load, slice to 10k+, format as ChatML, split
│   ├── model_utils.py       # Unsloth model+tokenizer loading, LoRA config, chat template
│   ├── train.py             # QLoRA training via SFTTrainer (response-only loss)
│   ├── inference.py         # schema-aware SQL generation from a fine-tuned adapter
│   └── evaluate.py          # SQL exact-match accuracy + base-vs-finetuned comparison
├── app/
│   └── streamlit_app.py     # text-to-SQL chat demo (schema box + SQL output)
├── scripts/
│   ├── run_training.sh
│   ├── run_app.sh
│   └── colab_notebook_link.txt
├── tests/
│   └── test_data_preparation.py
├── requirements.txt
└── README.md
```

## Dataset

[`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
(~78k question + table-schema → SQL pairs). `config.yaml` slices this down to
**10,000 records** to match the resume's dataset size — increase `data.num_records`
if you want to train on more.

Each record is formatted as a 3-turn **ChatML** conversation:
```
<|im_start|>system
You are a text-to-SQL assistant... Schema: CREATE TABLE ...<|im_end|>
<|im_start|>user
How many employees are in the Sales department?<|im_end|>
<|im_start|>assistant
SELECT COUNT(*) FROM employees WHERE department = 'Sales'<|im_end|>
```

## Quickstart (Colab, free T4 GPU)

```bash
pip install -r requirements.txt --break-system-packages

python -m src.data_preparation --config config/config.yaml
python -m src.train --config config/config.yaml
python -m src.evaluate --config config/config.yaml

streamlit run app/streamlit_app.py
```

Or: `bash scripts/run_training.sh`

See `scripts/colab_notebook_link.txt` for exact Colab clone/run commands.

## How each resume claim is produced

| Resume claim | Where it comes from |
|---|---|
| "10,000+ domain-specific records" | `config.data.num_records: 10000`, enforced in `data_preparation.py` |
| "ChatML conversational templates" | `get_chat_template(tokenizer, chat_template="chatml")` in `model_utils.py`, applied identically in `data_preparation.py` and `inference.py` |
| "QLoRA and Unsloth via Google Colab" | 4-bit `load_in_4bit: true` base model + `FastLanguageModel.get_peft_model` LoRA adapters in `model_utils.py`, trained with TRL's `SFTTrainer` |
| "Reducing memory footprint by 70%" | Compare 4-bit QLoRA VRAM usage vs. full fp16 fine-tuning for the same model (log both in `logs/`, e.g. via `torch.cuda.max_memory_allocated()`) — fill in your measured number in the table below |
| "Published quantized LoRA adapters to Hugging Face" | `train.py` calls `model.push_to_hub(repo_id)` when `huggingface.push_to_hub: true` in config |
| "40% improvement in task-specific accuracy (SQL generation)" | `evaluate.py` computes SQL exact-match accuracy for both the fine-tuned adapter and the untouched base model, and reports `improvement_over_base_pct` in `logs/eval_results.json` |

## Results (fill in after training — pulled from `logs/eval_results.json`)

| Metric | Base Model | Fine-Tuned |
|---|---|---|
| SQL exact-match accuracy | - | - |
| ROUGE-L | - | - |
| Improvement over base | - | -% |
| Peak GPU memory (QLoRA vs. full fine-tune) | - | -% reduction |

## Hardware

Runs on a free Google Colab T4 GPU (16GB) in 4-bit for both Llama-3-8B and Phi-3-mini.
