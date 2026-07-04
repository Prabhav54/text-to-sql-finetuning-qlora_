# Text-to-SQL Fine-Tuning (QLoRA)

Fine-tunes Qwen2.5-1.5B-Instruct on a text-to-SQL instruction dataset formatted in
**ChatML**, using **QLoRA via Unsloth**, then evaluates it against the untouched
base model to quantify the improvement from fine-tuning.

**Scope:** fine-tuning and evaluation only. No orchestration framework, no hosted
API. `app/streamlit_app.py` is a local demo for trying the model interactively.

## Project Structure

```
text-to-sql-finetuning-qlora/
├── notebook.ipynb            # end-to-end reproducible run (Colab-ready)
├── config/
│   └── config.yaml           # model choice, ChatML template, dataset, hyperparameters
├── data/
│   ├── raw/                  # optional local dataset override
│   └── processed/            # ChatML-formatted train/val splits
├── src/
│   ├── data_preparation.py   # load, slice, format as ChatML, split
│   ├── model_utils.py        # Unsloth model+tokenizer loading, LoRA config, chat template
│   ├── train.py               # QLoRA training via SFTTrainer (response-only loss)
│   ├── inference.py           # schema-aware SQL generation from a fine-tuned adapter
│   └── evaluate.py            # SQL exact-match accuracy + base-vs-fine-tuned comparison
├── app/
│   └── streamlit_app.py       # text-to-SQL chat demo (schema box + SQL output)
├── scripts/
│   ├── run_training.sh
│   └── run_app.sh
├── tests/
│   └── test_data_preparation.py
├── requirements.txt
└── README.md
```

## Dataset

[`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
— synthetic question + database-schema → SQL triples. `config.yaml` slices this
down to 3,000 records for fast iteration; increase `data.num_records` to train
on more.

Each record is formatted as a 3-turn **ChatML** conversation:
```
<|im_start|>system
You are a text-to-SQL assistant. Given a database schema and a natural language
question, generate the correct SQL query.

Schema:
CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)<|im_end|>
<|im_start|>user
How many employees are in the Sales department?<|im_end|>
<|im_start|>assistant
SELECT COUNT(*) FROM employees WHERE department = 'Sales'<|im_end|>
```

## Quickstart (Colab, free T4 GPU)

Open `notebook.ipynb` in Colab and run top to bottom, or run each step manually:

```bash
pip install unsloth -q
pip install pyyaml rouge-score -q

python -m src.data_preparation --config config/config.yaml
python -m src.train --config config/config.yaml
python -m src.evaluate --config config/config.yaml

streamlit run app/streamlit_app.py
```

## How it works

| Component | Implementation |
|---|---|
| Dataset formatting | ChatML conversations via `get_chat_template(tokenizer, chat_template="chatml")`, applied identically in `data_preparation.py`, `train.py`, and `inference.py` |
| Fine-tuning method | QLoRA — 4-bit quantized base model (`load_in_4bit: true`) + LoRA adapters (`FastLanguageModel.get_peft_model`), trained with TRL's `SFTTrainer` |
| Loss masking | `train_on_responses_only` — loss computed only on the assistant's SQL output, not the schema/question tokens |
| Evaluation | `evaluate.py` runs the held-out validation set through both the fine-tuned adapter and the untouched base model, computing SQL exact-match accuracy for each and the percentage improvement |
| Publishing | `train.py` calls `model.push_to_hub(repo_id)` automatically when `huggingface.push_to_hub: true` in config |

## Results (from `logs/eval_results.json`)

| Metric | Base Model | Fine-Tuned |
|---|---|---|
| SQL exact-match accuracy | - | - |
| ROUGE-L | - | - |
| Improvement over base | - | -% |

## Config-driven design

Everything — model choice, dataset, LoRA rank, hyperparameters — lives in
`config/config.yaml`. Swapping models or datasets is a one-line edit; no code
changes needed elsewhere in the pipeline.

## Hardware

Runs on a free Google Colab T4 GPU (16GB) in 4-bit. With Qwen2.5-1.5B and 3,000
records, a full training run completes in well under 15 minutes.
