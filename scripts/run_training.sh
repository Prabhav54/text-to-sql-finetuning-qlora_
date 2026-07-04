#!/bin/bash
# One-command training pipeline: prepare data -> train -> evaluate
set -e

CONFIG=${1:-config/config.yaml}

echo "== Step 1/3: Preparing data =="
python -m src.data_preparation --config "$CONFIG"

echo "== Step 2/3: Training (QLoRA) =="
python -m src.train --config "$CONFIG"

echo "== Step 3/3: Evaluating =="
python -m src.evaluate --config "$CONFIG"

echo "Done. Run 'streamlit run app/streamlit_app.py' to try the demo."
