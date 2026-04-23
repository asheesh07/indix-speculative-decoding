#!/bin/bash
set -e

echo "========================================"
echo "Starting Speculative Decoding Pipeline"
echo "========================================"

echo "Step 1: Collecting Data..."
python3 scripts/collect_data.py

echo "Step 2: Training Tokenizer..."
python3 tokenizer/train_tokenizer.py

echo "Step 3: Computing Baselines..."
python3 baselines/compute_baselines.py

echo "Step 4: Training Hindi GPT-2 (20,000 steps)..."
# This step may take hours depending on the GPU
python3 training/train.py

echo "Step 5: Running Speculative Decoding Evaluation..."
python3 speculative_decoding/speculative_decoding.py

echo "Step 6: Comparing Experiments..."
python3 evaluation/compare_experiments.py
python3 evaluation/perplexity.py

echo "========================================"
echo "Pipeline Completed Successfully!"
echo "========================================"
