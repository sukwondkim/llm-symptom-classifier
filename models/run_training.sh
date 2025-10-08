#!/bin/bash
# ======================================================
# LoRA Fine-Tuning for Symptom Category Classifier
# ======================================================

# Set devices and train file
export PYTHONPATH=$(pwd)

TRAIN_FILE=data/mock_train.tsv

# Output directory
OUTPUT_DIR=outputs

# Training parameter
MAX_TOKEN_LEN=256
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
EPOCHS=10
VALIDATION_SPLIT=0.2
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_THRESHOLD=0.001

# LoRA parameter
LORA_R=8
LORA_ALPHA=16
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
LORA_DROPOUT=0.1

# Run train.py
mkdir -p $OUTPUT_DIR
python -m models.train \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --max_token_len $MAX_TOKEN_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --base_model microsoft/biogpt \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --validation_split $VALIDATION_SPLIT \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_threshold $EARLY_STOPPING_THRESHOLD
