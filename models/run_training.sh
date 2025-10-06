#!/bin/bash
# ======================================================
# LoRA + DeepSpeed Fine-Tuning for HPO Classifier
# ======================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_FILE=data/train.tsv

# DeepSpeed config
DS_CONFIG=models/ds_config.json

# Output directory
OUTPUT_DIR=outputs

# Training parameter
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
EPOCHS=10

# LoRA parameter
LORA_R=8
LORA_ALPHA=16
LORA_TARGET_MODULES="q_proj,v_proj"
LORA_TASK_TYPE="SEQ_CLS"

# Run train.py with deepseed
deepspeed --num_gpus=4 models/train.py \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --num_train_epochs $EPOCHS \
    --deepspeed $DS_CONFIG \
    --model_name_or_path microsoft/biogpt \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules $LORA_TARGET_MODULES \
    --lora_task_type $LORA_TASK_TYPE \
    --validation_split 0.1 