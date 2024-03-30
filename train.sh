#!/bin/bash

echo "Model name"
echo $MODEL_NAME
echo "Instance dir"
echo $INSTANCE_DIR
echo "Out dir"
echo $OUTPUT_DIR
echo "Prompt"
echo $PROMPT
echo "LR"
echo $LR
echo "STEPS"
echo $MAX_STEPS
echo "Loss"
echo $LOSS_TYPE

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="$MODEL_NAME"  \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$PROMPT" \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate="$LR" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_validation_images=0 \
  --max_train_steps=$MAX_STEPS \
  --loss_type="$LOSS_TYPE" \
  --mixed_precision="bf16"
