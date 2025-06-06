#!/bin/bash

BETAS=(0.1 0.2 0.3 0.5)
LRS=(1e-4 5e-5 1e-5 5e-6)
EPOCHS=(2)

MODEL_NAME="Qwen"
MODEL_PATH="../model/base_models/Qwen2.5-7B-Instruct"
DATA_PATH="../data/trainset/chat4seniors_dpo_trainset.json"
OUTPUT_BASE="../out/chat4seniors_model/gridsearch-qwen"
LOG_BASE="../log/chat4seniors_model/qwen/gridsearch"
DEEPSPEED_CONFIG="../config/ds_stage_2_config.json"


for beta in "${BETAS[@]}"; do
  for lr in "${LRS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do

      RUN_NAME="qwen_dpo_b${beta}_lr${lr}_e${epoch}"
      OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
      LOG_PATH="${LOG_BASE}/${RUN_NAME}.log"

      echo "🧪 Running: beta=${beta}, lr=${lr}, epochs=${epoch}"

      torchrun --nproc_per_node=2 --master_port=29501 ../src/dpo_train.py \
        --llm_model_name ${MODEL_NAME} \
        --llm_model_path ${MODEL_PATH} \
        --dataset_path ${DATA_PATH} \
        --log_path ${LOG_PATH} \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --beta ${beta} \
        --output_dir ${OUTPUT_DIR} \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs ${epoch} \
        --learning_rate ${lr} \
        --lr_scheduler_type cosine \
        --save_strategy epoch \
        --save_total_limit 2 \
        --logging_steps 100 \
        --report_to tensorboard \
        --warmup_ratio 0.05 \
        --deepspeed ${DEEPSPEED_CONFIG} \
        --max_prompt_length 512 \
        --fp16 True

    done
  done
done
