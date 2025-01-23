mlx worker launch -- python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type classification \
    --llm_model_name Qwen \
    --llm_model_path ../model/car_lora_qwen2.5_7b \
    --dataset_path ../data/trainset/car_sft_dataset.json \
    --log_path ../log/car_model_eval.log \
    --max_length 1024
    