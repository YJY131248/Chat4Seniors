mlx worker launch -- python3 ./src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Qwen \
    --llm_model_path ../model/qwen2.5-7b-instruct \
    --peft_checkpoint_path ../out/car_lora_model/checkpoint-3000/ \
    --merge_save_path ../model/car_lora_qwen2.5_7b \
    --log_path ../log/car_lora_merge.log \
    