mlx worker launch -- python3 ../src/inference.py \
    --llm_model_name Qwen \
    --llm_model_path ../model/qwen2.5-7b-instruct \
    --peft_type lora \
    --merge_save_path ../out/car_lora_model/checkpoint-3000/ \
    --use_merge_model True \
    --log_path ../out/car_model_inference.log
    