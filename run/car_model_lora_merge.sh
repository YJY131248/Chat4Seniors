mlx worker launch -- python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Qwen \
    --llm_model_path ../model/base_models/Qwen2.5-7B-Instruct \
    --peft_checkpoint_path ../out/car_model/car_qwen_lora_model/checkpoint-645/ \
    --merge_save_path ../model/car_model/qwen_lora_model \
    --log_path ../log/car_model/qwen/lora_model_merge.log
    