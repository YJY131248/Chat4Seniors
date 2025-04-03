python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Mistral \
    --llm_model_path ../model/base_models/Mistral-7B-Instruct-v0.2 \
    --peft_checkpoint_path ../out/car_model/car_mistral_lora_model/checkpoint-645/ \
    --merge_save_path ../model/car_model \
    --log_path ../log/car_model/mistral/lora_model_merge.log
    