python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Mistral \
    --llm_model_path ../model/base_models/Mistral-7B-Instruct-v0.2 \
    --peft_checkpoint_path ../out/chat4seniors_model/mistral_dpo_b0.5_lr1e-5_e5/checkpoint-4855/ \
    --merge_save_path ../model/chat4seniors_model \
    --log_path ../log/chat4seniors_model/mistral/lora_model_merge.log
    