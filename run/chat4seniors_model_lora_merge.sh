python3 ../src/lora_merge.py \
    --peft_type Mistral \
    --llm_model_name Qwen \
    --llm_model_path ../model/base_models/mistral \
    --peft_checkpoint_path ../out/chat4seniors_model/gridsearch-mistral/mistral_dpo_b0.5_lr1e-4_e5/checkpoint-1942/ \
    --merge_save_path ../model/chat4seniors_model/mistral \
    --log_path ../log/chat4seniors_model/mistral/lora_model_merge.log

python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Qwen \
    --llm_model_path ../model/base_models/Qwen2.5-7B-Instruct \
    --peft_checkpoint_path ../out/chat4seniors_model/gridsearch-qwen/qwen_dpo_b0.5_lr1e-4_e2/checkpoint-1942/ \
    --merge_save_path ../model/chat4seniors_model/qwen \
    --log_path ../log/chat4seniors_model/qwen/lora_model_merge.log

python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Llama \
    --llm_model_path ../model/base_models/Llama-3-8B \
    --peft_checkpoint_path ../out/chat4seniors_model/gridsearch-llama/llama_dpo_b0.5_lr1e-4_e2/checkpoint-1942/ \
    --merge_save_path ../model/chat4seniors_model/llama \
    --log_path ../log/chat4seniors_model/llama/lora_model_merge.log

python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name BaiChuan \
    --llm_model_path ../model/base_models/Baichuan2-7B-Base \
    --peft_checkpoint_path ../out/chat4seniors_model/gridsearch-baichuan/baichuan_dpo_b0.2_lr5e-5_e2/checkpoint-1942/ \
    --merge_save_path ../model/chat4seniors_model/baichuan \
    --log_path ../log/chat4seniors_model/baichuan/lora_model_merge.log

python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Yi \
    --llm_model_path ../model/base_models/Yi-1.5-6B \
    --peft_checkpoint_path ../out/chat4seniors_model/gridsearch-yi/yi_dpo_b0.5_lr1e-4_e2/checkpoint-1942/ \
    --merge_save_path ../model/chat4seniors_model/yi \
    --log_path ../log/chat4seniors_model/yi/lora_model_merge.log