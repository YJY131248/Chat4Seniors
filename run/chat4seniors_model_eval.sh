python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Mistral \
    --llm_model_path ../model/chat4seniors_model/mistral \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-mistral/eval_res_dpo.csv \
    --log_path ../log/chat4seniors_model/mistral/dpo_model_eval.log \
    --use_peft_model False\
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2

python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Mistral \
    --llm_model_path ../model/base_models/Mistral-7B-Instruct-v0.2 \
    --dataset_path ../data/trainset/chat4seniors_dpo_trainset_hf_eval.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-mistral/eval_res_hf.csv \
    --log_path ../log/chat4seniors_model/mistral/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2

python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Mistral \
    --llm_model_path ../model/base_models/Mistral-7B-Instruct-v0.2 \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-mistral/eval_res_wo_dpo.csv \
    --log_path ../log/chat4seniors_model/mistral/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2


python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Qwen \
    --llm_model_path ../model/chat4seniors_model/qwen \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-qwen/eval_res_dpo.csv \
    --log_path ../log/chat4seniors_model/qwen/dpo_model_eval.log \
    --use_peft_model False\
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2

python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Qwen \
    --llm_model_path ../model/base_models/Qwen2.5-7B-Instruct \
    --dataset_path ../data/trainset/chat4seniors_dpo_trainset_hf_eval.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-qwen/eval_res_hf.csv \
    --log_path ../log/chat4seniors_model/qwen/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2

python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Qwen \
    --llm_model_path ../model/base_models/Qwen2.5-7B-Instruct \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-qwen/eval_res_wo_dpo.csv \
    --log_path ../log/chat4seniors_model/qwen/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2


python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Llama \
    --llm_model_path ../model/chat4seniors_model/llama \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-llama/eval_res_dpo.csv \
    --log_path ../log/chat4seniors_model/llama/dpo_model_eval.log \
    --use_peft_model False\
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2

python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Llama \
    --llm_model_path ../model/base_models/Llama-3-8B \
    --dataset_path ../data/trainset/chat4seniors_dpo_trainset_hf_eval.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-llama/eval_res_hf.csv \
    --log_path ../log/chat4seniors_model/llama/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2


python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Llama \
    --llm_model_path ../model/base_models/Llama-3-8B \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-llama/eval_res_wo_dpo.csv \
    --log_path ../log/chat4seniors_model/llama/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2


python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name BaiChuan \
    --llm_model_path ../model/chat4seniors_model/baichuan \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-baichuan/eval_res_dpo.csv \
    --log_path ../log/chat4seniors_model/baichuan/dpo_model_eval.log \
    --use_peft_model False\
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2

python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name BaiChuan \
    --llm_model_path ../model/base_models/Baichuan2-7B-Base \
    --dataset_path ../data/trainset/chat4seniors_dpo_trainset_hf_eval.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-baichuan/eval_res_hf.csv \
    --log_path ../log/chat4seniors_model/baichuan/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2


python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name BaiChuan \
    --llm_model_path ../model/base_models/Baichuan2-7B-Base \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-baichuan/eval_res_wo_dpo.csv \
    --log_path ../log/chat4seniors_model/baichuan/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2


python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Yi \
    --llm_model_path ../model/chat4seniors_model/yi \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-yi/eval_res_dpo.csv \
    --log_path ../log/chat4seniors_model/yi/dpo_model_eval.log \
    --use_peft_model False\
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2

python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Yi \
    --llm_model_path ../model/base_models/Yi-1.5-6B \
    --dataset_path ../data/trainset/chat4seniors_dpo_trainset_hf_eval.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-yi/eval_res_hf.csv \
    --log_path ../log/chat4seniors_model/yi/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2


python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type qa \
    --llm_model_name Yi \
    --llm_model_path ../model/base_models/Yi-1.5-6B \
    --dataset_path ../data/trainset/chat4seniors_rlvr_grpo_val.json \
    --save_eval_res_path ../out/chat4seniors_model/gridsearch-yi/eval_res_wo_dpo.csv \
    --log_path ../log/chat4seniors_model/yi/hf_model_eval.log \
    --use_peft_model False \
    --max_new_tokens 512 \
    --do_sample False \
    --top_p 0.9 \
    --temperature 0.7 \
    --repetition_penalty 1.2