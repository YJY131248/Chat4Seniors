mlx worker launch --gpu=4 --cpu=20 --memory=300 --type=Tesla-V100-SXM2-32GB -- torchrun --nproc_per_node=4 --master_port=29501 ../src/finetune.py \
    --peft_type lora \
    --llm_model_name Mistral \
    --llm_model_path ../model/base_models/Mistral-7B-Instruct-v0.2 \
    --dataset_path ../data/trainset/car_sft_dataset_augmentation.json \
    --log_path ../log/car_model/mistral/lora_model_finetune.log \
    --max_length 1024 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir ../out/car_model/car_mistral_lora_model \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --fp16 \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --dataloader_num_workers 4

mlx worker launch -- python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Mistral \
    --llm_model_path ../model/base_models/Mistral-7B-Instruct-v0.2 \
    --peft_checkpoint_path ../out/car_model/car_mistral_lora_model/checkpoint-645/ \
    --merge_save_path ../model/car_model/mistral_lora_model \
    --log_path ../log/car_model/mistral/lora_model_merge.log
    
mlx worker launch -- python3 ../src/evaluate.py \
    --peft_type lora \
    --task_type classification \
    --llm_model_name Mistral \
    --llm_model_path ../model/car_model/mistral_lora_model \
    --dataset_path ../data/trainset/car_sft_dataset_augmentation.json \
    --save_eval_res_path ../out/car_model/car_mistral_lora_model/eval_res/eval_res.csv \
    --log_path ../log/car_model/mistral/lora_model_eval.log \
    --use_peft_model True \
    --max_new_tokens 10 \
    --do_sample False \
    --top_p 0.1 \
    --temperature 0.1 \
    --repetition_penalty 1.2