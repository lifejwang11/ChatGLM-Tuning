#!/bin/bash
python3 finetune.py \
    --dataset_path data/alpaca1 \
    --lora_rank 8 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --max_steps 150 \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output \
    --chatglm_path ~/.cache/huggingface/hub/models--THUDM--chatglm2-6B/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40