#!/bin/bash

rm data3/alpaca_data.jsonl
rm -rf data3/alpaca
python3 cover_alpaca2jsonl.py \
    --data_path data3/alpaca_data.json \
    --save_path data3/alpaca_data.jsonl

python3 tokenize_dataset_rows.py \
    --jsonl_path data3/alpaca_data.jsonl \
    --save_path data3/alpaca1 \
    --max_seq_length 200 \
    --skip_overlength  False \
    --chatglm_path ~/.cache/huggingface/hub/models--THUDM--chatglm2-6B/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40 \
    --version v2
    