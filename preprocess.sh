#!/bin/bash

rm data/alpaca_data.jsonl
rm -rf data/alpaca
python3 cover_alpaca2jsonl.py \
    --data_path data/alpaca_data.json \
    --save_path data/alpaca_data.jsonl

python3 tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca1 \
    --max_seq_length 200 \
    --skip_overlength  False \
    --chatglm_path THUDM/chatglm2-6B \
    --version v2
    