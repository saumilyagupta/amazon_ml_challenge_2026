#!/usr/bin/env bash
# run via: nohup bash /workspace/saumilya/amazon-ml/work/common/gpu_run.sh 7 bash run_xlmr.sh > logs/train_xlmr.log 2>&1 &
set -e
cd /workspace/saumilya/amazon-ml/work/matching/cross_encoder
python3 train_ce.py --model FacebookAI/xlm-roberta-base --epochs 1 --bs 64 --lr 3e-5 --max_len 128 --out models/ce_xlmr
python3 bench_infer.py --model_dir models/ce_xlmr/ep0 --out models/ce_xlmr/bench.json
