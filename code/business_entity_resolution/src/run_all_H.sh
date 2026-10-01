#!/usr/bin/env bash
# End-to-end chain of the PREVIOUS submission candidate hybrid H (v1 + v2a; val 0.98587). The submitted model is v2b: see run_all.sh.
# usage: bash run_all_H.sh <dataset dir with train/ test/> <work dir> [gpu id | cpu]
# Every step is restartable (finished artefacts are skipped). See README.md for per-step runtimes / memory.
set -euo pipefail
DATA=${1:?dataset dir}; WORK=${2:?work dir}; GPU=${3:-cpu}
S=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-/opt/conda/bin/python3}
export CUDA_VISIBLE_DEVICES=""
run_dense() {  # dense retrieval on GPU if given (encode + exact search), else CPU
  if [ "$GPU" != "cpu" ]; then CUDA_VISIBLE_DEVICES=$GPU $PY $S/block.py --work $WORK --split $1 --stage dense --encoders zs --device cuda --threads 8
  else $PY $S/block.py --work $WORK --split $1 --stage dense --encoders zs --device cpu --threads 8; fi
}
$PY $S/prepare.py --data $DATA --work $WORK --threads 8
for SP in train test; do
  run_dense $SP
  $PY $S/block.py --work $WORK --split $SP --stage lexical union --union v1 --config v1 --threads 8
done
for V in v1 v2a; do
  $PY $S/features.py --work $WORK --variant $V --split train --threads 16
  $PY $S/features.py --work $WORK --variant $V --split test  --threads 8
done
$PY $S/train.py --work $WORK --variant v1  --threads 8
$PY $S/predict.py --work $WORK --variant v1 --split test --threads 8          # also provides the competitor table for v2a / H
$PY $S/train.py --work $WORK --variant v2a --threads 8
$PY $S/predict.py --work $WORK --variant v2a --split test --threads 8
$PY $S/train.py --work $WORK --variant H   --threads 8                        # combines v1 + v2a probabilities, tunes the decision layer
$PY $S/predict.py --work $WORK --variant H --split test --threads 8 --out $WORK/output/H
$PY $S/validate.py --output $WORK/output/H --test-dir $DATA/test --check-ids
echo "final files: $WORK/output/H/matching_results.tsv  $WORK/output/H/candidate_pairs.tsv"
