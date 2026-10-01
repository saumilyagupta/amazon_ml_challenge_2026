#!/usr/bin/env bash
# End-to-end reproduction of the PREVIOUS submission, matcher v2b (the current submission is v3: run_all.sh).
# usage: bash run_all.sh <dataset dir with train/ test/> <work dir> [gpu id | cpu] [--use-shipped-models]
#   default              : full chain, both splits: records -> dense zs + ft -> lexical + unions v1 / v2 -> matcher v1 (record-side
#                          competitor) -> v2b features -> v2b training + decision tuning (R10c) -> test prediction -> validator
#   --use-shipped-models : test split only, with the trained models shipped in resources/models/v1 (competitor) and resources/models/v2b
#                          (no training; the train split is only needed for the per-split records of prepare.py)
# Every step is restartable (finished artefacts are skipped). See README.md for per-step runtimes / memory.
set -euo pipefail
DATA=${1:?dataset dir}; WORK=${2:?work dir}; GPU=${3:-cpu}; SHIP=${4:-}
S=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-/opt/conda/bin/python3}
export CUDA_VISIBLE_DEVICES=""
SPLITS="train test"; [ "$SHIP" = "--use-shipped-models" ] && SPLITS="test"
run_dense() {  # dense retrieval, zero-shot + fine-tuned encoder (split-aware checkpoint), on GPU if given (encode + exact search), else CPU
  if [ "$GPU" != "cpu" ]; then CUDA_VISIBLE_DEVICES=$GPU $PY $S/block.py --work $WORK --split $1 --stage dense --encoders zs ft --device cuda --threads 8
  else $PY $S/block.py --work $WORK --split $1 --stage dense --encoders zs ft --device cpu --threads 8; fi
}
$PY $S/prepare.py --data $DATA --work $WORK --threads 8 --splits $SPLITS                  # 1. records + transliteration view
for SP in $SPLITS; do
  run_dense $SP                                                                             # 2a. dense channels (zs, ft)
  $PY $S/block.py --work $WORK --split $SP --stage lexical union --union v1 v2 --config v2b --threads 8   # 2b. lexical channels + unions v1 / v2
done
for SP in $SPLITS; do $PY $S/features.py --work $WORK --variant v1 --split $SP --threads 16; done     # 3a. v1 features (competitor matcher)
for SP in $SPLITS; do $PY $S/features.py --work $WORK --variant v2b --split $SP --threads 16; done    # 3b. v2b features (incl. explainer)
if [ "$SHIP" = "--use-shipped-models" ]; then
  $PY $S/predict.py --work $WORK --variant v1  --split test --threads 8 --models-dir $S/resources/models/v1                        # 5a
  $PY $S/predict.py --work $WORK --variant v2b --split test --threads 8 --models-dir $S/resources/models/v2b --out $WORK/output/v2b  # 5b
else
  $PY $S/train.py   --work $WORK --variant v1  --threads 8                                  # 4a. matcher v1 (record-side competitor of v2b)
  $PY $S/predict.py --work $WORK --variant v1  --split test --threads 8                     # 5a. v1 test probabilities (competitor table)
  $PY $S/train.py   --work $WORK --variant v2b --threads 8                                  # 4b. 2-fold 2-stage LightGBM + decision layer (R10c)
  $PY $S/predict.py --work $WORK --variant v2b --split test --threads 8 --out $WORK/output/v2b                                     # 5b
fi
$PY $S/validate.py --output $WORK/output/v2b --test-dir $DATA/test --check-ids              # 6. official validator
echo "final files: $WORK/output/v2b/matching_results.tsv  $WORK/output/v2b/candidate_pairs.tsv"
