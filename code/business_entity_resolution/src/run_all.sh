#!/usr/bin/env bash
# End-to-end reproduction of output/matching_results.tsv + output/candidate_pairs.tsv for the SUBMITTED matcher v3 (configs/v3.yaml).
# usage: bash run_all.sh <dataset dir with train/ test/> <work dir> [gpu id | cpu] [--use-shipped-models]
#   default              : full chain, both splits: records -> dense zs + ft -> lexical + unions v1 / v2 -> matcher v1 (record-side
#                          competitor) -> v3 features (v2b features + decoy block + branch Q14 + France-pack views of France pairs)
#                          -> v3 training + decision tuning (R10c, own competition, one-owner) -> test prediction + label-free decoy
#                          post-pass (config 'postpass'; raw v3 also written to WORK/output/v3_raw) -> validator
#   --use-shipped-models : test split only (+ the train S1 records, needed for the branch IDF of US / India), scored with the trained
#                          models shipped in resources/models/v1 (competitor) and resources/models/v3 (no training)
# Every step is restartable (finished artefacts are skipped). See README.md for per-step runtimes / memory. run_all_v2b.sh = previous chain.
set -euo pipefail
DATA=${1:?dataset dir}; WORK=${2:?work dir}; GPU=${3:-cpu}; SHIP=${4:-}
S=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-/opt/conda/bin/python3}
export CUDA_VISIBLE_DEVICES=""
SPLITS="train test"; [ "$SHIP" = "--use-shipped-models" ] && SPLITS="test"
run_dense() {  # dense retrieval, zero-shot + fine-tuned encoder (split-aware checkpoint), on GPU if given (encode + exact search), else CPU
  if [ "$GPU" != "cpu" ]; then CUDA_VISIBLE_DEVICES=$GPU $PY $S/block.py --work $WORK --split $1 --stage dense --encoders zs ft --device cuda --threads 8
  else $PY $S/block.py --work $WORK --split $1 --stage dense --encoders zs ft --device cpu --threads 8; fi
}
$PY $S/prepare.py --data $DATA --work $WORK --threads 8 --splits train test                # 1. records + transliteration view (both splits:
                                                                                           #    the branch IDF uses the train S1 for US / India)
for SP in $SPLITS; do
  run_dense $SP                                                                             # 2a. dense channels (zs, ft)
  $PY $S/block.py --work $WORK --split $SP --stage lexical union --union v1 v2 --config v3 --threads 8   # 2b. lexical channels + unions v1 / v2
done
for SP in $SPLITS; do $PY $S/features.py --work $WORK --variant v1 --split $SP --threads 16; done     # 3a. v1 features (competitor matcher)
for SP in $SPLITS; do $PY $S/features.py --work $WORK --variant v3 --split $SP --threads 16; done     # 3b. v3 features
if [ "$SHIP" = "--use-shipped-models" ]; then
  $PY $S/predict.py --work $WORK --variant v1 --split test --threads 8 --models-dir $S/resources/models/v1                       # 5a
  $PY $S/predict.py --work $WORK --variant v3 --split test --threads 8 --models-dir $S/resources/models/v3 --out $WORK/output/v3 --pre-postpass-out $WORK/output/v3_raw  # 5b
else
  $PY $S/train.py   --work $WORK --variant v1 --threads 8                                   # 4a. matcher v1 (record-side competitor)
  $PY $S/predict.py --work $WORK --variant v1 --split test --threads 8                      # 5a. v1 test probabilities (stage-2 competitor table)
  $PY $S/train.py   --work $WORK --variant v3 --threads 8                                   # 4b. 2-fold 2-stage LightGBM + decision layer
  $PY $S/predict.py --work $WORK --variant v3 --split test --threads 8 --out $WORK/output/v3 --pre-postpass-out $WORK/output/v3_raw   # 5b
fi
$PY $S/validate.py --output $WORK/output/v3 --test-dir $DATA/test --check-ids               # 6. official validator
echo "final files: $WORK/output/v3/matching_results.tsv  $WORK/output/v3/candidate_pairs.tsv"
