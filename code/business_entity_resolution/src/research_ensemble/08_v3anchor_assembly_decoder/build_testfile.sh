#!/bin/bash
# BUILD: submission test file for ensemble <name> via the UNMODIFIED B1 chain (prod_v2c/build/interim_v2b_E13/src), re-targeted with B1_HOME
# (as chain_E06.sh): p = coalesce(ensemble p, v2b p2) over all 83,761,275 union-v2 test pairs (full-length file -> ensemble p everywhere),
# 03_decide.py: (i) v2b shipped-file reproduction (p2 + frozen decoder, must be exact), R10c decide_set with the refit decoder
# results/eval_ENS_<name>_R10c_m0.txt at margin 0 (competition = v1 p2), post-pass (offset-conditioned ADD-decoy veto + strict one-owner by p),
# TSVs, official validator --check-ids; then 15_check_submission.py and 04_gate.py (diff vs v3 / v3x_ash), all writing ONLY under build/<name>/.
# Summary -> build/<name>/SUMMARY.json; progress lines -> results/PROGRESS.log. usage: build_testfile.sh <name>
set -u
NAME=${1:?usage: build_testfile.sh <name>}
H=/workspace/saumilya/amazon-ml/work/matching/ensemble_v1
S=/workspace/saumilya/amazon-ml/work/matching/prod_v2c/build/interim_v2b_E13/src
PY=/opt/conda/bin/python3
BH=$H/build/$NAME; P3=$H/data/test_$NAME.parquet; DEC=$H/results/eval_ENS_${NAME}_R10c_m0.txt
V3F=/workspace/saumilya/amazon-ml/work/matching/prod_v3/output/matching_results.tsv
ASH=/workspace/saumilya/amazon-ml/work/matching/prod_v3x_ash/output_ash/matching_results_ash.tsv
mkdir -p $BH/logs $BH/output
LOG=$BH/logs/build.log
exec > >(tee -a $LOG) 2>&1
prog() { $PY -c "import sys; sys.path.insert(0,'$H/common'); import ens_common as C; C.progress('BUILD', sys.argv[1])" "$1"; }
memavail() { awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo; }
for f in $P3 $DEC; do [ -s $f ] || { echo "BUILD-ABORT missing input $f"; exit 3; }; done
# input sanity: full-length, has column p, no nulls, unique keys are asserted inside 03_decide (non-null join count == file rows)
$PY - "$P3" <<'PYEOF' || { echo "BUILD-ABORT p file check failed"; exit 3; }
import sys, polars as pl
f = sys.argv[1]; lf = pl.scan_parquet(f); sch = lf.collect_schema()
assert 'p' in sch and 's1_idx' in sch and 'cand_idx' in sch, sch
r = lf.select(pl.len().alias('n'), pl.col('p').null_count().alias('nul'), pl.col('p').min().alias('mn'), pl.col('p').max().alias('mx')).collect().row(0, named=True)
print('p file', f, r); assert r['n'] == 83761275 and r['nul'] == 0 and r['mn'] >= 0 and r['mx'] <= 1, r
PYEOF
# one 03_decide.py at a time (this programme): exclusive lock held for the whole build
exec 9>$H/build/.decide.lock
echo "[$(date -u +%H:%M:%S)] waiting for build lock"; flock -x 9
until [ "$(memavail)" -ge 60 ]; do echo "[$(date -u +%H:%M:%S)] MemAvailable $(memavail) GB < 60 -> wait 30 s"; sleep 30; done
OTHER=$(pgrep -af "03_decide.py" | grep -v "$BH" | grep -v pgrep | cut -c1-200)
T0=$(date +%s); START=$(date -u +%H:%M:%S)
echo "BUILD-START $NAME $START MemAvailable $(memavail) GB; other 03_decide running: ${OTHER:-none}"
prog "build $NAME START $START (B1 chain, decoder $(basename $DEC), p = test_$NAME.parquet full-length, out build/$NAME/output/; MemAvailable $(memavail) GB)"
export B1_HOME=$BH B1_MODE=direct B1_THREADS=8 PYTHONDONTWRITEBYTECODE=1
cd $S
$PY $S/03_decide.py --p3 $P3 --p3col p --decoder $DEC --margin 0 --outdir $BH/output --no-band-check --no-diag > $BH/logs/03_decide.log 2>&1
RCD=$?; echo "[$(date -u +%H:%M:%S)] 03_decide rc=$RCD ($(( $(date +%s) - T0 )) s)"; tail -4 $BH/logs/03_decide.log | cut -c1-600
RCC=9; RCG=9
if [ $RCD -eq 0 ]; then
  # independent check and gate statistics (both read-only on the outputs; run concurrently)
  $PY $S/15_check_submission.py $BH/output $P3 p > $BH/logs/15_check_submission.log 2>&1 & PC=$!
  $PY $S/04_gate.py --outdir $BH/output --p3 $P3 --p3col p --extra-ref v3=$V3F --extra-ref v3x_ash=$ASH > $BH/logs/04_gate.log 2>&1 & PG=$!
  wait $PC; RCC=$?; echo "[$(date -u +%H:%M:%S)] 15_check rc=$RCC"
  wait $PG; RCG=$?; echo "[$(date -u +%H:%M:%S)] 04_gate rc=$RCG"
fi
END=$(date -u +%H:%M:%S); WALL=$(( $(date +%s) - T0 ))
cd $H
SUM=$($PY $H/build/src/build_summary.py $NAME $WALL $RCD $RCC $RCG $START $END 2>&1 | tail -1)
echo "$SUM"
prog "build $NAME END $END rc=$RCD/$RCC/$RCG: $SUM"
echo "BUILD-END $NAME rc=$RCD/$RCC/$RCG $END wall ${WALL}s"
[ $RCD -eq 0 ] || exit 2
