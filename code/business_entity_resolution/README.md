# Business Entity Resolution — ML Challenge 2026

**Team KL_converge** · reproducible pipeline: the base matcher **v3** plus the decision-level ensemble chain that produced the submitted file.

## At a glance

| | |
|---|---|
| **Submitted matches** | `output/matching_results.tsv` = **`CLEAN13_FRLONE`** · md5 `9287023c12d622ae9de958cfc7602353` · 1,732,544 rows · 5,858,729 matches |
| **Candidate set** | `output/candidate_pairs.tsv` = blocking union v2 · md5 `56ce4634b2176304f84e1189d13831a7` · 83,761,275 pairs · identical for every upload since #2 |
| **Rebuild the submitted file** | `python3 src/final_reproduce.py` · byte-exact · 1–2 min on CPU · Python ≥ 3.11 + polars ([§10](#10-the-submitted-file-clean13_frlone)) |
| **Rebuild candidates + base matcher** | `bash src/run_all.sh` · one GPU for the dense channels, 16 CPU threads, ≤ 50 GB RAM, ~150 GB disk ([§4](#4-reproducing-matcher-v3-from-raw-data)) |
| **External data, APIs, services** | none ([details](#no-external-data)) |
| **Licences** | team code MIT; every model MIT / Apache-2.0 / BSD / ISC, each ≤ 568M parameters ([§7](#7-models-licences-parameters)) |

## Quick start

**1. Rebuild the submitted file** (CPU, about a minute):

```bash
cd code/business_entity_resolution/src
pip install -r ../requirements.txt                 # only polars is needed for this step
python3 final_reproduce.py                         # asserts sha256 + md5 of the rebuilt file
cmp output_final/CLEAN13_FRLONE/matching_results.tsv ../../../output/matching_results.tsv   # byte-identical
```

**2. Check both output files with the official validator:**

```bash
python3 validate.py --output ../../../output --test-dir <dataset>/test --check-ids
```

**3. Re-run blocking and the base matcher from the raw TSVs** (full data needs a GPU; see [§3](#3-environment-hardware-runtimes) and [§4](#4-reproducing-matcher-v3-from-raw-data)):

```bash
bash run_all.sh <dataset> <work dir> [gpu id|cpu] [--use-shipped-models]
```

## Contents

1. [How the submitted file is built](#1-how-the-submitted-file-is-built)
2. [Folder layout](#2-folder-layout)
3. [Environment, hardware, runtimes](#3-environment-hardware-runtimes)
4. [Reproducing matcher v3 from raw data](#4-reproducing-matcher-v3-from-raw-data)
5. [Smoke test (CPU only)](#5-smoke-test-cpu-only)
6. [Method summary](#6-method-summary)
7. [Models, licences, parameters](#7-models-licences-parameters)
8. [What v3 changed relative to v2b](#8-what-v3-changed-relative-to-v2b)
9. [From v3 to upload #10: the ensemble chain](#9-from-v3-to-upload-10-the-ensemble-chain)
10. [The submitted file (`CLEAN13_FRLONE`)](#10-the-submitted-file-clean13_frlone)

---

## 1. How the submitted file is built

The submitted file is built in three stages. The package re-runs stage A from the raw data; stages B and C are rebuilt byte-exactly from frozen decision tables.

| stage | what it is | leaderboard | rebuilt by |
|---|---|---|---|
| **A. Matcher v3** | blocking union v2 → 294 pair features → 2-stage LightGBM → R10c set decoder → one-owner → decoy post-pass | upload #5, 0.986761 (raw v3) | `run_all.sh`, from the raw TSVs ([§4](#4-reproducing-matcher-v3-from-raw-data)) |
| **B. Ensemble chain** | v3 + cross-encoder and graph members, number residuals, blend decoder, empty-address specialist, France mask, label-free decoy removals | upload #8 0.989292 → upload #10 0.989701 | `ens8_reproduce.py`, from `resources/ens8_frozen/` ([§9](#9-from-v3-to-upload-10-the-ensemble-chain)) |
| **C. Later layers** | CE-NC US / India correction, re-decoded US / India rows, label-free France and US / India removal and addition layers | the submitted file | `final_reproduce.py`, from `resources/final_frozen/` ([§10](#10-the-submitted-file-clean13_frlone)) |

The ensemble members and the CE-NC correction ship as verbatim research code plus models and stored scores; re-running them needs the research tree and a GPU.

### Matcher v3 pipeline

1. **Blocking union v2.** Zero-shot and fine-tuned multilingual-e5-small dense channels; lexical compound-key, address-only and empty-address channels; forward and reverse.
2. **294 pair features.**
   - 89 string / number / competition features on a raw and a dictionary-transliterated view
   - 44 channel features
   - **127 noise-inversion "explainer" features**
   - 12 multi-part-number / decoy-token / admin / provenance features
   - **15 offset-conditioned decoy features**
   - **14 features of a second, independently written explainer**
   - for France pairs, the 43 name-core / address / number features come from a **France canonicalisation pack**
3. **2-fold cross-fit LightGBM stage 1 → stage 2** (+ 13 list / record-competition features + 6 sibling-corroboration features).
4. **R10c learned prefix set-decoder**, with exclusivity against the matcher's **own** stage-2 probabilities, then **strict one-owner**.
5. **Label-free decoy post-pass** (removal only): six-word decoys, "S1 + country tag" and legal-form switches at the generator's decoy house offsets.
6. Output: `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

### Scores of the packaged matchers

Validation = 220,730 entities matched against the full 10.32M-record training pool.

| matcher | files | validation macro F0.5 | public LB |
|---|---|---|---|
| **v3** | `configs/v3.yaml`, `resources/models/v3/`, `run_all.sh` | **0.99001** (0.9900117 with the post-pass, 0.9900100 without; US 0.98979 / India 0.99033); locked 30,000-entity holdout **0.98954**; paired vs v2b **+0.00051 [95% CI +0.00037, +0.00065]** | **0.986761** = upload #5 (raw v3, `predict.py --no-postpass`, md5 998d1912…) |
| v2b | `configs/v2b.yaml`, `run_all_v2b.sh`, `resources/models/v2b/` | 0.98950 | 0.98435 (upload #2) |
| v2a | `configs/v2a.yaml` | 0.98503 | 0.960193 (upload #1) |
| v1, H | `configs/v1.yaml`, `configs/H.yaml` | 0.98391, 0.98587 | not uploaded |

v3 + post-pass (md5 701ec73e…, the final file of `run_all.sh`) was never uploaded.

> **The submitted `output/matching_results.tsv` is not the output of `run_all.sh`.** It is the ensemble chain built on top of v3 ([§9](#9-from-v3-to-upload-10-the-ensemble-chain)–[§10](#10-the-submitted-file-clean13_frlone); validation 0.991851).

### No external data

**No external data, lookups, APIs, geocoders or services are used anywhere.**

- The only inputs are the challenge TSVs. Every word list, dictionary and fine-tuned weight was derived from them.
- All models run offline. Telemetry of the HF / wandb stacks is disabled (`ber/env.py`); nothing is sent off the machine.
- The only network access is the one-time download of the public MIT base model `intfloat/multilingual-e5-small` into the local HF cache. Set `HF_HOME` to a pre-populated cache for a fully offline run.
- **Cloud compute.** Five of the six sibling-aware context cross-encoders used by the later layers ([§10](#10-the-submitted-file-clean13_frlone)) were trained and scored on rented cloud GPUs (Modal); their launch scripts are in `resources/final_frozen/audit_opus995/04_*` and `05_*`. This is compute only: our own models on the challenge data, with no external data, lookups or model APIs.

---

## 2. Folder layout

```
code/business_entity_resolution/
├── README.md                  this file
├── requirements.txt           pinned dependencies
├── LICENSE                    MIT (the team's code)
├── THIRD_PARTY_LICENSES.md    models and libraries
├── docs/figures/              figures of the root README.md
└── src/                       all source code, trained models and frozen tables
```

All intermediate artefacts go to one work directory (`--work`, layout in `ber/paths.py`). Every step skips artefacts that already exist.

### Entry points (`src/`)

| file | step | what it does |
|---|---|---|
| `prepare.py` | 1 | TSVs → records (row index = id), transliteration view, matcher views, lexical views, labels |
| `block.py` | 2 | dense channels (e5 zero-shot `zs` + fine-tuned `ft`: encode + exact search); lexical channels C1 / C2 / C3 + reverse; blocking unions v1 (zero-shot dense member) and v2 (fine-tuned dense member, reverse gate 0.03) |
| `features.py` | 3 | candidate table + pair features (v3: + explainer + additions + decoy block + branch Q14 + France-pack views of France pairs), streamed in 3M-row parquet parts |
| `train.py` | 4 | 2-fold LightGBM stage 1 + stage 2; decision-layer tuning on the validation S1 (incl. the R10c set-decoder with own-probability competition and one-owner) |
| `predict.py` | 5 | score a split (France-gated pack), apply the decision policy, run the label-free decoy post-pass (config `postpass`; `--no-postpass` = raw v3), write the two TSVs (`--models-dir resources/models/v3` = the shipped models) |
| `validate.py` | 6 | official validator (vendored copy in `tools/`) |
| `run_all.sh` | 1–6 | the whole chain for matcher v3 (`run_all_v2b.sh` / `run_all_H.sh`: the previous v2b / hybrid H chains) |
| `ens8_reproduce.py` | – | rebuild upload #10 (the base of the submitted file; also #8, #9 and two variants that were not uploaded) byte-exactly from `resources/ens8_frozen/` (polars + stdlib, < 2 min, sha256 asserted; [§9](#9-from-v3-to-upload-10-the-ensemble-chain)) |
| `final_reproduce.py` | – | rebuild the **submitted** file byte-exactly: #8 / #10 via `ens8_reproduce.py`, then the frozen pair lists of `resources/final_frozen/` (polars + stdlib, 1–2 min, sha256 + md5 asserted; [§10](#10-the-submitted-file-clean13_frlone)) |
| `finetune.py` | optional | re-create a fine-tuned encoder (GPU, ~11 min on a V100) |

### Code and tool folders (`src/`)

| folder | contents |
|---|---|
| `ber/` | core library: env, paths, io, config, splits, labels, translit, text, lexnorm, records, dense, lexical, union, cands, pairfeats, chan, model, decide |
| `ber/` (matcher parts) | `featsets` (per-variant feature lists, France gate); `v2bfeats` (v2b additions + sibling corroboration); `setdecoder` (R10c / R7g set decoders, own competition, one-owner); `v3decoy` (offset-conditioned decoy block) |
| `ber/explainer/` | noise-inversion feature library: views, tables, wordlists, features, runner |
| `ber/branch/` | second explainer: explain_pairs, features, fastpatch memoisation, Q14 runner + IDF |
| `ber/` (France pack) | `francepack` + `packrecords` + `packfeats`: France pack views, 43 features |
| `ber/` (post-pass) | `postpass` + `postpass_rules`: record views, neighbour mining, pattern tables, vetoes |
| `configs/` | `v1.yaml`, `v2a.yaml`, `H.yaml`, `v2b.yaml`, `v3.yaml` (the submitted matcher) |
| `tools/` | `validate_submission.py` (official helper, byte-identical), `score.py` (macro F0.5 + oracle scorer), `make_smoke_dataset.py`, `learn_wordlists.py` (re-learn the explainer word lists), `learn_branch_lists.py` (re-learn `branch_lists/coverage.json`), `test_explainer_views.py` (unit tests on the noise-spec examples) |
| `research_ensemble/` | verbatim **audit copies** (76 files, cmp-verified) of the research scripts of every ensemble-chain step: E06, E13, v3anchor assembly + R10c refit, vr2 band, number residuals, Round C blend decoder, specialist, France mask, #8 assembly, A / R / FR9 removals. Its README maps each file to its original path. Not wired: they need the research tree and a GPU ([§9](#9-from-v3-to-upload-10-the-ensemble-chain)) |

### Resources (`src/resources/`)

| path | contents |
|---|---|
| `translit_dictionary.tsv`, `LICENSE.IndicXlit.txt` | 1,534 Indic words with provenance columns |
| `explainer_wordlists/` | learned per-country word lists / IDF tables of the explainer (51 MB); France anchor pairs |
| `models/e5s_ft_trainsplit/` | fine-tuned e5-small for the TRAIN split (471 MB) |
| `models/e5s_ft_all/` | fine-tuned e5-small for the TEST split (471 MB) |
| `ft_pairs/` | the 150k training pairs of each fine-tuned checkpoint (→ `ft_seen` flags) |
| `models/v3/` | the **submitted** matcher's trained models: `s1_fold{0,1}.txt`, `s2_fold{0,1}.txt`, `R10c_m0.0_own.txt`, `decision.json` (27 MB) |
| `models/v2b/` | upload #2's models (31 MB) |
| `models/v1/` | trained matcher v1 = stage-2 record-side competitor of v2b / v3: s1 / s2 fold models, row model (14 MB) |
| `decoy_lists.json` | per-country decoy / true-copy filler words of the decoy block (France words from unlabelled test) |
| `branch_lists/coverage.json` | filler lists of the second explainer, learned from train-split pairs |
| `postpass_decoy_words.json` | decoy word lists of the post-pass six-word veto: US 24 / India 13 from train, France 6 from test |
| `splits/` | validation / locked / training-sample S1 ids |
| `ens8_frozen/` | 302 MB, with `manifest.json` (sha256 + provenance of every file): `decision/` (NC blend selection, specialist additions, France mask), `removals/` (frozen decoy-removal lists of A / B / R / R_FR9), `ids/` (test id tables), `models/` (the chain's LightGBM decoders, residual seeds, specialist, E06 stage 3, E13), `audit/` (stored E06 cross-encoder test logits, specialist scores, number-residual deltas, evidence features), `provenance/` (research run records) |
| `final_frozen/` | the frozen pair lists of the submitted file vs upload #8 (`CLEAN13_FRLONE/`) with `manifest.json`, plus the audit bundles `audit_cenc/` and `audit_opus995/` (verbatim research code and frozen lists of the later layers; [§10](#10-the-submitted-file-clean13_frlone)) |

---

## 3. Environment, hardware, runtimes

- **Python** 3.13, packages pinned in `requirements.txt` (`pip install -r requirements.txt`). Interpreter used here: `/opt/conda/bin/python3`.
- **CPU.** The whole matcher runs on CPU (polars, rapidfuzz, LightGBM, fork-based explainer workers). Entry points cap BLAS / OpenMP threads (`--threads`, default 8) and set `OMP_WAIT_POLICY=PASSIVE`. Plan on 16 threads and ≤ 50 GB RAM for the full data.
- **GPU** (only for the two dense channels): one V100-32GB. Run `block.py --stage dense --device cuda` with `CUDA_VISIBLE_DEVICES` set; every other step exports `CUDA_VISIBLE_DEVICES=""`.
- **CPU fallback caveat.** `--device cpu` is exact and gives the same results, but it takes days on the full data: encoding the 24.2M records of both splits twice (zs + ft) runs at ~60–70 records/s with 8 CPU threads, plus an exact search of ~1.7e13 scores per split. The CPU path is meant for small pools (the [smoke test](#5-smoke-test-cpu-only)).
- **Reference machine** of the research runs: shared 80-core box (load average 100–200), 375 GB RAM, V100-32GB GPUs.

| step (full data) | wall-clock on the reference machine | peak RSS |
|---|---|---|
| `prepare.py` (both splits, 16 threads) | ~12 min per split | 20 GB |
| dense zs + ft, both splits (GPU): encode 24.2M records per encoder + exact forward top-50 / reverse top-5 | ~20 GPU-min per encoder and split (ft test: 543 s encode + 635 s search) | 20 GB GPU |
| lexical channels + unions v1 / v2 (CPU) | train ~25 min, test ~37 min (union v2 build 5 min) | 35 GB |
| features v1 (competitor matcher): train 48.0M / test 40.0M pairs | ~15 min each at 16 threads | 21 GB |
| train v1 + predict v1 test | ~2h25m | 21 GB |
| features v2b: train 28.3M / test 83.8M pairs, incl. explainer (~20k pairs/s with 16 workers) | ~40 min / ~2 h (estimate: research feature pass 14 / 43 min + separate explainer pass ~24 / ~70 min) | ~45 GB (estimate) |
| train v2b: stage 1 2 × ~15 min (folds in parallel, 8 threads each; 658 / 690 trees), stage 2 2 × ~6 min, table prediction + stage-2 / sibling features ~30 min, decision tuning incl. R10c ~15 min | ~1h15m | ~35 GB |
| predict v2b test: stage 1 50 min, stage-2 + sibling features 14 min, stage 2 42 min, R10c decision 10 min | ~2 h | 30 GB |
| **v3 additions** (research runs, 4–8 threads on the loaded box): decoy block train 5 / test 25 min; branch features at 14–20k pairs/s with 16 workers, train ~25 / test ~70–100 min; France-pack record views of the test split 21 min + pack features of the 12.8M France test pairs; the research materialised these blocks next to the v2b parts (feature build train 23 / test 52 min) | + ~3 h | ~60 GB |
| train v3: stage 1 2 × 25–29 min (557 / 718 trees), stage 2 2 × 10–12 min (252 / 204), decision incl. R10c with own competition ~25 min | ~1h45m | 30 GB |
| predict v3 test: stage 1 ~45 min, stage-2 + sibling features 26 min, stage 2 ~15 min per half (2 workers), decision + one-owner 16 min | ~2 h | 35 GB |
| post-pass (inside `predict.py`): explainer views of the 11.7M test records (~5 min with 8 workers, cached), neighbour mining + pattern tables among the 83.8M scored pairs + vetoes | views ~5 min (300 s measured, cached); patterns + vetoes ~2.5 min (161 s measured with cached views) | ~12 GB (26 GB for the whole `predict.py --from-preds` run) |
| `validate.py --check-ids` | 4 min | 3 GB |

- **Disk:** ~150 GB of intermediates (v3 feature parts: train 16 × 0.43 GB, test 48 × 0.43 GB).
- **Research wall-clock** for v3, from the v2b feature parts to the test file: about 4.5 h including queueing on the shared box.
- **Measured on the CPU smoke slice** (5k S1 per split, ~47k records): fine-tuned dense channel 39.5 min (CPU encoding); union v2 + v2b features + training + R10c + prediction 13.7 min (from the package build notes, which are not shipped).

---

## 4. Reproducing matcher v3 from raw data

Flow: data → union v2 → features → 2-stage training → R10c (own competition, one-owner) → predict → validate.

This chain reproduces the packaged base matcher: the shipped `candidate_pairs.tsv` and the v3 + post-pass file (md5 701ec73e…; with `--no-postpass`, raw v3 = upload #5, 0.986761). The shipped `matching_results.tsv` is the ensemble chain of [§9](#9-from-v3-to-upload-10-the-ensemble-chain)–[§10](#10-the-submitted-file-clean13_frlone), which starts from v3's stored probabilities and is rebuilt from frozen decision tables.

```bash
S=code/business_entity_resolution/src; PY=/opt/conda/bin/python3
DATA=student_resource/dataset            # train/{train_source1,2,3.tsv,train_ground_truth.tsv} test/{test_source1,2,3.tsv}
WORK=/path/with/200GB/free
export CUDA_VISIBLE_DEVICES=""

# 1. Records + transliteration view (12 min per split at 16 threads, 20 GB).
#    Both splits are needed even for test-only scoring (the second explainer's US / India IDF is computed over the TRAIN S1).
$PY $S/prepare.py --data $DATA --work $WORK --threads 16

# 2a. Dense channels, zero-shot (zs) and fine-tuned (ft). The ft checkpoint is split-aware:
#     train -> resources/models/e5s_ft_trainsplit (fine-tuned on train-split pairs only, so validation stays clean),
#     test  -> resources/models/e5s_ft_all (fine-tuned on all train pairs).
for SP in train test; do CUDA_VISIBLE_DEVICES=0 $PY $S/block.py --work $WORK --split $SP --stage dense --encoders zs ft --device cuda; done

# 2b. Lexical channels + unions v1 and v2 (v2 = REC20 with the ft dense member fwd@20 + reverse gated 0.03).
#     On train it flags the fine-tune's own 150k training pairs (resources/ft_pairs/), so their 29,095 S1 are dropped from the training sample.
for SP in train test; do $PY $S/block.py --work $WORK --split $SP --stage lexical union --union v1 v2 --config v3 --threads 16; done

# 3. (optional) Re-learn the learned lists from the provided files (the shipped ones are in resources/).
$PY $S/tools/learn_wordlists.py --work $WORK --gt $DATA/train/train_ground_truth.tsv --out $WORK/explainer_wordlists   # explainer, 10-20 min
$PY $S/tools/learn_branch_lists.py --data $DATA --out $WORK/branch_lists                                              # second explainer, ~10 min

# 4. Matcher v1 = stage-2 record-side competitor (its p1 over the dense table of ALL S1 feeds v3's stage-2 list / record features).
for SP in train test; do $PY $S/features.py --work $WORK --variant v1 --split $SP --threads 16; done
$PY $S/train.py   --work $WORK --variant v1 --threads 8
$PY $S/predict.py --work $WORK --variant v1 --split test

# 5. v3 features: 89 + 44 channel + 127 explainer + 14 additions (2 dropped) + 15 decoy + 14 branch per union-v2 pair.
#    On the test split the France-pack record views are built once (WORK/records/pack_test_*) and the 43 pack features
#    of every France pair are stored as pk_*.
for SP in train test; do $PY $S/features.py --work $WORK --variant v3 --split $SP --threads 16; done

# 6. v3 training: 2-fold cross-fit stage 1 (294 features) + stage 2 (313), lr 0.05, <= 1,500 rounds, early stopping 50,
#    lexical-only negatives 25% / weight 4. Decision tuning on the 220,730 validation S1 with exclusivity against v3's OWN p2
#    (v1's p2 outside the scored S1): threshold / one-to-one / margin / row model / expected-F0.5 / R10c (margin 0) + one-owner.
#    Final policy fixed to R10c:m0.0 (configs/v3.yaml).
$PY $S/train.py   --work $WORK --variant v3 --threads 8

# 7. Test prediction: France-gated pack -> stage 1 -> stage-2 + sibling features -> stage 2 -> R10c with own-probability
#    exclusivity -> one-owner -> label-free decoy post-pass (removal only). --no-postpass writes raw v3 (= upload #5);
#    --pre-postpass-out DIR writes raw v3 as well. Result = v3 + post-pass (701ec73e..., not the shipped file).
$PY $S/predict.py --work $WORK --variant v3 --split test --out $WORK/output/v3 --pre-postpass-out $WORK/output/v3_raw
#    ... or with the SHIPPED trained models (no training at all; steps 1, 2 and 4-5 features for the test split only,
#    v1 predict with its shipped models):
#    $PY $S/predict.py --work $WORK --variant v1 --split test --models-dir $S/resources/models/v1
#    $PY $S/predict.py --work $WORK --variant v3 --split test --models-dir $S/resources/models/v3 --out $WORK/output/v3

# 8. Official validator. $WORK/output/v3/candidate_pairs.tsv is the shipped candidate file; $WORK/output/v3/matching_results.tsv
#    is v3 + post-pass (701ec73e...), NOT the shipped matching file (that one is rebuilt by final_reproduce.py, section 10).
$PY $S/validate.py --output $WORK/output/v3 --test-dir $DATA/test --check-ids
```

- `bash src/run_all.sh $DATA $WORK [gpu id|cpu] [--use-shipped-models]` runs exactly this chain (without the optional step 3).
- With `--use-shipped-models` it processes the test split only (after preparing both splits' records) and scores it with `resources/models/{v1,v3}`.
- Upload #2 (matcher v2b) is reproduced by `run_all_v2b.sh` (same steps with `--variant v2b`).

### What was verified in this package

- **Decision layer.** Own-probability exclusivity, the shipped R10c decoder and one-owner, applied to the stored test probabilities of the research run, reproduce raw v3's `matching_results.tsv` and `candidate_pairs.tsv` byte for byte.
- **Post-pass.** With the packaged post-pass (pattern tables re-derived from the test records), the v3 + post-pass file (the final file of `run_all.sh`, never uploaded; upload #5 was raw v3) is reproduced byte for byte (md5 701ec73e…). The re-derived pattern tables equal the research tables pair for pair.
- **Scoring path.** France gate → stage 1 → stage-2 + sibling features → stage 2 on stored test feature parts with the shipped models reproduces the stored p1 bit for bit on every pair (incl. 485,986 France pairs), and p2 up to exact-tie effects.
- **Feature blocks.** The packaged decoy block, second-explainer features (with a package-built IDF) and France-pack record views / features reproduce the research values.
- **End to end.** The whole v3 chain runs end to end on the CPU smoke slice.
- **Not re-run from this package:** the full-data chain. It was run in the research tree (`work/matching/prod_v3` on top of `prod_v2b`, from which these modules are copied).

---

## 5. Smoke test (CPU only)

```bash
$PY $S/tools/make_smoke_dataset.py --data $DATA --out smoke/data --n_s1 5000 --n_distractors 25000   # 5k train S1 + 5k test S1 slices
$PY $S/prepare.py --data smoke/data --work smoke/work
for SP in train test; do $PY $S/block.py --work smoke/work --split $SP --device cpu --encoders zs ft --union v1 v2 --config <v3 cfg with split.mode=hash>; done
$PY $S/features.py --work smoke/work --variant <v1 cfg> --split train; ... (v1 train / predict, then v3 features / train / predict)
$PY $S/validate.py --output smoke/work/output/v3 --test-dir smoke/data/test --check-ids
$PY $S/tools/score.py --pred smoke/work/output/v3/matching_results.tsv --gt smoke/data/test_ground_truth.tsv
```

- On a subset use `split: {mode: hash}` (the production configs use the shipped validation id lists).
- The smoke slices are drawn from the official TRAIN data, so the "test" slice is in-sample for the `e5s_ft_all` checkpoint.
- The slices contain no France records, so the France gate is a no-op there. The package smoke run therefore also scores the test slice once with `pack_gate: [India]` to exercise the pack code path.
- Smoke scores are sanity checks only. Runtimes and scores of the packaged smoke runs are in the package build notes (not shipped).

---

## 6. Method summary

Details are in the root `README.md`.

- **Records.** TSVs read with quoting disabled; Indic-script words mapped to Latin through the 1,534-word dictionary (an additional view; the raw text is kept); anyascii folding; house numbers compared as integers (zero padding); legal-form / honorific stripping; phonetic skeleton; per-record frequencies; fake-name flag; IDF vocabularies.
- **Blocking union v2** (country-scoped, nothing country-specific):
  - fine-tuned e5-small on `'query: name | address'`: forward top-20 + reverse rank 1, or ranks 2–5 within 0.03 of the record's best S1;
  - lexical C1 (rarest-first compound keys on the transliterated name × address), C2 (address only), reverse C1r / C2r over all S1;
  - name-only char TF-IDF over empty-address records (C3 / C3r);
  - validation: 47.8 candidates per S1, pair recall 0.99686, oracle macro F0.5 0.99909 (union v1 with the zero-shot member: 0.99256 / 0.99766).
- **Features.**
  - v1's 89: rapidfuzz name / address similarities on two views, IDF-weighted overlap and coverage, integer house-number relations, flags, frequencies, dense cosine and ranks, S1-side and record-side competition over ALL S1.
  - 44 channel features of the union.
  - **127 explainer features** that invert the data generator's operations: name / address explained, per-operation flags, number-relation classes, street key, unexplained leftover words with IDF, learned per-country filler / legal / junk-name / street-abbreviation lists (incl. France lists learned from unlabelled test anchor pairs).
  - 14 additions: all-parts multi-part house-number comparison, decoy token, France-safe admin conflict via department → region aliases, provenance, hub flag; minus 7 density-shifting reverse-membership / count features and v2b's number-shift decoy flags.
  - **v3:** + 15 offset-conditioned decoy features (signed house-number offset; offset in the generator's decoy set {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}; "S1 core + one word" and whether that word is a country decoy / filler / country-tag word; their interactions with the offset).
  - **v3:** + 14 France-safe features of a second noise explainer (IDF mass of shared name words, added / leftover word statistics, house-number relation and signed offset, street-word similarity, operation counts / flags).
  - For France pairs, the 43 name-core / address / number / IDF features are recomputed on the France pack views (department → region, saint, bis / ter, French street types, country tags, French legal forms).
  - Stage 2: 13 list / record-competition features + 6 sibling-corroboration features.
- **Model.** LightGBM binary, 127 leaves, min_data_in_leaf 200, feature / bagging fraction 0.8, max_bin 63, lr 0.05, ≤ 1,500 rounds with early stopping 50 on 15% held-out training entities; 2 folds by hash of the S1 id over a 370,905-entity training sample, pooled over dense and lexical-only pairs (lexical-only negatives subsampled to 25% with weight 4); out-of-fold stage-1 probabilities feed stage 2.
- **Decision.**
  - A pair survives iff its p2 is at least the best OTHER S1's probability for the same record (argmax-vs-best-other exclusivity; v3 uses its OWN p2 of the other S1, v2b used matcher v1's).
  - Per S1, a LightGBM regressor (R10c) predicts the realised F0.5 of each prefix k = 0..10 of the p2-sorted survivors, and the best prefix is output.
  - Finally every S2/S3 record keeps only its highest-p2 selecting S1 (strict one-owner; 0 conflicts remain on test).
  - The decoder is trained on the training sample's out-of-fold scores; the policy was chosen on validation among six families × two competition rules.
- **Post-pass** (removal only, label-free). A selected pair is vetoed if the record sits on the S1's street at house S1 + k, k in D = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}, and its name is the S1 name + one country decoy word (all countries), + "France" (France), or has a switched / added legal form (France; US only for k ≥ 3) with all other number parts equal. Test: 5,873 pairs removed (France 4,963 / US 909 / India 1); validation +1.7e-6 (neutral).

---

## 7. Models, licences, parameters

| component | licence | parameters | use |
|---|---|---|---|
| LightGBM 4.7.0 gradient-boosted trees | MIT | v3 stage 1: 2 × 557 / 718 trees × 127 leaves (8–10 MB each); stage 2: 2 × 252 / 204 trees (3–4 MB); R10c decoder 600 trees × 63 leaves (3.6 MB); v2b and v1 competitor models of similar size | matcher, set decoder |
| `intfloat/multilingual-e5-small` (HF snapshot 614241f6) | MIT | 118M | zero-shot dense channel (union v1 member, zero-shot cosine features, v1 competitor matcher) |
| `resources/models/e5s_ft_trainsplit` and `e5s_ft_all` = multilingual-e5-small contrastively fine-tuned (1 epoch, MultipleNegativesSymmetricRankingLoss, 150k ground-truth pairs of the PROVIDED training labels; train-split S1 only / all train S1) | MIT (derived) | 118M each | dense member of union v2: train split / test split |
| AI4Bharat IndicXlit v1.0 (Indic → English transliteration) | MIT | ~11M | used ONCE offline to propose 165 of the 1,534 dictionary entries (restricted to the dataset's Latin vocabulary); 1,363 entries were read off the train ground-truth alignment, 6 manual. Only the dictionary is shipped (`resources/LICENSE.IndicXlit.txt`) |
| anyascii 0.3.3 | ISC | – | accent / script folding (no GPL `unidecode` anywhere; the second explainer folds with the standard library's `unicodedata`) |
| sparse_dot_topn 1.2.0 | Apache-2.0 | – | sparse top-k for the TF-IDF channels |
| rapidfuzz 3.14.6 | MIT | – | string similarities |
| polars / numpy / scipy / scikit-learn / pandas / pyarrow / PyYAML / torch / transformers / sentence-transformers | MIT / BSD / Apache-2.0 | – | infrastructure |
| `FacebookAI/xlm-roberta-base` fine-tuned as a pairwise cross-encoder (E06) | MIT | 278M | **used in the submitted file**: ensemble member E06 ([§9](#9-from-v3-to-upload-10-the-ensemble-chain); band of uncertain pairs only, scored on GPU). Not part of matcher v3 (there it was evaluated and set aside: in-band AUC 0.791 vs 0.917 for p2). The fine-tuned weights (3 × 1.1 GB) are not shipped; the stored test logits are (`resources/ens8_frozen/audit/ce_xw_*.parquet`) |
| LightGBM models of the ensemble chain (`resources/ens8_frozen/models/`) | MIT | number residuals 3 × ~4 MB, R10c refit 3.6 MB, evidence decoder 3.6 MB, specialist 2.8 MB, E06 stage 3 2 × 1.1 MB, E13 2 × 2.5 MB | [§9](#9-from-v3-to-upload-10-the-ensemble-chain) |
| team code: second noise explainer (`ber/branch/`, from a team member's reverse-engineering branch) | team's own code, MIT (`LICENSE`) | – | 14 v3 features |

The later layers of [§10](#10-the-submitted-file-clean13_frlone) add further fine-tuned cross-encoders (XLM-R-base CE-XL, `jhu-clsp/mmBERT-base`, `BAAI/bge-reranker-v2-m3`; MIT / Apache-2.0, each ≤ 568M parameters); they are described there.

- **Limits.** Every model is far below the 8B-parameter limit and MIT / Apache-2.0 / BSD / ISC licensed (`THIRD_PARTY_LICENSES.md`; the team's own code is MIT, `LICENSE`).
- **Training data.** No external data were used to train anything: the fine-tuned encoders, the dictionary, the explainer / branch / decoy word lists and all LightGBM models were learned from the provided files only.
- **Static tables in the code** (general knowledge, no lookup, no service): US state and Indian state abbreviations, street-type abbreviations, legal-form lists, and for the France pack the 13 French regions with their 101 departments.

### Transductive, label-free uses of the unlabelled test inputs

All of these are declared in the root `README.md`. No test label is used or created.

- France word lists and statistics: the explainer's France lists (from test anchor pairs), the 6-word France decoy list and the second explainer's France IDF;
- the post-pass pattern tables (derived at run time from the test records and scored pairs, not shipped) and its France decoy words;
- split-level record statistics of the test split (frequencies, IDF, record-side competition, sibling features), exactly as on the training split;
- **in the ensemble chain of the submitted file**, further label-free, test-derived decision tables: the France legal / country-tag mask, US decoy-removal lists from mirror-controlled ±k offset families, the France SWAP_GEN pseudo-label guard, and test-side frequency features of the specialist. Its last two removal steps were chosen with public-leaderboard feedback ([§9](#9-from-v3-to-upload-10-the-ensemble-chain)).

---

## 8. What v3 changed relative to v2b

| change | code | resources | effect on validation (R10c) |
|---|---|---|---|
| **A.** 15 offset-conditioned decoy features replace v2b's `dec_num_shift` / `dec_flag` | `ber/v3decoy.py`, `features.py` (`features.v3_decoy`) | `resources/decoy_lists.json` | +0.00029 (own competition + one-owner: 0.98962 → 0.98991; +0.00030 with v1 competition) |
| **B.** exclusivity against the own p2 + strict one-owner | `ber/setdecoder.py` (`own_competition`, `one_owner`), `train.py` / `predict.py` (`decision.competition: own`, `decision.one_owner`) | `resources/models/v3/R10c_m0.0_own.txt` | +0.00017 [+0.00008, +0.00025] on top of A + D |
| **C.** France-gated France pack (43 features of France pairs from the pack views) | `ber/francepack.py`, `ber/packrecords.py`, `ber/packfeats.py`, `features.py` (pk_* columns), `ber/featsets.pack_gate_expr`, `predict.py` | – (static tables in code) | 0 by construction (France proxies: see the root `README.md`) |
| **D.** 14 France-safe features of the second explainer | `ber/branch/`, `features.py` (`features.branch_q`) | `resources/branch_lists/coverage.json` | +0.00010 (0.98991 → 0.99001; +0.00002 with v1 competition); stage 1 alone A + D +0.00064 |
| **E.** label-free decoy post-pass (removal only) after the decision | `ber/postpass.py`, `ber/postpass_rules.py`, `predict.py` (config `postpass`, `--no-postpass`, `--pre-postpass-out`) | `resources/postpass_decoy_words.json` | +1.7e-6 (0.9900100 → 0.9900117); test: 5,873 pairs removed |

**Total:** 0.98950 (v2b) → **0.99001** (+0.00051 [+0.00037, +0.00065]). Everything is switched on in `configs/v3.yaml`; `configs/v2b.yaml` leaves it off.

Open items from the research tree:
- a separate flag for decoy offsets in D minus {1, 2} (legal-switch pattern);
- recomputing the two France-unsafe branch features with French address lists;
- a feature-level fix for the US / India legal-switch decoys at +1 / +2 (no rule can separate them from true copies with a mistyped number).

---

## 9. From v3 to upload #10: the ensemble chain

Upload #10 is the base of the submitted file: the #8 ensemble chain plus label-free decoy removals.

### Leaderboard history up to #10

Public board; `candidate_pairs.tsv` is identical from #2 on.

| upload | file | validation macro F0.5 | public LB | relation to the packaged code |
|---|---|---|---|---|
| 1 | v2a | 0.98503 | 0.960193 | `configs/v2a.yaml` |
| 2 | v2b (R10c decoder) | 0.98950 | 0.98435 | `configs/v2b.yaml`, `run_all_v2b.sh`, `resources/models/v2b/` |
| 4 | v3x_ash = v2b + residual specialist | 0.99001 | 0.984761 | research only |
| 5 | **matcher v3** (raw, no post-pass) | 0.99001 | 0.986761 | `run_all.sh` / `predict.py --no-postpass` (md5 998d1912…); v3 + post-pass (701ec73e…) never uploaded |
| 6 | v3anchor ensemble = v3 + [E06 − v2b] + [E13 − v2b] in logit space, R10c refit, decoy veto, one-owner | 0.990992 | 0.988419 | steps 1–3 below |
| 7 | vr2 consensus (v3anchor + 3-seed residual where the seeds agree) | 0.991176 | 0.988842 | superseded; not in the lineage of the final file |
| 8 | `nc_specialist_legal_fr` = Round C blend "NC" + empty-address specialist + France legal / country-tag mask | 0.991355 | 0.989292 | steps 4–8; `ens8_reproduce.py --target P8` |
| 9 | `A_US_LEG` = #8 minus 4,294 US legal-form decoy pairs | 0.991320 | 0.989495 | step 9; `--target A_US_LEG` |
| **10** | **`R_US_LEGADD_FR9`** = #9 minus 809 extended US removals minus 1,445 France SWAP_GEN pairs = **#8 minus 6,548 pairs** | 0.991316 | **0.989701** | step 9; `--target R_US_LEGADD_FR9` (default) = upload #10, the base of the shipped file ([§10](#10-the-submitted-file-clean13_frlone)) |

- Validation numbers are on the 220,730-entity split. France has no labels, so the France-only steps are validation-neutral by construction.
- The removal steps 9 cost −0.000035 / −0.000040 on validation and gained +0.000203 / +0.000206 on the public board.
- **The choice between A, B, R and R_FR9 was made with public-leaderboard feedback** (#9 confirmed A before #10 was uploaded). This is also stated in the methodology.

### The chain, step by step

`s1_idx` / `cand_idx` = row indices of `resources/ens8_frozen/ids/test_{s1,s23}.parquet`. Every step is removal-only or one-owner-preserving on the 83,761,275 union-v2 candidates, so `candidate_pairs.tsv` never changes.

| # | step | what it computes | code | how it is reproduced here |
|---|---|---|---|---|
| 1 | members v2b, v3 | stage-2 probabilities of the packaged matchers on all candidates | `predict.py --variant v2b\|v3 --score-only` | **recomputable by the package** (`run_all.sh`; re-scoring with the shipped models reproduces p1 bit for bit and p2 up to 166 tie-order differences in 2.9M checked pairs). The chain was frozen on the research run's STORED probabilities, which is why the later steps ship as decision tables |
| 2 | member E06 | XLM-R-base cross-encoder (fine-tuned on the v2b uncertain band, GPU) + LightGBM stage 3 → p3b on the band | `research_ensemble/06_E06_cross_encoder_stage3/` | research tree + GPU; the stored test logits and the stage-3 models are shipped (`ens8_frozen/audit/ce_xw_*.parquet`, `models/s3_xw_p3b_fold*.txt`), so the member can be rebuilt on CPU from them (`13_test_apply.py`) |
| 3 | member E13, v3anchor blend, R10c refit, decoy veto, one-owner (= upload #6) | graph / sibling specialist on the v2b band; `logit(v3) + [logit(E06) - logit(v2b)] + [logit(E13) - logit(v2b)]`; R10c prefix decoder refit on the sample OOF; ADD-decoy veto; strict one-owner | `research_ensemble/07_E13_graph_specialist/`, `08_v3anchor_assembly_decoder/` | research tree (`members_test.parquet` 1.8 GB, `test_v3anchor.parquet` 0.6 GB, not shipped); models `ens8_frozen/models/graph_cv2_fold*.txt`, `eval_ENS_v3anchor_R10c_m0.txt` |
| 4 | test band + competitor table | vr2 band table of the 3,325,555 uncertain v3anchor pairs (from the 23 GB v3 feature parts) and the v1 competitor table `p_other` for all pairs | `research_ensemble/09_vr2_test_infrastructure/`, `02_v1_competitor_table/` | research tree only (inputs 24 GB) |
| 5 | address-number residuals | 3 LightGBM seeds on parsed house / unit numbers of the band pairs; mean logit delta applied inside a missing-number gate | `research_ensemble/10_v2shash_number_residuals/`, `11_round_c_blend_decoder/release.py prepare` | models `ens8_frozen/models/numbers_{11,29,47}.txt`; the resulting deltas ship as `audit/test_corrections.parquet` |
| 6 | Round C blend decode "NC" (5,867,539 pairs) | per S1, prefix k = 0..10 of the corrected p-sorted survivors chosen by 0.75 × R10c(v3anchor) utility + 0.25 × evidence-decoder utility; ADD-decoy veto (789,457 same-street S1 + one-decoy-word pairs at offsets in D); global one-owner | `research_ensemble/11_round_c_blend_decoder/` (`release.py infer`, `run.py::pick`, `oof_decoder.py`, `v3_postpass/pp.py`) | result shipped as **`ens8_frozen/decision/test_selected.parquet`**; models `eval_ENS_v3anchor_R10c_m0.txt`, `decoder_evidence.txt`; evidence features `audit/test_evidence.parquet`. The ADD veto set equals the package's `ber.postpass` six-word plus-set (788,527 of 789,457 pairs lie in the candidates, 0 differences) |
| 7 | empty-address specialist (+2,303 US / India pairs) | LightGBM residual (seed 61, 147 features incl. test-side `n_core_shared` / `full_form_s1_count` frequencies) on unowned empty-address records with anchor p > 0.03; added at p ≥ 0.80 with an owner gate, alpha 1.0 | `research_ensemble/12_empty_address_specialist/` | result shipped as **`decision/test_nc_robust_additions.parquet`**; model `models/specialist_61.txt`, scores `audit/specialist_test_scores.parquet` |
| 8 | France legal / country-tag cleanup (−2,910 pairs) = **upload #8** | mask of France pairs at decoy offsets +k in D whose name is the S1 name + "France" (CTAG) or a legal-form switch / addition (LEG3, LEG12), all other number parts equal; the mirror −k side is ~100× rarer | `research_ensemble/13_france_cleanup_mask/`, `15_build_upload8/build_candidate.py` | mask shipped as **`decision/mask_ppLegal_FR.parquet`** (95,069 pairs). It equals the package's `ber.postpass` FRcountry + FRlegal plus-sets (24,229 + 70,840, 0 differences), i.e. it is re-derivable with the packaged post-pass rules |
| 9 | label-free removals A / R / FR9 = **upload #10** | A: 4,294 US pairs at +k with an added / switched legal form (LEG12 + LEG3) on S1 that keep ≥ 1 other match; R: + 809 US pairs whose copy dropped the unit / suite number (mirror-controlled ±k offset families, `S1_mirror_sweep`); FR9: 1,445 France generic-word-swap pairs (pseudo-label family SWAP_GEN) where the S1 already owns a strict anchor | `research_ensemble/16_removal_descendants/` | lists shipped as **`removals/<target>/removed_pairs.tsv`**; the A rule equals the package's `pat_test_US` legal table at `allnum & k in +D` intersected with the #8 selection on S1 with ≥ 2 selections (4,294, 0 differences) |

### Exact commands

From `code/business_entity_resolution/src/` (any Python ≥ 3.11 with polars; no other dependency; ~2 GB RAM, 4 threads):

```bash
PY=/opt/conda/bin/python3
$PY ens8_reproduce.py                          # -> output_ens8/R_US_LEGADD_FR9/matching_results.tsv, sha256 asserted == 2e84d62e... (upload #10)
$PY ens8_reproduce.py --all --check-candidate ../../../output/candidate_pairs.tsv     # all five targets + the candidate file's sha256
$PY ens8_reproduce.py --target P8              # upload #8 (sha256 7d04f128...); also --target A_US_LEG (#9), B_US_LEG_FR9, R_US_LEGADD
$PY validate.py --output output_ens8/R_US_LEGADD_FR9 --test-dir <dataset>/test           # official validator (copy candidate_pairs.tsv next to it for --check-ids on both)
# the shipped output/matching_results.tsv is rebuilt by final_reproduce.py (section 10)
```

What the script does:
1. re-hashes the shipped input tables against `ens8_frozen/manifest.json`;
2. rebuilds the #8 selection = NC selection + specialist additions − France mask, with the assertions of the research `build_candidate.py` (additions unowned, US / India, same-country; mask France-only; one owner per record);
3. maps the frozen `removed_pairs.tsv` ids to indices, asserts every removed pair is in #8 and the count matches the manifest, and anti-joins;
4. writes the TSV with the writer of the research `release.py::write` (polars `write_csv`, tab, never quoted, ids sorted inside a row);
5. asserts the sha256.

When this package was built, all five targets matched: #8 03150a02…, A e63f7fe3…, B 3648573e…, R 9b169608…, R_FR9 66ae359f… (package build record, not shipped).

### Upload #10 statistics

97,964,972 B · 1,732,544 rows · 99,791 empty (5.76%) · **5,860,384 matches** (3.383 per entity) · 0 records matched to more than one S1 · every match inside the 83,761,275 candidates · official validator PASS with `--check-ids`.

| country | S1 | matches | empty |
|---|---|---|---|
| US | 663,106 | 2,248,511 | 38,229 |
| India | 809,986 | 2,741,745 | 46,677 |
| France | 259,452 | 870,128 | 14,885 |

### What the package does NOT reproduce from raw data

- Steps 2–7 need the research tree (~150 GB of intermediates). The E06 tokenised inputs were deleted, its 3 × 1.1 GB fine-tuned XLM-R weights are not shipped, and the CE scoring needs a GPU.
- The members were frozen on the stored probabilities of the research run.
- The competition's "regenerate both output files from the training / test data using only what is in this folder" therefore holds for `candidate_pairs.tsv` and for the base matcher v3 (upload #5). The submitted `matching_results.tsv` is regenerated from the shipped frozen decision tables (byte-exact), with the research code shipped as audit copies.
- Everything downstream of the members is deterministic set logic; nothing in the chain trains on or looks at any test label (there are none).

---

## 10. The submitted file (`CLEAN13_FRLONE`)

Upload #10 + later label-free France removal layers + the CE-NC US / India correction re-decoded by the 4-backbone head ALLCE_A1A2A5wM1_all_t031.

### What ships

| | |
|---|---|
| file | `output/matching_results.tsv` = **`CLEAN13_FRLONE`**, the final submitted file |
| md5 | `9287023c12d622ae9de958cfc7602353` |
| sha256 | `a4b2e4520aded0635cf42d2f67420ebfce56ea7ab82f4e94a2d83eb913e78b05` |
| size | 97,943,819 B |
| lineage | upload #16 `CLEAN8_FR2X` (md5 `955936b0c018e052a1d175a9ef918d86`) minus 406 France pairs (layers L30 / L31 below) |
| frozen lists | upload #8 minus `resources/final_frozen/CLEAN13_FRLONE/removed_vs8.tsv` (19,846 pairs) plus `added_vs8.tsv` (11,643 pairs); `layers_vs8.tsv` says which layer produced each listed pair |

### Leaderboard history after #10

Continues the table of [§9](#9-from-v3-to-upload-10-the-ensemble-chain). Validation is on the US / India split; France has no labels.

| upload | file | validation | public LB | layers |
|---|---|---|---|---|
| 11 | `FR9b_on10` = #10 minus 757 France cross-encoder-only look-alikes | 0.991316 | 0.98978 | L3 |
| 12 | `FB_onN1s` = #11 + CE-NC on US / India + France −1,309 look-alikes (L4a–c) | 0.991481 | 0.99017 | L4a–c, L5 |
| 13 | `HEADM1_FRSYN` = #12 France rows − 1,402 UNION keyword / FR_UNION look-alikes (L8, L8b) − 283 FR-SYNTH veto (L10); US / India re-decoded by head M1_add_t031 | 0.991830 | 0.990774 | see the layer tables |
| 16 | `CLEAN8_FR2X` = every layer below except L30 / L31 | 0.991851 | 0.991356 | L1–L29 |
| **–** | **`CLEAN13_FRLONE`** (shipped) = #16 minus 406 France pairs (L30, L31; validation unchanged) | 0.991851 | **see board** | all layers below |

### Layers relative to upload #8

Counts are pairs; the layer names are those of `layers_vs8.tsv`. The full rule text of every layer is in `resources/final_frozen/manifest.json` (`targets.CLEAN13_FRLONE.layer_docs`).

**France removals** (label-free, removal-only; see [France: no training labels](#france-no-training-labels))

| layer | first on the board | pairs vs #8 | what it removes |
|---|---|---|---|
| `L2_FR_SWAPGEN_ANCHORED` | #10 | −1,445 | generic-word swaps (SWAP_GEN) whose S1 already owns a strict anchor |
| `L3_FR_CE_ONLY_LOOKALIKE` | #11 (LB 0.98978) | −757 | look-alikes selected only because of the cross-encoders (both base matchers reject them) |
| `L4a_FR_SHAREDKEY_SWAP` | #12 (LB 0.99017) | −278 | shared-key one-generic-word swaps accepted only by the cross-encoder chain |
| `L4b_FR_GENERIC_LOOKALIKE` | #12 | −420 | generic look-alikes flagged by a consensus of label-free detectors |
| `L4c_FR_SAMENAME_COLLISION` | #12 | −611 | records whose name equals another S1's name at a different street / number / department |
| `L6_FR_P1_LONE_CE_ONLY` | new | −151 | S1 whose only selected pairs are cross-encoder-only generic swaps at the same address (151 S1 emptied) |
| `L8_FR_KEYWORD_SWAP` | new | −661 | category-keyword swaps at the same address (Club / Comite, Amis / Amicale, …), flagged by ≥ 2 of 3 detectors |
| `L8b_FR_UNION_FU` | new | −743 | number / street mismatches while the S1 owns its anchor, keyword swaps at +k offsets, near-typos of word pairs known to be false |
| `L10_FR_SYNTH_VETO` | #13 (LB 0.990774) | −283 | FR-SYNTH veto ([below](#france-no-training-labels)) |
| `L14_GSW` | new | −112 | same-address generic-word swaps vetoed by the country-blind FR_A matcher (p2 < 0.3) |
| `L15_U2T4pluskne`, `L16_U1T2LUNlone`, `L17_U2T4plusklone`, `L18_FRCEFR2RL` | new | −105, −76, −10, −289 | frozen swap / French-CE removal lists of the research tree (L16–L18 may empty their S1) |
| `L29_FR2X` | new | −214 | context CE < −5 and FRCE1 < −4 |
| `L30_FR_THREE_MODEL_VETO` | new | −350 | context CE, FRCE1 and FRCE2 all reject the pair (four threshold steps: 69 + 89 + 111 + 81) |
| `L31_FR_LONE_THREE_MODEL` | new | −56 | an S1's last remaining pair when all three scores are < −4 (empties 56 S1) |

**US / India removals**

| layer | first on the board | pairs vs #8 | what it removes |
|---|---|---|---|
| `L1_US_LEGAL_ADD_DECOY` | #9 / #10 | −4,967 | US legal-form-added / switched pairs at the generator decoy offsets +k (4,294, upload #9) and their suite / unit-dropped variants (809, #10); mirror-controlled (+k in D vs −k vs +k not in D) |
| `L5_CENC_US_INDIA` | #12 | −1,959 | CE-NC correction on US / India ([below](#ce-nc-correction)): pairs of #8 it drops |
| `L9_NEW_UNATTRIBUTED` (India) | new | −1,310 | India pairs of #8 that the re-decoded India rows drop beyond the CE-NC list ([below](#us--india-rows)) |
| `L9_NEW_UNATTRIBUTED` (US) | new | −1,882 | US pairs of #8 that the re-decoded US rows drop beyond the CE-NC list ([below](#us--india-rows)) |
| `L11_CEXLV` | new | −1,020 | US / India CE-XL confident-pair veto: selected US / India pairs whose CE-XL cross-encoder probability (XLM-R base fine-tune, MIT) is below 0.002; an S1 whose every scored selected pair is flagged keeps its highest-probability pair (never empties an S1); removal-only |
| `L12_E2S` | new | −1,092 | Shared-name empty-address rule: drop empty-address selected records of S1 whose raw-normalised name is shared by ≥ 2 same-country S1 when the S1 keeps ≥ 1 non-empty-address selected record; label-free, removal-only |
| `L19_VETO2` | new | −366 | US / India two-CE veto (CE-XL p < 0.02 AND mmBERT-CE p < 0.1), never emptying an S1; removal-only |
| `L20_CTXV` | new | −129 | US / India context-CE ensemble veto (mean logit of 6 sibling-aware context CEs < −4), never emptying an S1; removal-only |
| `L25_ENSARM` | new | −284 | Head swap to ENSA_h4_bag3 (US / India): pairs the CTXAVG-head decode selected and the ENSA_h4_bag3-head decode drops (3-seed bag of the t031 residual head over HEADALL4 features + the 6 context-CE arms as per-arm and ensemble columns); removal |
| `L28_RMX` | new | −276 | Uncertain empty-address removal (US / India): drop a selected empty-address record with ENSA head p < 0.8 and CE-XL logit in [−2, 0) when its S1 keeps ≥ 1 other selected record (never empties an S1); removal-only |

**Additions and kept pairs (US / India only; no France pair is ever added)**

| layer | first on the board | pairs vs #8 | what it adds |
|---|---|---|---|
| `L5_CENC_US_INDIA` (keep) | #12 | +136 | #10 removals that the CE-NC / re-decoded US-India selection keeps (in #8 and in this file) |
| `L5_CENC_US_INDIA` (add) | #12 | +1,889 | CE-NC / frozen-fuzzy US / India additions (records no S1 owns after the removals; same country) |
| `L9_NEW_ADD` | new | +5,862 | additions outside the CE-NC list (India 3,389, US 2,473; same country, unowned records; from the re-decoded US / India rows, [below](#us--india-rows)) |
| `L13_ADDT` | new | +932 | Unique-name empty-address additions: an unowned empty-address record added to an S1 whose normalised name is unique in its country when the head probability is > 0.7 and the record name is a typo / word-swap of the S1 name; one owner per record; the CE-XL veto wins |
| `L21_G1` | new | +81 | E2S carve-out: restore E2S-removed empty-address records whose raw-normalised name equals the S1 name (apply_strategy GAP_CLOSER G1) |
| `L22_ADDTC` | new | +788 | Context-confirmed unique-name empty-address additions: unowned empty-address record, S1 name unique in its country, typo / word-swap name relation, head p in [0.5, 0.7) and mean context-CE logit > 2; one owner per record |
| `L23_ADDHN` | new | +175 | Context-confirmed near-address additions: unowned non-empty-address record with head p > 0.8 and mean context-CE logit (6 sibling-aware context CEs) > 3, one owner per record (highest head p); mostly small house-number offsets the decoder guards leave out |
| `L24_K1C` | new | +893 | Context-confirmed unique-name additions: unowned record for an S1 whose normalised name is unique in its country and that keeps ≥ 1 selected record, head p in (0.7, 0.85] and mean context-CE logit (6 sibling-aware context CEs) in [2, 4); one owner per record (highest head p) |
| `L26_ENSAADD` | new | +540 | Head swap to ENSA_h4_bag3: pairs the ENSA_h4_bag3-head decode adds over the CTXAVG-head decode, after the rider rules (CE-XL veto, two-CE veto, context veto, E2S with G1 carve-out) and one owner per record |
| `L27_DXL` | new | +483 | CE-XL-confirmed alias-name additions: unowned record whose name shares no token with the S1 name (NAME_DISJOINT), ENSA head p in (0.7, 0.85] and CE-XL logit > 0.9; one owner per record (highest head p) |

### CE-NC correction

US / India only, first uploaded in #12.

- **Model.** A LightGBM residual (32 features, 15 leaves, one fixed seed) learned on the training SAMPLE only, on top of the number-corrected Round C probabilities ("NC"), with the scores of a second fine-tuned XLM-R-base cross-encoder ("CE-XL", MIT, 278M parameters, warm-started from the E06 cross-encoder recipe) as extra evidence. Strength 0.5, chosen on TUNE between 0.25 / 0.5.
- **Decoding.** Then the unchanged Round C prefix decoder, global one-owner, the empty-address specialist and #10's US decoy guards, plus the frozen fuzzy empty-address specialist (78 test pairs).
- **Validation** (US / India, 220,730 S1): 0.991481 ordinary / 0.990797 density vs #10 0.991316 / 0.990625 (name-cluster bootstrap 95% CI of the gain +0.00011 / +0.00022).
- France rows are copied from the base. It empties 107 US / India S1 by its own policy.
- **What ships.** The CE-XL weights (1.1 GB) and its test scores are NOT shipped. The residual model, its protocol / fit / freeze records and the verbatim research code are in `resources/final_frozen/audit_cenc/` (audit only; they need the research tree and a GPU for the CE-XL scores). The package applies the resulting frozen add / remove lists, byte-exactly.

### US / India rows

The US / India rows of this file are re-decoded, then spliced in by S1 country.

- **Head.** Both US and India rows come from the **4-backbone head ALLCE_A1A2A5wM1_all_t031**: t031 params (alpha 1.0) + the score columns of four fine-tuned cross-encoders:
  - A1 (CE-XL, XLM-R-base, MIT, + 1 epoch);
  - A2 (`BAAI/bge-reranker-v2-m3`, Apache-2.0, 568M, full pool);
  - A5w (CE-XL 3rd epoch, cell-weighted);
  - M1 (`jhu-clsp/mmBERT-base`, MIT, 308M, full pool).
  - Validation 0.991851 ordinary / 0.991326 density.
- **Decoding.** Every re-decoded head goes through the verbatim CE-NC test path: Round C prefixes, 0.75 anchor + 0.25 evidence utilities, ADD-word veto, global one-owner, empty-address specialist, #10 US legal-add guards recomputed on the new selection, frozen fuzzy rider (alpha 0.5 / threshold 0.85) recomposed on the final selection. Decoder parity with the uploaded CE-NC rows was proven byte for byte first.
- **France rows** are the France rows of upload #8 minus the France layers above (removal-only). Research provenance: `final_frozen/<target>/release_manifest.json`.
- **Splice checks.** Every row is byte-identical to its source file; one owner per record and country consistency are re-checked on the splice.
- **What ships.** The frozen add / remove lists vs #8. The heads and their stored test deltas are in the research tree (not shipped). The fine-tuned cross-encoder weights (MIT / Apache-2.0 backbones, each ≤ 568M parameters) and their stored test scores are not shipped either (1–2.3 GB each; re-creatable with a GPU from the research tree).

### How the layers were chosen

- On test, from label-free pair classes: house-offset mirrors −k / +k, same-address / same-key structure, agreement or disagreement of the packaged base matchers v2b / v3 with the cross-encoder-informed chain, generator keyword pools.
- Priced with label-free internal evaluations.
- **Public-leaderboard feedback was used** to decide which layers to keep (#11, #12 and #13 were read on the board before the later layers were added).
- No test label exists or was used.

### France: no training labels

France appears only in the test set, so there are no French labels. Every France layer is label-free and removal-only; no France pair is ever added.

- **Synthetic French pairs.** Three removal vetoes use cross-encoders fine-tuned on synthetic French pairs generated from the name / address strings of the test France records (no match information of any kind): FR-SYNTH (L10; XLM-R CE-XL warm start; 150k pairs from half of the France S1, mixed with 294k US / India training pairs) and FRCE1 / FRCE2 (L29–L31; mmBERT-base). Positives replay the generator's true-copy edits; negatives replay false-pair patterns the public board had priced as false (keyword swaps, +k house offsets, the same name at another S1).
- **Gate.** FR-SYNTH was accepted after a held-out check (AUC 0.947 between clean anchors and board-false pairs, ≤ 0.5% anchor rejection) and is used only on France because it slightly hurt India (AUC −0.010). Same-street / different-number pairs were excluded because their ±k mirror is symmetric, leaving 283 pairs on 279 S1.
- **The other France layers** rely on label-free structure (house-offset mirrors, same-address word swaps, generator keyword pools), the packaged base matchers, the US / India-trained cross-encoders and context CEs, and the country-blind FR_A matcher (trained on the US / India labels with French canonicalisation views).
- Model weights and test scores are not shipped. The frozen pair lists are in `final_frozen/`; the French cross-encoder code is in `final_frozen/audit_opus995/10_france_ce/`.

### Exact commands

From `code/business_entity_resolution/src/` (polars + stdlib; ~2 GB RAM; about 1–2 min at 4 threads):

```bash
PY=/opt/conda/bin/python3
$PY final_reproduce.py                 # -> output_final/CLEAN13_FRLONE/matching_results.tsv (sha256 + md5 asserted) and output_final/R_US_LEGADD_FR9/ (#10, sha256 asserted)
cmp output_final/CLEAN13_FRLONE/matching_results.tsv ../../../output/matching_results.tsv   # byte-identical
```

What it does:
1. re-hashes the ens8 input tables and the final lists;
2. rebuilds #8 and #10 with `ens8_reproduce.py`;
3. target = #8 − `removed_vs8` + `added_vs8`, with assertions: removed pairs inside #8; added pairs outside #8 and on records no remaining pair owns; additions non-France and same-country; one owner per record; same set as the delta written on top of #10;
4. writes with `ens8_reproduce.write_tsv` (rows in test_source1 order, ids of a row sorted bytewise, comma-joined, tab separated, never quoted).

### Shipped file statistics

1,732,544 rows · 100,000 empty · **5,858,729 matches** · 0 records matched to more than one S1 · every match inside the 83,761,275 candidates.

| country | S1 | matches | per S1 | empty |
|---|---|---|---|---|
| France | 259,452 | 865,012 | 3.334 | 5.86% |
| India | 809,986 | 2,744,679 | 3.389 | 5.75% |
| US | 663,106 | 2,249,038 | 3.392 | 5.76% |

| relative to | removed | added | notes |
|---|---|---|---|
| upload #8 | −19,846 (France 6,561, India 4,016, US 9,269) | +11,643 (India 6,950, US 4,693) | |
| upload #10 | −13,434 on top (France 5,116, India 4,016, US 4,302) | +11,643 | 136 of #10's removals kept |
