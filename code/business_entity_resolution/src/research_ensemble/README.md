# research_ensemble/ -- verbatim AUDIT COPIES of the research code behind the submitted file

The submitted `output/matching_results.tsv` (upload #10, `R_US_LEGADD_FR9`) is upload #8 (the v3anchor ensemble chain, `nc_specialist_legal_fr`)
minus a frozen list of label-free decoy removals.  Its lineage runs through about 15 research directories.  This directory holds a **verbatim copy**
(byte-identical, `cmp`-checked) of the research scripts of every lineage step that is not covered by the packaged matcher (`ber/`, `run_all.sh`),
so that reviewers can read exactly what was run.  **These copies are NOT wired into the package and are not runnable from it**:

* they keep their absolute research-tree paths (`/workspace/saumilya/amazon-ml/work/...`) and `sys.path` imports across research directories;
* they need the research tree (about 150 GB of intermediates: union / feature parts, the v2b / v3 / v1 prediction tables, `members_test.parquet`,
  `test_v3anchor.parquet`, `test_comp.parquet`, `test_anchor_band.parquet`, the internal_eval pair-class tables), a GPU for the E06 cross-encoder
  (training + ~9 GPU-min of scoring; the tokenised inputs were deleted, the 3 x 1.1 GB XLM-R weights are not shipped) and 25-100 GB RAM peaks;
* nothing in here is imported by `ens8_reproduce.py`, `predict.py` or any other package entry point.

What IS reproducible inside the package: `src/ens8_reproduce.py` rebuilds every target TSV (#8, A, B, R, R_FR9) byte-exactly from the frozen
decision-layer tables in `src/resources/ens8_frozen/` (see README.md section 8 of the package).  Matcher v3 (upload #5) is fully reproducible
with `run_all.sh`.  The steps 00 (transliteration), 01 (blocking union v2), 03-05 (features, v2b, v3) of the lineage are the packaged code itself
(`prepare.py`, `block.py`, `features.py`, `train.py`, `predict.py`, `configs/v2b.yaml`, `configs/v3.yaml`, `resources/models/{v1,v2b,v3}`).

Models and stored intermediate tables of these steps: `src/resources/ens8_frozen/models/` and `audit/` (sha256 + provenance in its `manifest.json`).

| step dir | lineage step | file in this directory | original path (research tree, relative to `work/`) |
|---|---|---|---|
| `02_v1_competitor_table` | v1 competitor table -> `test_comp.parquet` (p_other / src for all 83,761,275 pairs); consumed by the NC decode and the specialist | `a3_comp.py` | `matching/vr2_stress_submission/src/a3_comp.py` |
| `02_v1_competitor_table` |  | `a3_main.py` | `matching/vr2_stress_submission/src/a3_main.py` |
| `06_E06_cross_encoder_stage3` | E06: XLM-RoBERTa-base cross-encoder on the uncertain band (GPU) + LightGBM stage 3 -> member `p3_test_E06_p3b.parquet` (m_e06). `cross_encoder_warm_start/`: the earlier xlm-roberta-base cross-encoder `03_ce_fold.py --init` starts from (`common_text.py` of that directory is NOT copied: it imports the GPL `unidecode`; it is not used by the E06 scripts) | `01_band.py` | `matching/prod_v2c/exp/E06_ce_band/01_band.py` |
| `06_E06_cross_encoder_stage3` |  | `02_tok.py` | `matching/prod_v2c/exp/E06_ce_band/02_tok.py` |
| `06_E06_cross_encoder_stage3` |  | `03_ce_fold.py` | `matching/prod_v2c/exp/E06_ce_band/03_ce_fold.py` |
| `06_E06_cross_encoder_stage3` |  | `04_stage3.py` | `matching/prod_v2c/exp/E06_ce_band/04_stage3.py` |
| `06_E06_cross_encoder_stage3` |  | `09_france_prep.py` | `matching/prod_v2c/exp/E06_ce_band/09_france_prep.py` |
| `06_E06_cross_encoder_stage3` |  | `12_test_prep.py` | `matching/prod_v2c/exp/E06_ce_band/12_test_prep.py` |
| `06_E06_cross_encoder_stage3` |  | `10_score_gpu.py` | `matching/prod_v2c/exp/E06_ce_band/10_score_gpu.py` |
| `06_E06_cross_encoder_stage3` |  | `13_test_apply.py` | `matching/prod_v2c/exp/E06_ce_band/13_test_apply.py` |
| `06_E06_cross_encoder_stage3` |  | `cross_encoder_warm_start/train_ce.py` | `matching/cross_encoder/train_ce.py` |
| `06_E06_cross_encoder_stage3` |  | `cross_encoder_warm_start/prep_tokens.py` | `matching/cross_encoder/prep_tokens.py` |
| `06_E06_cross_encoder_stage3` |  | `cross_encoder_warm_start/prep_text.py` | `matching/cross_encoder/prep_text.py` |
| `06_E06_cross_encoder_stage3` |  | `cross_encoder_warm_start/run_xlmr.sh` | `matching/cross_encoder/run_xlmr.sh` |
| `07_E13_graph_specialist` | E13: graph / sibling specialist on the v2b band -> member `p3_all_variants_test.parquet` column `p_cv2` (m_e13) | `fit_cv2.py` | `matching/prod_v2c/exp/E13_codex_refit/fit_cv2.py` |
| `07_E13_graph_specialist` |  | `e13_block.py` | `matching/prod_v2c/exp/E13_codex_refit/e13_block.py` |
| `07_E13_graph_specialist` |  | `splits.py` | `matching/prod_v2c/exp/E13_codex_refit/splits.py` |
| `07_E13_graph_specialist` |  | `sibling_features.py` | `matching/accuracy_lab_20260925/sibling_features.py` |
| `07_E13_graph_specialist` |  | `prepare.py` | `matching/accuracy_lab_20260925/prepare.py` |
| `07_E13_graph_specialist` |  | `experiment.py` | `matching/accuracy_lab_20260925/experiment.py` |
| `07_E13_graph_specialist` |  | `interim_v2b_E13/b1common.py` | `matching/prod_v2c/build/interim_v2b_E13/src/b1common.py` |
| `07_E13_graph_specialist` |  | `interim_v2b_E13/01_band.py` | `matching/prod_v2c/build/interim_v2b_E13/src/01_band.py` |
| `07_E13_graph_specialist` |  | `interim_v2b_E13/02_score.py` | `matching/prod_v2c/build/interim_v2b_E13/src/02_score.py` |
| `08_v3anchor_assembly_decoder` | v3anchor = logit(v3) + [logit(E06) - logit(v2b)] + [logit(E13) - logit(v2b)]; R10c decoder refit on the sample OOF; decoy veto + one-owner -> upload #6 and `test_v3anchor.parquet` (column p). WARNING: `01_assemble.py` reads `build/p3_test_E06.parquet`, which was later overwritten with the p3bg variant; the member actually consumed equals `build/p3_test_E06_p3b.parquet` (re-runs must point there) | `01_assemble.py` | `matching/ensemble_v1/src/01_assemble.py` |
| `08_v3anchor_assembly_decoder` |  | `BLEND_run.py` | `matching/ensemble_v1/src/BLEND_run.py` |
| `08_v3anchor_assembly_decoder` |  | `ens_common.py` | `matching/ensemble_v1/common/ens_common.py` |
| `08_v3anchor_assembly_decoder` |  | `build_testfile.sh` | `matching/ensemble_v1/build/build_testfile.sh` |
| `08_v3anchor_assembly_decoder` |  | `build_summary.py` | `matching/ensemble_v1/build/src/build_summary.py` |
| `08_v3anchor_assembly_decoder` |  | `eval_pairs.py` | `matching/prod_v2c/common/eval_pairs.py` |
| `08_v3anchor_assembly_decoder` |  | `decide_lib.py` | `research/postproc/decide_lib.py` |
| `08_v3anchor_assembly_decoder` |  | `interim_v2b_E13/03_decide.py` | `matching/prod_v2c/build/interim_v2b_E13/src/03_decide.py` |
| `08_v3anchor_assembly_decoder` |  | `interim_v2b_E13/pp_common.py` | `matching/prod_v2c/build/interim_v2b_E13/src/pp_common.py` |
| `08_v3anchor_assembly_decoder` |  | `interim_v2b_E13/15_check_submission.py` | `matching/prod_v2c/build/interim_v2b_E13/src/15_check_submission.py` |
| `08_v3anchor_assembly_decoder` |  | `interim_v2b_E13/04_gate.py` | `matching/prod_v2c/build/interim_v2b_E13/src/04_gate.py` |
| `09_vr2_test_infrastructure` | vr2 test infrastructure inherited by the chain: the test band table `test_anchor_band.parquet` (`prep_test.py`, needs the 23 GB prod_v3 feature parts), the residual / evaluate / study modules imported by Round C (`evaluate.py::modified`, `study.py::readsel/D`). The vr2 residual seeds themselves are NOT in the #8 lineage | `prep_test.py` | `matching/vr2_stress_submission/src/prep_test.py` |
| `09_vr2_test_infrastructure` |  | `evaluate.py` | `matching/variance_research_round2_20260926/evaluate.py` |
| `09_vr2_test_infrastructure` |  | `residual.py` | `matching/variance_research_round2_20260926/residual.py` |
| `09_vr2_test_infrastructure` |  | `seed_consensus.py` | `matching/variance_research_round2_20260926/seed_consensus.py` |
| `09_vr2_test_infrastructure` |  | `features.json` | `matching/variance_research_round2_20260926/features.json` |
| `09_vr2_test_infrastructure` |  | `study.py` | `matching/variance_study_20260926/study.py` |
| `10_v2shash_number_residuals` | v2shash Round A "numbers" arm: address-number parsing / pair features and the three residual seeds `numbers_{11,29,47}.txt` (models in `resources/ens8_frozen/models/`) | `train.py` | `matching/v2shash/train.py` |
| `10_v2shash_number_residuals` |  | `experiment.py` | `matching/v2shash/experiment.py` |
| `10_v2shash_number_residuals` |  | `numbers_features.json` | `matching/v2shash/numbers_features.json` |
| `11_round_c_blend_decoder` | Round C blend_e0.25 = "NC": mean residual delta inside the gate, prefix chosen by 0.75 x R10c(v3anchor) utility + 0.25 x evidence-decoder utility, ADD-decoy veto (`v3_postpass/pp.py::add_pairs`), global one-owner -> `test_selected.parquet` (shipped) and the NC TSV (sha256 1900e2be...). `release.py::write` is the TSV writer that `ens8_reproduce.py` replicates | `release.py` | `matching/v2shash/round_c/test_release/release.py` |
| `11_round_c_blend_decoder` |  | `reproduce.sh` | `matching/v2shash/round_c/test_release/reproduce.sh` |
| `11_round_c_blend_decoder` |  | `write_report.py` | `matching/v2shash/round_c/test_release/write_report.py` |
| `11_round_c_blend_decoder` |  | `run.py` | `matching/v2shash/round_c/run.py` |
| `11_round_c_blend_decoder` |  | `oof_decoder.py` | `matching/v2shash/oof_decoder.py` |
| `11_round_c_blend_decoder` |  | `set_decode.py` | `matching/accuracy_lab_20260925/set_decode.py` |
| `11_round_c_blend_decoder` |  | `score.py` | `common/score.py` |
| `11_round_c_blend_decoder` |  | `v3_postpass/pp.py` | `matching/v3_postpass/src/pp.py` |
| `11_round_c_blend_decoder` |  | `v3_postpass/rules.py` | `matching/v3_postpass/src/rules.py` |
| `11_round_c_blend_decoder` |  | `v3_postpass/vx/common.py` | `matching/v3_postpass/src/vx/common.py` |
| `12_empty_address_specialist` | US/India empty-address specialist: LightGBM residual on the anchor band (`specialist.py fit|test`, seed 61), applied with the frozen policy alpha 1.0 / threshold 0.80 (`apply_test.py`) -> `test_nc_robust_additions.parquet` (shipped). `archetype.py` builds the record views caches the specialist features read | `specialist.py` | `matching/iterate_20260926/empty_address/specialist.py` |
| `12_empty_address_specialist` |  | `apply_test.py` | `matching/iterate_20260926/empty_address/apply_test.py` |
| `12_empty_address_specialist` |  | `prepare.py` | `matching/iterate_20260926/empty_address/prepare.py` |
| `12_empty_address_specialist` |  | `nc_tune_grid.py` | `matching/iterate_20260926/empty_address/nc_tune_grid.py` |
| `12_empty_address_specialist` |  | `eval_nc.py` | `matching/iterate_20260926/empty_address/eval_nc.py` |
| `12_empty_address_specialist` |  | `archetype.py` | `internal_eval/agents/B_archetypes/src/archetype.py` |
| `13_france_cleanup_mask` | France legal-form / country-tag cleanup: `audit.py` writes `mask_ppLegal_FR.parquet` (families CTAG, LEG3, LEG12 at +k) from the pair-class table `pc_test_fam.parquet` built by `03_pairclass.py` (+ `ie_common.py`) | `audit.py` | `matching/iterate_20260926/france/src/audit.py` |
| `13_france_cleanup_mask` |  | `package.py` | `matching/iterate_20260926/france/src/package.py` |
| `13_france_cleanup_mask` |  | `val_audit.py` | `matching/iterate_20260926/france/src/val_audit.py` |
| `13_france_cleanup_mask` |  | `sensitivity.py` | `matching/iterate_20260926/france/src/sensitivity.py` |
| `13_france_cleanup_mask` |  | `03_pairclass.py` | `internal_eval/src/03_pairclass.py` |
| `13_france_cleanup_mask` |  | `ie_common.py` | `internal_eval/src/ie_common.py` |
| `15_build_upload8` | Upload #8 assembly: NC selection + specialist additions - France mask -> `nc_specialist_legal_fr/matching_results.tsv` (sha256 7d04f128...); `validate_outputs.py` = official validator + containment / ownership / country audit | `build_candidate.py` | `matching/iterate_20260926/src/build_candidate.py` |
| `15_build_upload8` |  | `validate_outputs.py` | `matching/iterate_20260926/src/validate_outputs.py` |
| `16_removal_descendants` | Removal-only descendants of #8: A_US_LEG (`audit.py::original_rule`), R extension and FR9 guard (`prepare_release.py`), stdlib re-application (`reproduce_candidate.py`); `S1_mirror_sweep/` builds the +-k offset pair families (mirror control) behind the US extension; `01_hybrids7.py` is the source of the France anchored SWAP_GEN guard (FR9) | `audit.py` | `winning_strategy_20260926/execution_20260926/src/audit.py` |
| `16_removal_descendants` |  | `prepare_release.py` | `winning_strategy_20260926/execution_20260926/src/prepare_release.py` |
| `16_removal_descendants` |  | `reproduce_candidate.py` | `winning_strategy_20260926/execution_20260926/src/reproduce_candidate.py` |
| `16_removal_descendants` |  | `check_integrity.py` | `winning_strategy_20260926/execution_20260926/src/check_integrity.py` |
| `16_removal_descendants` |  | `score_official.py` | `winning_strategy_20260926/execution_20260926/src/score_official.py` |
| `16_removal_descendants` |  | `01_hybrids7.py` | `matching/ensemble_v7/src/01_hybrids7.py` |
| `16_removal_descendants` |  | `S1_mirror_sweep/s01_pairs.py` | `winning_strategy_20260926/agents/S1_mirror_sweep/src/s01_pairs.py` |
| `16_removal_descendants` |  | `S1_mirror_sweep/s02_mirror.py` | `winning_strategy_20260926/agents/S1_mirror_sweep/src/s02_mirror.py` |
| `16_removal_descendants` |  | `S1_mirror_sweep/s03_value.py` | `winning_strategy_20260926/agents/S1_mirror_sweep/src/s03_value.py` |

76 files, all byte-identical to their originals (checked with `filecmp.cmp(shallow=False)` when this README was generated).

## Notes for reviewers

* Lineage steps 00-05 (dataset + transliteration, blocking union v2, feature libraries, v2b, v3) are the packaged pipeline; the members v2b / v3
  are its `predict.py --variant v2b|v3 --score-only` probabilities (re-scoring with the shipped LightGBM models reproduces p1 bit for bit and p2 up
  to 166 tie-order differences in 2.9M checked pairs, see BUILD_NOTES; the chain was therefore frozen on the STORED probabilities, not re-scored).
* Step 06 (E06) is the only lineage step that ran on a GPU besides blocking / transliteration; its stored test logits are shipped in
  `resources/ens8_frozen/audit/ce_xw_*.parquet` (33 MB), so `13_test_apply.py` could rebuild the member on CPU from them + the stage-3 models.
* Step 08: `01_assemble.py` reads `build/p3_test_E06.parquet`; that file was overwritten with the `p3bg` variant AFTER `members_test.parquet` was
  built; the member consumed is byte-identical to `build/p3_test_E06_p3b.parquet`.  Any re-run must point at `_p3b` explicitly.
* Step 11: `release.py infer` asserts `validation_verification.json` (heavy verify step), the model sha256 in `preparation.json`, the frozen
  `robust_winner`, and `package()` re-hashes ~400 protected research files; it is brittle to re-run outside the frozen research tree.
* Steps 12 / 13 / 16 depend on label-free, test-derived tables (`internal_eval/data/pc_test_fam.parquet`, `pc_test_pl.parquet` pseudo-label SWAP_GEN
  chain, `pc_test_frstrict.parquet`, `S1_mirror_sweep/out/pairs_test.parquet`); the RESULTS of those rules are shipped as the frozen mask and removal
  lists, the rules themselves are documented in the root README.md section 4 (ensemble stages) and section 5 (transductive uses).
* `06_E06_cross_encoder_stage3/cross_encoder_warm_start/common_text.py` is deliberately absent (it imports the GPL library `unidecode`, which the
  package must not ship; the E06 scripts do not import it).
* Credentials / upload helpers of the research tree are not copied (checked with grep before zipping).
