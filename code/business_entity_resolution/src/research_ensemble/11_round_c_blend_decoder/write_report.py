"""Write the release explanation from completed verification manifests."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
v = json.loads((OUT / 'validation_verification.json').read_text())
i = json.loads((OUT / 'inference.json').read_text())
c = json.loads((OUT / 'submission_integrity.json').read_text())
p = json.loads((OUT / 'france_proxies.json').read_text())
peers = json.loads((OUT / 'peer_comparisons.json').read_text())
o, d = v['ordinary'], v['density']

def interval(x):
    return f'[{x[0]:+.9f}, {x[1]:+.9f}]'

lines = [
    '# Frozen blend_e0.25: verification and test release', '',
    '**The test files are complete and validated.** The blend improves both local validation scores over the best recorded uploaded approach, VR2 consensus (recorded leaderboard score 0.988842). It is a promising submission candidate, not a verified leaderboard improvement.', '',
    'Against Round A, ordinary validation is effectively tied and density validation improves. Round B has the higher ordinary score, while this blend has the higher density score. These results do not establish that the blend is best under every distribution.', '',
    '## Submission files', '',
    '- [matching_results.tsv](output/matching_results.tsv): the full-test file to upload for scoring.',
    '- [candidate_pairs.tsv](output/candidate_pairs.tsv): the unchanged full-test candidate universe.',
    '- [blend_e0.25_outputs.zip](blend_e0.25_outputs.zip): both output TSVs and verification records.',
    '- [SHA256SUMS](SHA256SUMS): file checksums.', '',
    f'There are **{i["queries"]:,} test queries**, **{i["pairs"]:,} selected matches**, and **{c["candidate_pairs"]:,} candidate pairs**. This ZIP is an output-file bundle; it does not replace the competition’s final source-code package. No leaderboard upload was performed.', '',
    '## Like-for-like validation', '',
    'Scores are per-query macro F0.5 on a 0–1 scale, with the official empty-query convention. Within each column every approach uses the identical query set. Ordinary and density columns use different query sets and should not be subtracted to measure density cost.', '',
    '| Approach | Ordinary: 220,730 queries | Density: 178,891 queries |',
    '|---|---:|---:|',
]
for name, label in [('v3anchor', 'v3anchor'), ('vr2_consensus', 'VR2 consensus, best recorded upload'), ('round_a', 'Round A, number-correction consensus'), ('round_b', 'Round B, evidence decoder')]:
    lines.append(f'| {label} | {o["comparisons"][name]["baseline"]:.9f} | {d["comparisons"][name]["baseline"]:.9f} |')
lines += [f'| **Requested blend_e0.25** | **{o["independent_tsv_score"]["macro_f05"]:.9f}** | **{d["independent_tsv_score"]["macro_f05"]:.9f}** |', '',
          '| Comparator | Ordinary delta [paired 95% interval] | Density delta [paired 95% interval] |', '|---|---|---|']
for name in ('vr2_consensus', 'round_a', 'round_b'):
    a, b = o['comparisons'][name], d['comparisons'][name]
    lines.append(f'| {name} | {a["delta"]:+.9f} {interval(a["query_ci95"])} | {b["delta"]:+.9f} {interval(b["query_ci95"])} |')
lines += ['', 'Paired intervals use 2,000 query bootstrap replicates; country/name-cluster intervals are also saved in validation_verification.json. The aggregate gains over VR2 are positive under both methods. They are small absolute gains, not percentage-point gains of the same numeric size.', '',
          '| Subset, vs VR2 consensus | Ordinary delta [95% interval] | Density delta [95% interval] |', '|---|---|---|']
for subset in ('report', 'locked', 'US', 'India'):
    a, b = o['comparisons']['vr2_consensus']['slices'][subset], d['comparisons']['vr2_consensus']['slices'][subset]
    lines.append(f'| {subset} | {a["delta"]:+.9f} {interval(a["ci95"])} | {b["delta"]:+.9f} {interval(b["ci95"])} |')
lines += ['', 'REPORT and locked intervals include zero. Density TUNE was used to select this blend; density reuses validation identities, and historical REPORT/locked labels have prior project exposure. Bootstrap intervals do not correct for repeated experimentation. There are no labeled France queries. A public or private leaderboard gain cannot be inferred with certainty.', '',
          'The aggregate gain also includes individual regressions:', '']
for tag in ('ordinary', 'density'):
    a = v[tag]['comparisons']['vr2_consensus']
    lines.append(f'- {tag}: {a["improved"]:,} queries improve, {a["harmed"]:,} worsen, including {a["previously_perfect_harmed"]:,} previously perfect queries.')
lines += ['', 'The strongest completed ordinary-validation run in each of the three ensemble directories was also independently rescored from saved selections. These additional comparisons are diagnostic; they do not change the frozen submission policy.', '',
          '| Ensemble directory | Best completed ordinary run | Rescored ordinary F0.5 | Blend delta [95% interval] |', '|---|---|---:|---|']
for folder, record in peers.items():
    a = record['comparison']
    lines.append(f'| {folder} | {record["selected_name"]} | {a["baseline"]:.9f} | {a["delta"]:+.9f} {interval(a["query_ci95"])} |')
lines += ['', '## What the approach does', '',
          '1. Start with the existing v3anchor ensemble’s candidate-pair probabilities and the existing candidate pool.',
          '2. For pairs inside the existing uncertainty gate, run the address-number correction models with seeds 11, 29, and 47. They inspect evidence such as digit edits, transpositions, leading zeros, unit/house-number placement, remaining address text, sibling matches, and component disagreement. Average their raw corrections, add that average to the anchor logit, then convert back to a probability. Scores outside the gate are unchanged.',
          '3. For each query, retain eligible candidates whose corrected probability is above 0.02 and at least the stored competing-query probability. Sort them by corrected probability, and consider selecting none, the top one, the top two, and so on, up to the top ten.',
          '4. The original decoder and Round B evidence decoder each estimate the F0.5 utility of each possible prefix. The evidence decoder additionally sees last/next-candidate evidence and query-level missingness, number-conflict, and disagreement summaries. Choose the prefix maximizing `0.75 × original utility + 0.25 × evidence utility`; ties prefer fewer matches.',
          '5. Remove existing ADD-pattern decoys, then resolve ownership across the entire test set. If multiple queries claim the same S2/S3 record, keep the query with the highest corrected probability; ties prefer the lower S1 index.', '',
          'The 25% weight blends predicted set utilities, not pair probabilities or final answer sets. Both decoders see the same mean-corrected probabilities. This selected policy uses no whole-query seed-consensus or empty-status fallback, and none of Round C’s four query-balanced decoder models. The Round B evidence decoder was trained on SAMPLE prefix targets using fresh cross-fitted number corrections and complete truth counts; validation labels did not fit its trees.', '',
          '## Full-test behavior and France diagnostics', '',
          f'The blend changes **{i["changed_queries_vs_vr2"]:,}** query answers versus the uploaded VR2 consensus. Changed-query details are saved in data/changed_vs_vr2.parquet.', '',
          '| Country | Test queries | Matches/query | Empty queries | Changed vs VR2 |', '|---|---:|---:|---:|---:|']
for row in i['country_profile']:
    name = row['country']
    lines.append(f'| {name} | {row["queries"]:,} | {row["matches_per_query"]:.5f} | {row["empty_fraction"]:.3%} | {i["changed_queries_by_country"].get(name, 0):,} |')
lines += ['', 'France diagnostics use heuristic proxy sets, not ground-truth accuracy. They do not alter the frozen policy.', '',
          '| France proxy | VR2 consensus | blend_e0.25 |', '|---|---:|---:|']
for key, label in [('strict_anchor_accept', 'Strict-anchor acceptance'), ('op_word_drop_accept', 'Word-drop acceptance'), ('copy_filler_accept', 'Copy-filler acceptance'), ('op_filler_add_accept', 'Added-filler acceptance'), ('op_amp_accept', 'Ampersand-operation acceptance'), ('op_country_tag_accept', 'Country-tag-operation acceptance'), ('decoy_accept', 'Mined decoys accepted')]:
    vals = []
    for name in ('vr2_consensus', 'blend_e0.25'):
        value = p[name][key]
        vals.append(str(value[2]) if key == 'decoy_accept' else f'{(value[1] if isinstance(value, list) else value):.3%}')
    lines.append(f'| {label} | {vals[0]} | {vals[1]} |')
lines += ['', '## Verification and reproduction', '',
          '- Fresh predictions from all three correction models reproduced the saved ordinary and density corrections exactly.',
          '- The same chunked inference function used for test reproduced every selected validation pair in both environments. The exported validation TSVs were independently scored with common/score.py.',
          '- Uncorrected full-test inference reproduced the shipped v3anchor TSV byte-for-byte. The blend preserves the existing competition table and resolves ownership globally after processing whole-query chunks.',
          '- The official validator passed both output TSVs with ID checks. Independent streaming checks verified all candidate IDs, all query rows, no duplicate pairs, match containment, and global ownership; a separate check verified country consistency.',
          f'- SHA-256 checks confirmed all {c["protected_prior_files"]} protected earlier files and all {c["protected_round_c_files"]} frozen Round C artifacts were unchanged.', '',
          'Saved base-model probabilities, competition scores, and the verified 124-feature test band were reused. The number features and three number-correction predictions were computed for this release. No retraining, threshold search, or blend reselection was performed.', '',
          'From the workspace root:', '', '```bash',
          'bash matching/v2shash/round_c/test_release/reproduce.sh', '```', '',
          'Evidence: [validation_verification.json](validation_verification.json), [inference.json](inference.json), [submission_integrity.json](submission_integrity.json), [france_proxies.json](france_proxies.json), and [official validator log](logs/official_validator.log).',
          'The original Round C report describes the earlier validation-only run; this separate report documents the later full-test release.', '']
(OUT / 'REPORT.md').write_text('\n'.join(lines))
print(OUT / 'REPORT.md')
