"""Frozen blend_e0.25 test release. Existing experiment artifacts are read-only."""
import os
for key in ('POLARS_MAX_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
import sys, json, time, gc, hashlib, argparse, resource
from pathlib import Path
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
RC_DIR = OUT.parent
sys.path.insert(0, str(RC_DIR))
import run as RC
import train as T
import numpy as np
import polars as pl
import lightgbm as lgb
O, R, S = RC.O, RC.R, RC.S
W, M, K = RC.M.parent, RC.M, RC.K
VR = M / 'vr2_release_20260926'
for folder in ('data', 'logs', 'output'):
    (OUT / folder).mkdir(exist_ok=True)


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def model_paths():
    return [RC.HERE / f'models/numbers_{s}.txt' for s in RC.SEEDS] + [
        M / 'ensemble_v1/results/eval_ENS_v3anchor_R10c_m0.txt',
        O.D / 'models/decoder_evidence.txt']


def prepare():
    start = time.time()
    B = pl.read_parquet(VR / 'data/test_anchor_band.parquet')
    base_features = json.loads((RC.OLD / 'features.json').read_text())
    features = json.loads((RC.HERE / 'numbers_features.json').read_text())
    assert B.columns[4:] == base_features and features == base_features + T.NF
    assert B.height == 3325555 and B.select(pl.struct(K).n_unique()).item() == B.height
    nf_path = OUT / 'data/test_number_features.parquet'
    if nf_path.exists():
        nf = pl.read_parquet(nf_path)
        assert nf.select(K).equals(B.select(K))
    else:
        records = []
        for kind, col in [('s1', 's1_idx'), ('s23', 'cand_idx')]:
            need = B.select(pl.col(col).cast(pl.UInt32).alias('erow')).unique()
            rows = (pl.scan_parquet(M / f'prod_v1/data/rec_test_{kind}.parquet')
                    .select('erow', 'addr_raw').join(need.lazy(), on='erow', how='semi').collect())
            records.append({int(i): T.parse(a) for i, a in rows.iter_rows()})
            log('Parsed test addresses', kind, len(records[-1]))
        matrix = np.empty((B.height, len(T.NF)), dtype=np.float32)
        for j, (s, c) in enumerate(B.select(K).iter_rows()):
            matrix[j] = T.feature_pair(records[0][s], records[1][c])
            if j and j % 250000 == 0:
                log('Number features', j, '/', B.height)
        nf = B.select(K).hstack(pl.DataFrame(matrix, schema=T.NF))
        nf.write_parquet(nf_path)
        del records, matrix
        gc.collect()
    B = B.join(nf, on=K, how='left', maintain_order='left')
    assert B.select(T.NF).null_count().sum_horizontal().item() == 0
    del nf
    x = np.nan_to_num(B.select(features).to_numpy().astype(np.float32), nan=-1, posinf=1e6, neginf=-1e6)
    corr = B.select(K + ['missing_gate', 'p_anchor'])
    provenance = {}
    for seed, path in zip(RC.SEEDS, model_paths()[:3]):
        model = lgb.Booster(model_file=str(path))
        assert model.feature_name() == features
        corr = corr.with_columns(pl.Series(f'delta_{seed}', model.predict(x, raw_score=True, num_threads=4).astype(np.float32)))
        provenance[str(path)] = digest(path)
        log('Test correction predicted', seed, model.current_iteration())
    assert np.isfinite(corr.select([f'delta_{s}' for s in RC.SEEDS]).to_numpy()).all()
    corr.write_parquet(OUT / 'data/test_corrections.parquet')
    B.select(K + O.EV).write_parquet(OUT / 'data/test_evidence.parquet')
    save('preparation.json', dict(rows=B.height, features=features, model_sha256=provenance,
                                 source_band=str(VR / 'data/test_anchor_band.parquet'),
                                 seconds=time.time()-start))
    log('PREPARATION COMPLETE', round(time.time()-start), 'seconds')


def decoders():
    original, evidence = [lgb.Booster(model_file=str(p)) for p in model_paths()[3:]]
    # The archived original was fitted from an unnamed numpy matrix, in O.BASE order.
    assert original.num_feature() == len(O.BASE)
    assert original.feature_name() in (O.BASE, [f'Column_{i}' for i in range(len(O.BASE))])
    assert evidence.feature_name() == O.BASE + O.EXTRA
    return original, evidence


def own(selected):
    return (selected.sort(['cand_idx', 'p', 's1_idx'], descending=[False, True, False])
            .unique('cand_idx', keep='first', maintain_order=True).select(K))


def decode_chunks(full, evidence, corr, veto, tag, reproduce_anchor=False):
    """Whole-query chunks; ownership is resolved globally only after concatenation."""
    original, extra = decoders()
    blend_parts, anchor_parts = [], []
    chunk_ids = sorted((full['s1_idx'] // 100000).unique().to_list())
    for n, chunk in enumerate(chunk_ids):
        lo, hi = chunk * 100000, (chunk + 1) * 100000 - 1
        part = full.filter(pl.col('s1_idx').is_between(lo, hi))
        ev = evidence.filter(pl.col('s1_idx').is_between(lo, hi))
        cr = corr.filter(pl.col('s1_idx').is_between(lo, hi))
        if reproduce_anchor:
            a = R.decide_fast(part, original).join(veto, on=K, how='anti')
            anchor_parts.append(a.join(part.select(K + ['p']), on=K))
        corrected = R.modified(part, cr, 1., 'all')
        t, w = O.prefixes(corrected, ev)
        ua = original.predict(t.select(O.BASE).to_numpy().astype(np.float32), num_threads=4)
        ue = extra.predict(t.select(O.BASE + O.EXTRA).to_numpy().astype(np.float32), num_threads=4)
        selected = RC.pick(t, w, .75 * ua + .25 * ue).join(veto, on=K, how='anti')
        blend_parts.append(selected.join(corrected.select(K + ['p']), on=K))
        log('Decoded', tag, n+1, '/', len(chunk_ids), 'pairs', part.height, 'selected', selected.height)
        del part, corrected, t, w, ua, ue
        gc.collect()
    raw = pl.concat(blend_parts)
    result = own(raw)
    log('Global ownership', tag, 'removed', raw.height-result.height)
    return result, own(pl.concat(anchor_parts)) if reproduce_anchor else None


def compare_rows(rows, base):
    assert rows['s1_idx'].equals(base['s1_idx'])
    d = rows['f'].to_numpy() - base['f'].to_numpy()
    result = dict(score=float(rows['f'].mean()), baseline=float(base['f'].mean()),
                  delta=float(d.mean()), query_ci95=S.bootstrap(d),
                  name_cluster_ci95=S.bootstrap(d, rows['name_group'].to_numpy()),
                  improved=int((d > 1e-12).sum()), harmed=int((d < -1e-12).sum()),
                  previously_perfect_harmed=int(((base['f'].to_numpy() == 1) & (d < -1e-12)).sum()), slices={})
    for col, values in [('eval_split', ['tune', 'report', 'locked']), ('country', ['US', 'India'])]:
        for value in values:
            mask = (rows[col] == value).to_numpy()
            result['slices'][value] = dict(n=int(mask.sum()), delta=float(d[mask].mean()), ci95=S.bootstrap(d[mask]))
    return result


def verify():
    """Fresh correction predictions, fresh decoding, and independent TSV scoring."""
    start = time.time()
    from score import load_id_lists, macro_f05
    truth = load_id_lists(W / 'splits/val_ground_truth.tsv')
    output = {}
    for density in (False, True):
        tag = 'density' if density else 'ordinary'
        env = RC.load_environment(density)
        full, _, meta, g, veto, incumbent, _, _ = env
        b, features = T.data('numbers', density)
        b = b.join(meta.select('s1_idx'), on='s1_idx', how='semi')
        cached = RC.correction('numbers', density).join(b.select(K), on=K, how='semi')
        assert cached.height == b.height
        x = np.nan_to_num(b.select(features).to_numpy().astype(np.float32), nan=-1, posinf=1e6, neginf=-1e6)
        corr = b.select(K + ['missing_gate'])
        for seed, path in zip(RC.SEEDS, model_paths()[:3]):
            model = lgb.Booster(model_file=str(path))
            assert model.feature_name() == features
            corr = corr.with_columns(pl.Series(f'delta_{seed}', model.predict(x, raw_score=True, num_threads=4).astype(np.float32)))
            log('Validation correction predicted', tag, seed)
        expected = cached.sort(K).select([f'delta_{s}' for s in RC.SEEDS])
        assert corr.sort(K).select(expected.columns).equals(expected), 'Correction prediction parity failed'
        del x
        selected, _ = decode_chunks(full, b.select(K + O.EV), corr, veto, tag)
        wanted = pl.read_parquet(RC_DIR / f'data/{tag}_blend_e0.25_selected.parquet')
        assert selected.sort(K).equals(wanted.sort(K)), 'Frozen blend decision parity failed'
        rows = S.score(selected, meta, g)
        pred = load_id_lists(RC_DIR / f'{tag}_blend_e0.25_matching_results.tsv')
        expected_ids = set(meta['entity_id'])
        assert set(pred) == expected_ids
        score = macro_f05(pred, truth, ids=sorted(expected_ids), verbose=False)
        assert abs(score['macro_f05'] - rows['f'].mean()) < 1e-10
        output[tag] = dict(queries=meta.height, correction_prediction_parity=True,
                           fresh_chunked_decision_parity=True, independent_tsv_score=score,
                           comparisons={})
        controls = dict(vr2_consensus=incumbent, v3anchor=env[1],
                        round_a=pl.read_parquet(RC.source('round_a', tag)),
                        round_b=pl.read_parquet(RC.source('round_b', tag)))
        for name, sel in controls.items():
            output[tag]['comparisons'][name] = compare_rows(rows, S.score(sel, meta, g))
        save('validation_verification.json', output)
        log('VALIDATION VERIFIED', tag, score['macro_f05'])
        del env, full, b, corr, cached, selected, rows, controls
        gc.collect()
    output['seconds'] = time.time()-start
    save('validation_verification.json', output)


def infer():
    start = time.time()
    verification = json.loads((OUT / 'validation_verification.json').read_text())
    for tag in ('ordinary', 'density'):
        assert verification[tag]['correction_prediction_parity'] and verification[tag]['fresh_chunked_decision_parity']
    assert json.loads((RC_DIR / 'frozen_selection.json').read_text())['robust_winner']['name'] == 'blend_e0.25'
    for path, expected in json.loads((OUT / 'preparation.json').read_text())['model_sha256'].items():
        assert digest(path) == expected
    comp = pl.read_parquet(VR / 'data/test_comp.parquet', columns=K + ['p_other', 'src'])
    anchor = pl.read_parquet(M / 'ensemble_v1/data/test_v3anchor.parquet', columns=K + ['p'])
    assert comp.height == anchor.height == 83761275
    if comp.select(K).equals(anchor.select(K)):
        full = comp.with_columns(anchor['p'].cast(pl.Float32))
    else:
        full = comp.join(anchor, on=K, how='left', maintain_order='left')
    assert full.height == 83761275 and full['p'].null_count() == 0
    del comp, anchor
    evidence = pl.read_parquet(OUT / 'data/test_evidence.parquet')
    corr = pl.read_parquet(OUT / 'data/test_corrections.parquet')
    joined = corr.join(full.select(K + ['p']), on=K)
    assert joined.height == corr.height and joined['p'].equals(joined['p_anchor'])
    del joined
    veto = S.pp.add_pairs('test').filter(pl.col('k').is_in(S.D)).select(K)
    selected, reproduced = decode_chunks(full, evidence, corr, veto, 'test', reproduce_anchor=True)
    selected.write_parquet(OUT / 'data/test_selected.parquet')
    shipped = S.readsel(M / 'ensemble_v1/build/v3anchor/data/sel_final.parquet')
    assert reproduced.sort(K).equals(shipped.sort(K)), 'Full-test anchor parity failed'
    ids = W / 'blocking/embedding/full/ids'
    i1 = pl.read_parquet(ids / 'test_s1.parquet', columns=['entity_id', 'country']).with_row_index('s1_idx').with_columns(pl.col('s1_idx').cast(pl.Int32))
    i2 = pl.read_parquet(ids / 'test_s23.parquet', columns=['entity_id']).with_row_index('cand_idx').with_columns(pl.col('cand_idx').cast(pl.Int32))
    def write(sel, path):
        grouped = sel.join(i2, on='cand_idx').group_by('s1_idx').agg(pl.col('entity_id').sort().str.join(',').alias('matched_entity_ids'))
        table = i1.join(grouped, on='s1_idx', how='left').sort('s1_idx').select(pl.col('entity_id').alias('source1_entity_id'), pl.col('matched_entity_ids').fill_null(''))
        assert table.height == 1732544
        table.write_csv(path, separator='\t', quote_style='never')
        return table
    write(reproduced, OUT / 'data/anchor_reproduced.tsv')
    assert digest(OUT / 'data/anchor_reproduced.tsv') == digest(M / 'ensemble_v1/build/v3anchor/output/matching_results.tsv')
    table = write(selected, OUT / 'output/matching_results.tsv')
    previous = pl.read_csv(VR / 'submissions/consensus/matching_results.tsv', separator='\t', infer_schema_length=0, quote_char=None, missing_utf8_is_empty_string=True)
    changes = table.join(previous, on='source1_entity_id', suffix='_previous').filter(pl.col('matched_entity_ids') != pl.col('matched_entity_ids_previous'))
    changes = changes.join(i1.select(pl.col('entity_id').alias('source1_entity_id'), 'country'), on='source1_entity_id')
    changes.write_parquet(OUT / 'data/changed_vs_vr2.parquet')
    grouped = selected.group_by('s1_idx').len()
    profile = i1.join(grouped, on='s1_idx', how='left').with_columns(pl.col('len').fill_null(0)).group_by('country').agg(pl.len().alias('queries'), pl.col('len').sum().alias('pairs'), pl.col('len').mean().alias('matches_per_query'), (pl.col('len') == 0).mean().alias('empty_fraction')).sort('country').to_dicts()
    result = dict(policy='blend_e0.25', correction_seeds=RC.SEEDS, correction_strength=1.,
                  utility_weights=dict(original=.75, evidence=.25), empty_status_guard=False,
                  all_countries=True, queries=table.height, pairs=selected.height,
                  anchor_pair_parity=True, anchor_tsv_byte_parity=True,
                  changed_queries_vs_vr2=changes.height,
                  changed_queries_by_country=dict(changes.group_by('country').len().sort('country').iter_rows()),
                  country_profile=profile, veto_pairs=veto.height,
                  model_sha256={str(p):digest(p) for p in model_paths()},
                  matching_sha256=digest(OUT / 'output/matching_results.tsv'),
                  seconds=time.time()-start, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    save('inference.json', result)
    log('INFERENCE COMPLETE', result)


def package():
    import subprocess, shutil, itertools, zipfile
    start = time.time()
    info = json.loads((OUT / 'inference.json').read_text())
    assert info['anchor_pair_parity'] and info['anchor_tsv_byte_parity']
    output = OUT / 'output'
    match = output / 'matching_results.tsv'
    assert digest(match) == info['matching_sha256']
    source = M / 'ensemble_v1/build/v3anchor/output/candidate_pairs.tsv'
    cand = output / 'candidate_pairs.tsv'
    shutil.copy2(source, cand)
    candidate_hash = digest(cand)
    assert candidate_hash == digest(source)
    sr = W.parent / 'student_resource'
    command = [sys.executable, str(sr / 'utils/validate_submission.py'), '--matching', str(match),
               '--test-dir', str(sr / 'dataset/test'), '--check-ids']
    validator = subprocess.run(command, capture_output=True, text=True, cwd=OUT)
    (OUT / 'logs/official_validator.log').write_text(validator.stdout + validator.stderr)
    assert validator.returncode == 0 and 'PASS' in validator.stdout, validator.stdout + validator.stderr
    log('Official matching validator passed with ID checks')
    ids = W / 'blocking/embedding/full/ids'
    i1 = pl.read_parquet(ids / 'test_s1.parquet', columns=['entity_id', 'country'])
    i2 = pl.read_parquet(ids / 'test_s23.parquet', columns=['entity_id', 'country'])
    required, known = set(i1['entity_id']), set(i2['entity_id'])
    seen, matched_ids = set(), set()
    count = matches = 0
    with cand.open() as cf, match.open() as mf:
        assert cf.readline().rstrip('\n') == 'source1_entity_id\tcandidate_entity_ids'
        assert mf.readline().rstrip('\n') == 'source1_entity_id\tmatched_entity_ids'
        for cline, mline in itertools.zip_longest(cf, mf):
            assert cline is not None and mline is not None
            q, cs = cline.rstrip('\n').split('\t')
            mq, ms = mline.rstrip('\n').split('\t')
            assert q == mq and q not in seen
            seen.add(q)
            cc, mm = (cs.split(',') if cs else []), (ms.split(',') if ms else [])
            cset, mset = set(cc), set(mm)
            assert len(cc) == len(cset) and len(mm) == len(mset)
            assert cset <= known and mset <= cset and not (mset & matched_ids)
            matched_ids.update(mset)
            count += len(cc)
            matches += len(mm)
            if len(seen) % 250000 == 0:
                log('Candidate and ownership checks', len(seen), '/', len(required))
    assert seen == required and len(seen) == 1732544 and count == 83761275 and matches == info['pairs']
    del required, known, seen, matched_ids
    table = pl.read_csv(match, separator='\t', quote_char=None, infer_schema_length=0, missing_utf8_is_empty_string=True)
    pairs = table.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids') != '')
    linked = pairs.join(i1.rename({'entity_id':'source1_entity_id'}), on='source1_entity_id').join(i2.rename({'entity_id':'matched_entity_ids', 'country':'match_country'}), on='matched_entity_ids')
    assert linked.height == matches and linked.filter(pl.col('country') != pl.col('match_country')).height == 0
    checks = dict(rows=table.height, matching_pairs=matches, candidate_pairs=count,
                  official_matching_validator_exit=validator.returncode, official_matching_id_check=True,
                  duplicate_query_rows=0, duplicate_pairs=0, unknown_candidate_ids=0,
                  matches_outside_candidates=0, ownership_violations=0, country_mismatches=0,
                  matching_sha256=digest(match), candidate_sha256=candidate_hash,
                  anchor_byte_parity=True, seconds_before_archive=time.time()-start)
    save('submission_integrity.json', checks)
    # Verify both the earlier protected files and the complete frozen Round C manifest.
    old = json.loads((RC_DIR / 'preservation.json').read_text())
    changed = [p for p, v in old.items() if digest(p) != v['sha256']]
    frozen = json.loads((RC_DIR / 'artifact_manifest.json').read_text())
    changed += [str(RC_DIR / p) for p, v in frozen.items() if digest(RC_DIR / p) != v['sha256']]
    assert not changed, changed
    checks.update(protected_prior_files=len(old), protected_round_c_files=len(frozen), changed_protected_files=changed)
    save('submission_integrity.json', checks)
    archive = OUT / 'blend_e0.25_outputs.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(match, 'output/matching_results.tsv')
        bundle.write(cand, 'output/candidate_pairs.tsv')
        bundle.write(OUT / 'submission_integrity.json', 'submission_integrity.json')
        bundle.write(OUT / 'inference.json', 'inference.json')
        bundle.write(OUT / 'validation_verification.json', 'validation_verification.json')
        bundle.writestr('README.txt', 'Frozen v2shash Round C blend_e0.25.\nThree address-number correction seeds 11/29/47 averaged in logit space; strength 1.\n75% original decoder utility + 25% Round B evidence decoder utility; ADD veto; global one owner.\nAll 1,732,544 test queries. Upload output/matching_results.tsv where a leaderboard TSV is requested.\nThis is an output-file bundle, not the final source-code package.\n')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    checks.update(archive=str(archive), archive_bytes=archive.stat().st_size,
                  archive_sha256=digest(archive), seconds=time.time()-start)
    save('submission_integrity.json', checks)
    (OUT / 'SHA256SUMS').write_text(f"{checks['matching_sha256']}  output/matching_results.tsv\n{candidate_hash}  output/candidate_pairs.tsv\n{checks['archive_sha256']}  blend_e0.25_outputs.zip\n")
    log('PACKAGING COMPLETE', checks)


def proxies():
    out = {}
    for name, path in [('blend_e0.25', OUT / 'data/test_selected.parquet'),
                       ('vr2_consensus', VR / 'data/sel_consensus.parquet')]:
        out[name] = S.france_proxies(pl.read_parquet(path, columns=K))
        log('France proxies', name, out[name])
    save('france_proxies.json', out)


def peers():
    """Rescore each ensemble round's strongest completed ordinary-validation run."""
    meta = pl.read_parquet(M / 'variance_study_20260926/val_meta.parquet')
    splits = pl.read_parquet(M / 'accuracy_lab_20260925/data/baseline_rows.parquet', columns=['s1_idx', 'eval_split'])
    meta = meta.join(splits, on='s1_idx')
    truth = pl.read_parquet(M / 'variance_study_20260926/val_truth.parquet')
    requested = S.score(pl.read_parquet(RC_DIR / 'data/ordinary_blend_e0.25_selected.parquet'), meta, truth)
    result = {}
    for folder in ('ensemble_v1', 'ensemble_v2', 'ensemble_v3'):
        completed = []
        for path in (M / folder / 'results').glob('eval_ENS_*.json'):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            val = data.get('val', {})
            selected = path.with_name(path.stem + '_val_selected.parquet')
            if val.get('n') == 220730 and 'macro_f05' in val and selected.exists():
                completed.append((val['macro_f05'], path, selected))
        if not completed:
            continue
        _, source, selected = max(completed, key=lambda x: x[0])
        rows = S.score(pl.read_parquet(selected, columns=K).with_columns(pl.col(K).cast(pl.Int32)), meta, truth)
        comparison = compare_rows(requested, rows)
        result[folder] = dict(selected_name=source.stem, source=str(source), selected=str(selected),
                              comparison=comparison, scope='Best recorded ordinary score among completed full-validation runs in this ensemble directory; diagnostic comparison only.')
        log('Peer comparison', folder, source.stem, comparison['baseline'], comparison['delta'])
    save('peer_comparisons.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify', 'infer', 'package', 'proxies', 'peers'])
    args = parser.parse_args()
    globals()[args.action]()
