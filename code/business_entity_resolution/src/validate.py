#!/opt/conda/bin/python3
"""Step 6: run the official submission validator (utils/validate_submission.py, stdlib only) on an output folder.

usage: /opt/conda/bin/python3 validate.py --output <dir with matching_results.tsv + candidate_pairs.tsv> --test-dir <dataset/test> [--check-ids]
       [--validator <path to the official utils/validate_submission.py>]   (default: the vendored copy in tools/, byte-identical)
Exit code 0 = PASS."""
import argparse, os, subprocess, sys
ap = argparse.ArgumentParser()
ap.add_argument('--output', required=True); ap.add_argument('--test-dir', required=True); ap.add_argument('--check-ids', action='store_true')
ap.add_argument('--validator', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'validate_submission.py'))
A = ap.parse_args()
cmd = [sys.executable, A.validator, '--matching', os.path.join(A.output, 'matching_results.tsv'), '--candidate', os.path.join(A.output, 'candidate_pairs.tsv'), '--test-dir', A.test_dir]
if A.check_ids: cmd.append('--check-ids')
print(' '.join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
