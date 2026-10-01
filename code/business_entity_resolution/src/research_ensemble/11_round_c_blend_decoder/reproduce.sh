#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
TASK_PYTHON="${TASK_PYTHON:-/opt/conda/bin/python3}"
TASK_RELEASE=matching/v2shash/round_c/test_release
"$TASK_PYTHON" -u "$TASK_RELEASE/release.py" prepare
"$TASK_PYTHON" -u "$TASK_RELEASE/release.py" verify
"$TASK_PYTHON" -u "$TASK_RELEASE/release.py" infer
"$TASK_PYTHON" -u "$TASK_RELEASE/release.py" package
"$TASK_PYTHON" -u "$TASK_RELEASE/release.py" proxies
"$TASK_PYTHON" -u "$TASK_RELEASE/release.py" peers
"$TASK_PYTHON" "$TASK_RELEASE/write_report.py"
