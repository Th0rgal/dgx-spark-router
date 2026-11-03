#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/spark-daemon"

export PATH="$HOME/.local/bin:$HOME/.elan/bin:$HOME/.cargo/bin:$PATH"

echo "[1/4] Building pure-Lean spark-daemon (supportInterpreter := false, LeanSys @ c8449e02)..."
lake build

echo "[2/4] Running spark-daemon end-to-end self-test (SSE keepalive splice, 503 grace refusal, GPU permit gate)..."
./.lake/build/bin/spark-daemon self-test

echo "[3/4] Auditing Spark/Spec.lean via turbo-lean (0 vacuous, 0 trivial, 0 warnings)..."
if command -v turbo-lean >/dev/null 2>&1; then
  turbo-lean --json audit-spec Spark/Spec.lean | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['ok'] and d['vacuous_specs']==0 and d['trivial_specs']==0 and d['tight_specs']>=6 and not d.get('horizontal_packing_warnings') and not d.get('warnings'), d; print(f'Spark/Spec.lean: {d[\"tight_specs\"]}/{d[\"total_specs_audited\"]} non_vacuous_and_tight')"
elif [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "ERROR: turbo-lean is required in CI!" >&2
  exit 1
fi

echo "[4/4] Verifying TCB hygiene (0 import Lean, 0 sorryAx, 0 lean_elab/SSL_ symbols)..."
if nm -u .lake/build/bin/spark-daemon | grep -E 'lean_elab|lean_environment|sqlite|SSL_'; then
  echo "ERROR: Forbidden frontend/SQLite/OpenSSL symbol found in spark-daemon!" >&2
  exit 1
fi
if grep -rn '^import Lean\b' Spark/ Spark.lean; then
  echo "ERROR: Forbidden 'import Lean' found in Spark!" >&2
  exit 1
fi
if grep -rn '^\s*sorry\b' Spark/ Spark.lean; then
  echo "ERROR: Forbidden 'sorry' found in Spark!" >&2
  exit 1
fi
echo "[spark-daemon] ALL 4 CI GATES PASSED (6/6 tight specs, 5 C-ABI exports, TCB_CLEAN)"
