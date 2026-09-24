#!/usr/bin/env bash
# Mutation-testing harness.
#
# Three rules, each learned by getting it wrong:
#
#   1. RESTORE FROM GIT. An early version snapshotted the file to /tmp; a
#      timeout killed the loop mid-restore, the next run snapshotted the
#      ALREADY-MUTATED file as its "original", and a mutated implementation was
#      committed.
#
#   2. VERIFY THE BASELINE. A later version passed --timeout without
#      pytest-timeout installed, so pytest exited non-zero on every run and all
#      16 mutations reported "killed".
#
#   5. ONE OCCURRENCE ONLY. The replace below is `replace(old, new, 1)`, so a
#      statement duplicated in the source is mutated in only one place. An
#      `INSERT OR IGNORE` mutation once landed on an in-loop flush that small
#      tests never reach, and reported SURVIVED while the real path was
#      untouched. If a pattern appears more than once, either mutate a unique
#      anchor or refactor the duplication away.
#
#   4. NEVER LET PYTHON REUSE BYTECODE. `git checkout --` restores the .py but
#      Python can reuse a .pyc compiled from the MUTATED source, so the suite
#      fails on a clean tree and every later mutation aborts on the baseline
#      check. PYTHONDONTWRITEBYTECODE removes the failure mode entirely.
#
#   3. BASELINE AND MUTATION MUST RUN THE IDENTICAL COMMAND. The fix for (2)
#      used plain pytest for the baseline and `timeout 120 pytest` for the
#      mutation. `timeout` is GNU coreutils and absent on macOS, so the
#      mutation run failed with command-not-found — and reported "killed"
#      again. Any difference between the two invocations hides a broken runner.
#
# The common thread: a harness that cannot distinguish a failing TEST from a
# failing TEST RUNNER reports exactly what you hope to hear.
set -uo pipefail
FILE="$1"; TESTS="$2"; OLD="$3"; NEW="$4"; DESC="${5:-mutation}"

run_tests() { PYTHONDONTWRITEBYTECODE=1 python3 -m pytest "$TESTS" -q -p no:cacheprovider --timeout=120 >/dev/null 2>&1; }

git diff --quiet -- "$FILE" || { echo "ABORT: $FILE has uncommitted changes"; exit 1; }
run_tests || { echo "ABORT: baseline not green under the exact mutation command"; exit 1; }

trap 'git checkout -- "$FILE"; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null' EXIT INT TERM

python3 - "$FILE" "$OLD" "$NEW" <<'PY' || { printf "%-46s SKIP (no match)\n" "$DESC"; exit 0; }
import sys
from pathlib import Path
p, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
t = p.read_text()
sys.exit(3) if old not in t else p.write_text(t.replace(old, new, 1))
PY

if run_tests; then printf "%-46s %s\n" "$DESC" "SURVIVED"
else printf "%-46s %s\n" "$DESC" "killed"; fi
