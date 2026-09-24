#!/usr/bin/env bash
# Mutation-testing harness.
#
# Restores from GIT, never from a /tmp copy. An earlier version snapshotted the
# file to /tmp before each mutation; a timeout killed the loop mid-restore, the
# next run snapshotted the ALREADY-MUTATED file as its "original", and a mutated
# implementation was committed. Git is the only trustworthy baseline.
set -uo pipefail
FILE="$1"; TESTS="$2"; OLD="$3"; NEW="$4"; DESC="${5:-mutation}"

git diff --quiet -- "$FILE" || { echo "ABORT: $FILE has uncommitted changes"; exit 1; }
trap 'git checkout -- "$FILE"' EXIT INT TERM

python3 - "$FILE" "$OLD" "$NEW" <<'PY' || { printf "%-46s SKIP (no match)\n" "$DESC"; exit 0; }
import sys
from pathlib import Path
p, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
t = p.read_text()
sys.exit(3) if old not in t else p.write_text(t.replace(old, new, 1))
PY

if python3 -m pytest "$TESTS" -q -x --timeout=60 >/dev/null 2>&1; then
  printf "%-46s %s\n" "$DESC" "SURVIVED"
else
  printf "%-46s %s\n" "$DESC" "killed"
fi
