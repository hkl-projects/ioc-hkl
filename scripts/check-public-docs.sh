#!/usr/bin/env bash
# Scan public paths for site-specific strings. Facility patterns: documentation/local/check-patterns.txt
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SCAN_GLOBS=(
  documentation
  iocBoot/iocpydev/*.example
  configure/RELEASE
  configure/RELEASE.local.example
  README.md
)

# Generic patterns (no facility host naming conventions)
PATTERNS=(
  '/home/controls'
)

LOCAL_PATTERNS="${ROOT}/documentation/local/check-patterns.txt"
if [[ -f "$LOCAL_PATTERNS" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -z "$line" ]] && continue
    PATTERNS+=("$line")
  done < "$LOCAL_PATTERNS"
fi

EXCLUDE='documentation/local/|documentation/local\.example/check-patterns'

fail=0
for pat in "${PATTERNS[@]}"; do
  if matches=$(git grep -n -E "$pat" -- "${SCAN_GLOBS[@]}" 2>/dev/null | grep -Ev "$EXCLUDE" || true); then
    if [[ -n "$matches" ]]; then
      echo "FAIL: pattern /$pat/ found in public paths:"
      echo "$matches"
      echo
      fail=1
    fi
  fi
done

for pat in "${PATTERNS[@]}"; do
  while IFS= read -r -d '' f; do
    [[ "$f" == documentation/local/* ]] && continue
    [[ "$f" == documentation/local.example/check-patterns* ]] && continue
    if grep -qE "$pat" "$f" 2>/dev/null; then
      echo "FAIL: pattern /$pat/ in $f"
      fail=1
    fi
  done < <(find documentation -type f -name '*.md' ! -path 'documentation/local/*' -print0 2>/dev/null)
  while IFS= read -r -d '' f; do
    if grep -qE "$pat" "$f" 2>/dev/null; then
      echo "FAIL: pattern /$pat/ in $f"
      fail=1
    fi
  done < <(find iocBoot/iocpydev -type f -name '*.example' -print0 2>/dev/null)
  for f in configure/RELEASE configure/RELEASE.local.example README.md; do
    [[ -f "$f" ]] && grep -qE "$pat" "$f" 2>/dev/null && { echo "FAIL: pattern /$pat/ in $f"; fail=1; }
  done
done

if [[ "$fail" -ne 0 ]]; then
  echo "Move site-specific content to documentation/local/ (gitignored)."
  exit 1
fi

echo "OK: no banned site-specific patterns in public documentation."
