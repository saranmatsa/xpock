#!/usr/bin/env bash
# count_loc.sh — Count qualifying LOC in a Rust source file
# Usage: ./count_loc.sh <file.rs>
#
# Output format:
#   Total qualifying LOC:   <all non-blank, non-comment lines>
#   Real code LOC (no tests): <non-test production code lines>
#   Test LOC:               <test block lines>
#
# Excludes: blank lines, line comments (//), inner doc comments (//!),
#           block comments (/*, */), and entire #[cfg(test)] mod blocks.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <file.rs>"
    exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    echo "Error: file not found: $FILE"
    exit 1
fi

python3 - "$FILE" <<'PYEOF'
import sys

path = sys.argv[1]

with open(path, encoding='utf-8') as f:
    lines = f.readlines()

def is_non_code(line):
    s = line.strip()
    return (
        not s
        or s.startswith('//')
        or s.startswith('//!')
        or s.startswith('/*')
        or s.startswith('*')
        or s.endswith('*/')
    )

# Step 1: strip blanks and comments
filtered = [l for l in lines if not is_non_code(l)]
total = len(filtered)

# Step 2: remove ALL test constructs:
#   - #[cfg(test)] mod <name> { ... } blocks (brace-matched, any mod name)
#   - #[test] fn <name>() { ... } functions (brace-matched)
#   - standalone #[test] and #[cfg(test)] attribute lines
real = []
i = 0
while i < len(filtered):
    line = filtered[i]
    stripped = line.strip()

    # Case A: #[cfg(test)] mod <name> { ... } — brace-matched module block (any name)
    if '#[cfg(test)]' in stripped and i + 1 < len(filtered):
        nxt = filtered[i + 1].strip()
        if nxt.startswith('mod ') and '{' in nxt:
            i += 1
            depth = nxt.count('{') - nxt.count('}')
            i += 1
            while i < len(filtered) and depth > 0:
                depth += filtered[i].count('{') - filtered[i].count('}')
                i += 1
            continue

    # Case B: #[test] — attribute line, possibly followed by a function body
    if '#[test]' in stripped:
        j = i + 1
        while j < len(filtered) and ('#[' in filtered[j].strip() or filtered[j].strip().startswith('/')):
            j += 1
        if j < len(filtered):
            fn_line = filtered[j].strip()
            if 'fn ' in fn_line and '{' in fn_line:
                depth = fn_line.count('{') - fn_line.count('}')
                i = j + 1
                while i < len(filtered) and depth > 0:
                    depth += filtered[i].count('{') - filtered[i].count('}')
                    i += 1
                continue
        i += 1
        continue

    # Case C: standalone #[cfg(test)] not part of a mod block
    if '#[cfg(test)]' in stripped:
        i += 1
        continue

    real.append(line)
    i += 1

real_count = len(real)
test_count = total - real_count

print(f'Total qualifying LOC: {total}')
print(f'Real code LOC (no tests): {real_count}')
print(f'Test LOC: {test_count}')
PYEOF
