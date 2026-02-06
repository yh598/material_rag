"""
Optional helper to load an API key from a local file path.
Use:
  export AI_GATEWAY_API_KEY="$(python scripts/read_key.py /path/to/keyfile.txt)"
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
txt = p.read_text(encoding="utf-8", errors="ignore")
# Try to find a line like "Generated API Key: <key>"
for line in txt.splitlines():
    if "Generated API Key:" in line:
        print(line.split("Generated API Key:", 1)[1].strip())
        raise SystemExit(0)

# Fallback: print first non-empty token-like line (do not echo entire file)
for line in txt.splitlines():
    line = line.strip()
    if line and not line.startswith("---") and "WARNING" not in line:
        print(line)
        raise SystemExit(0)

raise SystemExit("No key found in file")
