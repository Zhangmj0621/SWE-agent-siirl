#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

echo "[siirl-agentic-docs] Building combined site (EN + ZH) in one pass..."
./build.sh all
echo "[siirl-agentic-docs] Done. Site generated at build/index.html"
