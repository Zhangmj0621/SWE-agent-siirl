#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TARGET=${1:-all}

case "$TARGET" in
  en|zh) ;;
  all) ;;
  *)
    echo "Target must be en, zh, or all"
    exit 1
    ;;
esac

cd "$SCRIPT_DIR"
if [ "$TARGET" = "all" ]; then
  UI_LANG=${SIIRL_DOC_LANG:-en}
  SIIRL_DOC_LANG=$UI_LANG sphinx-build -W -b html -D language=$UI_LANG --conf-dir ./ ./ ./build
else
  SIIRL_DOC_LANG=$TARGET sphinx-build -W -b html -D language=$TARGET --conf-dir ./ ./$TARGET ./build/$TARGET
fi
