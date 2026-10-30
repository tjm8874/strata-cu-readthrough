#!/bin/sh
# Strata 読解本 -> Studio Hub ポータル (~/llm-lab/comfyui-studio/apps/strata-readthrough)
set -eu
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$HOME/llm-lab/comfyui-studio/apps/strata-readthrough"
mkdir -p "$DST/docs" "$DST/diagrams"
rsync -a "$SRC/index.html" "$DST/index.html"
rsync -a --include='*/' --include='*.html' --exclude='*' "$SRC/docs/" "$DST/docs/"
rsync -a "$SRC/diagrams/" "$DST/diagrams/"
echo "synced: $SRC -> $DST"
