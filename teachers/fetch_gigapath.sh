#!/bin/bash
# Download the optional Prov-GigaPath weights (example teacher) via YOUR OWN
# HuggingFace access into teachers/GigaPath/.
#
# Teacher entries in the specs are EXAMPLE configurations, not statements
# about any particular training run. Prov-GigaPath is a third-party model
# governed by its own license — read the model card and confirm your intended
# use (including use of embeddings distilled from it) complies:
#   https://huggingface.co/prov-gigapath/prov-gigapath

set -e
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/GigaPath"
mkdir -p "$DEST"

if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download prov-gigapath/prov-gigapath pytorch_model.bin --local-dir "$DEST"
    huggingface-cli download prov-gigapath/prov-gigapath config.json --local-dir "$DEST"
else
    echo "huggingface-cli not found; install huggingface_hub or download manually:"
    echo "  https://huggingface.co/prov-gigapath/prov-gigapath"
    exit 1
fi

echo
echo "Recorded sha256 (compare against the model card / your own records):"
(cd "$DEST" && sha256sum pytorch_model.bin config.json)
