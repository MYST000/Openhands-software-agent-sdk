#!/usr/bin/env bash
set -euo pipefail
FLOWPILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAVA_HOME="$FLOWPILOT_ROOT/runtime/browsecomp-native/java21/jdk-21.0.12.1+1-jre"
export PATH="$JAVA_HOME/bin:$PATH"
export HF_HUB_CACHE="$FLOWPILOT_ROOT/runtime/browsecomp-native/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# This sparse runtime uses no CUDA or GPU model inference.
export CUDA_VISIBLE_DEVICES=""
exec "$FLOWPILOT_ROOT/.venv-retrieval/bin/python" "$FLOWPILOT_ROOT/runtime/browsecomp-native/serve.py" "$@"
