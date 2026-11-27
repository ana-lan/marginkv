#!/usr/bin/env bash
# scripts/11_serve.sh
# Phase 8a: start one vLLM server for a setting (qf | df | df_fp8). Run from the repo root in the vLLM venv:
#   nohup bash scripts/11_serve.sh df > artifacts/logs/serve_df.log 2>&1 &
set -euo pipefail
SETTING="${1:?usage: 11_serve.sh qf|df|df_fp8}"
PORT="${PORT:-8000}"
LOGPROBS_MODE="${LOGPROBS_MODE:-processed_logprobs}"
read -r BASE REV SEED < <(python -c "from marginkv.config import load_config as l; c = l(); print(c.models.qwen3.name, c.models.qwen3.revision, c.seed)")
MERGED="$(pwd)/artifacts/lora/df/merged"
case "$SETTING" in
  qf)     MODEL="$BASE";   MODEL_REV=(--revision "$REV"); KV=auto ;;
  df)     MODEL="$MERGED"; MODEL_REV=();                  KV=auto ;;
  df_fp8) MODEL="$MERGED"; MODEL_REV=();                  KV=fp8 ;;
  *) echo "unknown setting: $SETTING" >&2; exit 1 ;;
esac
export VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1
echo "setting=$SETTING model=$MODEL kv=$KV logprobs_mode=$LOGPROBS_MODE seed=$SEED"
exec vllm serve "$MODEL" "${MODEL_REV[@]}" \
  --tokenizer "$BASE" --tokenizer-revision "$REV" \
  --served-model-name marginkv --dtype bfloat16 --kv-cache-dtype "$KV" \
  --enable-prefix-caching --max-model-len 512 --gpu-memory-utilization 0.90 \
  --seed "$SEED" --max-logprobs 20 --logprobs-mode "$LOGPROBS_MODE" \
  --enable-prompt-tokens-details --port "$PORT"