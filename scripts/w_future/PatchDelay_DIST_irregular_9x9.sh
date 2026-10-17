#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../.."
python -u scripts/run_irregular.py \
  --data-path "${DATA_PATH:-dataset/forecasting/DIST_irregular_9x9.csv}" \
  --history 14400 --forecast 7200 --patch-seconds 960 \
  --batch-size "${BATCH_SIZE:-2}" --epochs "${EPOCHS:-1000}" \
  --patience "${PATIENCE:-10}" --seed "${SEED:-1}" \
  --delay-context "${DELAY_CONTEXT:-4}" --use-delay "${USE_DELAY:-1}" \
  --keep-rate "${KEEP_RATE:-1}" \
  --save-path "${SAVE_PATH:-result/DIST_irregular/PatchDelay}" "$@"
