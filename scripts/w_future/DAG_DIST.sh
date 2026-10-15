#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"

# DELAY_CONTEXT includes the current patch: 4 means [q-3, q-2, q-1, q].
# USE_DELAY=0 disables the added window path for an ablation.
python ./scripts/run_benchmark.py \
  --config-path "rolling_forecast_config.json" \
  --data-name-list "DIST.csv" \
  --strategy-args "{\"horizon\":360,\"target_channel\":[[9,18]],\"num_rollings\":${NUM_ROLLINGS:-48000}}" \
  --model-name "dag.PatchDelay" \
  --model-hyper-params "{\"seq_len\":720,\"horizon\":360,\"d_model\":64,\"d_ff\":128,\"te_dim\":10,\"patch_len\":48,\"stride\":48,\"delay_context\":${DELAY_CONTEXT:-4},\"use_delay\":${USE_DELAY:-1},\"dropout\":0.0,\"batch_size\":${BATCH_SIZE:-64},\"lr\":0.001,\"lradj\":\"type3\",\"num_epochs\":${EPOCHS:-50},\"patience\":5,\"loss\":\"MAE\",\"norm\":true,\"fusion_method\":\"\",\"infer_use_future\":true}" \
  --gpus "${GPU:-0}" \
  --num-workers 1 \
  --timeout 60000 \
  --save-path "DIST/PatchDelay_S${DELAY_CONTEXT:-4}_use${USE_DELAY:-1}"
