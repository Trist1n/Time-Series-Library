#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHON_BIN=/home/tri16/miniconda3/envs/tsl/bin/python
export MODELS="PatchTST iTransformer MSF_PatchTST"
export DATASETS="ECL"
export PRED_LENS="96 192 336 720"
export EPOCHS=10
export NUM_WORKERS=2
export SAVE_PREDICTIONS=0

bash scripts/long_term_forecast/run_ecl_weather_patch_itrans_msf.sh
