#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRED_LENS="${PRED_LENS:-96 192 336 720}"
MODELS="${MODELS:-PatchTST MSF_PatchTST MSF_PatchTST_v2_noFreqTopK MSF_PatchTST_v2_noDecomp MSF_PatchTST_v2 MSF_PatchTST_v2_noScaleFusion}"
EPOCHS="${EPOCHS:-10}"
ITR="${ITR:-1}"
NUM_WORKERS="${NUM_WORKERS:-10}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/tri16/miniconda3/envs/tsl/bin/python" ]]; then
    PYTHON_BIN="/home/tri16/miniconda3/envs/tsl/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

run_one() {
  local model="$1"
  local pred_len="$2"
  local n_heads=8
  local batch_size=32

  if [[ "$model" == "PatchTST" ]]; then
    if [[ "$pred_len" == "192" ]]; then
      n_heads=16
    elif [[ "$pred_len" == "96" || "$pred_len" == "336" || "$pred_len" == "720" ]]; then
      n_heads=4
    fi
    if [[ "$pred_len" == "336" || "$pred_len" == "720" ]]; then
      batch_size=128
    fi
  fi

  echo "Running ${model} Weather pred_len=${pred_len}"
  "$PYTHON_BIN" -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id "weather_96_${pred_len}" \
    --model "$model" \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des Exp \
    --d_model 512 \
    --d_ff 2048 \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --learning_rate 0.0001 \
    --train_epochs "$EPOCHS" \
    --num_workers "$NUM_WORKERS" \
    --save_predictions "$SAVE_PREDICTIONS" \
    --itr "$ITR"
}

for model in $MODELS; do
  for pred_len in $PRED_LENS; do
    run_one "$model" "$pred_len"
  done
done
