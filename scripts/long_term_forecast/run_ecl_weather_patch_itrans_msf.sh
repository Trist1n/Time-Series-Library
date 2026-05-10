#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PRED_LENS="${PRED_LENS:-96 192 336 720}"
ILI_PRED_LENS="${ILI_PRED_LENS:-24 36 48 60}"
MODELS="${MODELS:-PatchTST iTransformer MSF_PatchTST}"
DATASETS="${DATASETS:-ECL Weather}"
EPOCHS="${EPOCHS:-10}"
ITR="${ITR:-1}"
NUM_WORKERS="${NUM_WORKERS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

run_one() {
  local model="$1"
  local dataset="$2"
  local pred_len="$3"

  local root_path data_path model_id enc_in batch_size n_heads e_layers d_model d_ff lr extra_epochs seq_len label_len
  d_model=512
  d_ff=2048
  lr=0.0001
  extra_epochs="$EPOCHS"
  seq_len=96
  label_len=48

  if [[ "$dataset" == "ECL" ]]; then
    root_path="./dataset/electricity/"
    data_path="electricity.csv"
    model_id="ECL_96_${pred_len}"
    enc_in=321
    batch_size=16
    n_heads=8
  elif [[ "$dataset" == "Weather" ]]; then
    root_path="./dataset/weather/"
    data_path="weather.csv"
    model_id="weather_96_${pred_len}"
    enc_in=21
    batch_size=32
    n_heads=8
  elif [[ "$dataset" == "Exchange" ]]; then
    root_path="./dataset/exchange_rate/"
    data_path="exchange_rate.csv"
    model_id="Exchange_96_${pred_len}"
    enc_in=8
    batch_size=32
    n_heads=8
  elif [[ "$dataset" == "ILI" ]]; then
    root_path="./dataset/illness/"
    data_path="national_illness.csv"
    model_id="ili_36_${pred_len}"
    enc_in=7
    batch_size=32
    n_heads=4
    e_layers=4
    seq_len=36
    label_len=18
    if [[ "$model" == "PatchTST" || "$model" == "MSF_PatchTST" ]]; then
      d_model=1024
      if [[ "$pred_len" != "24" ]]; then
        d_model=2048
      fi
      if [[ "$pred_len" == "60" ]]; then
        n_heads=16
      fi
    fi
  else
    echo "Unknown dataset: ${dataset}" >&2
    exit 1
  fi

  if [[ "$model" == "iTransformer" ]]; then
    e_layers=3
    d_ff=512
    lr=0.0005
  else
    e_layers=2
  fi

  if [[ "$model" == "PatchTST" && "$dataset" == "Weather" ]]; then
    if [[ "$pred_len" == "192" ]]; then
      n_heads=16
    elif [[ "$pred_len" == "336" || "$pred_len" == "720" ]]; then
      batch_size=128
      n_heads=4
    else
      n_heads=4
    fi
  fi

  echo "Running ${model} ${dataset} pred_len=${pred_len}"
  "$PYTHON_BIN" -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "$root_path" \
    --data_path "$data_path" \
    --model_id "$model_id" \
    --model "$model" \
    --data custom \
    --features M \
    --seq_len "$seq_len" \
    --label_len "$label_len" \
    --pred_len "$pred_len" \
    --e_layers "$e_layers" \
    --d_layers 1 \
    --factor 3 \
    --enc_in "$enc_in" \
    --dec_in "$enc_in" \
    --c_out "$enc_in" \
    --des Exp \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --n_heads "$n_heads" \
    --batch_size "$batch_size" \
    --learning_rate "$lr" \
    --train_epochs "$extra_epochs" \
    --num_workers "$NUM_WORKERS" \
    --save_predictions "$SAVE_PREDICTIONS" \
    --itr "$ITR"
}

for model in $MODELS; do
  for dataset in $DATASETS; do
    pred_lens="$PRED_LENS"
    if [[ "$dataset" == "ILI" ]]; then
      pred_lens="$ILI_PRED_LENS"
    fi
    for pred_len in $pred_lens; do
      run_one "$model" "$dataset" "$pred_len"
    done
  done
done
