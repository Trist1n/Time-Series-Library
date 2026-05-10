from pathlib import Path
import csv
import re

import numpy as np


RESULTS_DIR = Path("results")
OUTPUT = Path("results_ecl_weather_patch_itrans_msf.csv")


def parse_setting(setting):
    pattern = re.compile(
        r"^long_term_forecast_(?P<model_id>.+?)_(?P<model>PatchTST|iTransformer|MSF_PatchTST)_"
        r"custom_ft(?P<features>.+?)_sl(?P<seq_len>\d+)_ll(?P<label_len>\d+)_pl(?P<pred_len>\d+)_"
    )
    match = pattern.match(setting)
    if not match:
        return None
    row = match.groupdict()
    if row["model_id"].startswith("ECL_"):
        row["dataset"] = "Electricity"
    elif row["model_id"].startswith("weather_"):
        row["dataset"] = "Weather"
    elif row["model_id"].startswith("Exchange_"):
        row["dataset"] = "Exchange"
    elif row["model_id"].startswith("ili_"):
        row["dataset"] = "ILI"
    else:
        return None
    return row


def main():
    rows = []
    if not RESULTS_DIR.exists():
        raise SystemExit("No results directory found.")

    for metrics_file in RESULTS_DIR.glob("*/metrics.npy"):
        setting = metrics_file.parent.name
        parsed = parse_setting(setting)
        if parsed is None:
            continue
        mae, mse, rmse, mape, mspe = np.load(metrics_file)
        rows.append({
            "model": parsed["model"],
            "dataset": parsed["dataset"],
            "pred_len": int(parsed["pred_len"]),
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "setting": setting,
        })

    rows.sort(key=lambda r: (r["dataset"], r["pred_len"], r["model"]))
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "dataset", "pred_len", "mse", "mae", "rmse", "mape", "mspe", "setting"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
