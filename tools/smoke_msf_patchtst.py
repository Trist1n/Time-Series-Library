from argparse import Namespace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.MSF_PatchTST import Model


def main():
    cfg = Namespace(
        task_name="long_term_forecast",
        seq_len=96,
        pred_len=96,
        d_model=64,
        dropout=0.1,
        factor=3,
        n_heads=4,
        d_ff=128,
        e_layers=1,
        activation="gelu",
        enc_in=21,
    )
    model = Model(cfg)
    x = torch.randn(2, 96, 21)
    x_mark = torch.randn(2, 96, 4)
    y = torch.randn(2, 144, 21)
    y_mark = torch.randn(2, 144, 4)
    out = model(x, x_mark, y, y_mark)
    assert out.shape == (2, 96, 21), out.shape
    print(out.shape)


if __name__ == "__main__":
    main()
