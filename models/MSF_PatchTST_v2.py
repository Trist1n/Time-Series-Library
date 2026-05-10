import torch
from torch import nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        if self.kernel_size <= 1:
            trend = x
        else:
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size - 1 - pad_left
            x_t = x.permute(0, 2, 1)
            front = x_t[:, :, :1].repeat(1, 1, pad_left)
            end = x_t[:, :, -1:].repeat(1, 1, pad_right)
            trend = self.avg(torch.cat([front, x_t, end], dim=-1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class MultiScalePatchHead(nn.Module):
    def __init__(self, pred_len, dropout, use_scale_fusion=True):
        super().__init__()
        self.use_scale_fusion = use_scale_fusion
        self.scales = (2, 4, 8)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool1d(kernel_size=scale, stride=scale, ceil_mode=True),
                nn.Flatten(start_dim=-1),
                nn.LazyLinear(pred_len),
            )
            for scale in self.scales
        ])
        if use_scale_fusion:
            self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))
            self.refine = nn.Sequential(
                nn.Linear(pred_len, pred_len),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pred_len, pred_len),
            )
        else:
            self.refine = nn.Sequential(
                nn.Linear(pred_len * len(self.scales), pred_len),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pred_len, pred_len),
            )

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        outs = [branch(x) for branch in self.branches]
        if self.use_scale_fusion:
            weights = F.softmax(self.scale_logits, dim=0)
            out = sum(weight * branch_out for weight, branch_out in zip(weights, outs))
            out = out + self.refine(out)
        else:
            out = self.refine(torch.cat(outs, dim=-1))
        return out.permute(0, 2, 1)


class FrequencyHead(nn.Module):
    def __init__(self, seq_len, pred_len, dropout, keep_ratio=0.25, use_topk=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_topk = use_topk
        self.keep_bins = max(2, int((seq_len // 2 + 1) * keep_ratio))
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len),
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)
        spectrum = torch.fft.rfft(x, dim=-1)
        filtered = torch.zeros_like(spectrum)

        if self.use_topk:
            magnitudes = spectrum.abs()
            if magnitudes.shape[-1] > 1:
                k = min(self.keep_bins, magnitudes.shape[-1] - 1)
                topk_idx = torch.topk(magnitudes[..., 1:], k=k, dim=-1).indices + 1
                filtered.scatter_(-1, topk_idx, spectrum.gather(-1, topk_idx))
                filtered[..., :1] = spectrum[..., :1]
            else:
                filtered = spectrum
        else:
            filtered[..., :self.keep_bins] = spectrum[..., :self.keep_bins]

        low_freq = torch.fft.irfft(filtered, n=self.seq_len, dim=-1)
        out = self.projection(low_freq)
        return out.permute(0, 2, 1)


class ChannelMixingHead(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, dropout):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_in, enc_in),
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len),
        )

    def forward(self, x):
        mixed = x + self.channel_mixer(x)
        out = self.temporal_projection(mixed.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class TrendHead(nn.Module):
    def __init__(self, seq_len, pred_len, dropout):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len),
        )

    def forward(self, x):
        out = self.projection(x.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class MSFPatchTSTV2Base(nn.Module):
    """
    Ablation-friendly MSF-PatchTST v2.

    The PatchTST backbone remains the stable main path. The residual path adds
    multi-scale seasonal modeling, adaptive frequency filtering, channel mixing,
    and an optional trend head in normalized space.
    """

    def __init__(
        self,
        configs,
        patch_len=16,
        stride=8,
        use_freq_topk=True,
        use_decomp=True,
        use_scale_fusion=True,
    ):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_decomp = use_decomp
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

        moving_avg = getattr(configs, 'moving_avg', 25)
        self.decomposition = MovingAverageDecomposition(moving_avg)
        self.multi_scale_head = MultiScalePatchHead(configs.pred_len, configs.dropout, use_scale_fusion)
        self.frequency_head = FrequencyHead(configs.seq_len, configs.pred_len, configs.dropout,
                                            use_topk=use_freq_topk)
        self.channel_head = ChannelMixingHead(configs.seq_len, configs.pred_len, configs.enc_in, configs.dropout)
        self.trend_head = TrendHead(configs.seq_len, configs.pred_len, configs.dropout)
        self.gate = nn.Sequential(
            nn.Linear(configs.enc_in, configs.enc_in),
            nn.Sigmoid()
        )
        self.branch_logits = nn.Parameter(torch.tensor([1.5, -0.5, -0.5, -0.5, -0.5]))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        x_patch = x_norm.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_patch)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        backbone = self.head(enc_out).permute(0, 2, 1)

        if self.use_decomp:
            seasonal, trend = self.decomposition(x_norm)
        else:
            seasonal = x_norm
            trend = torch.zeros_like(x_norm)

        multi_scale = self.multi_scale_head(seasonal)
        frequency = self.frequency_head(seasonal)
        channel = self.channel_head(x_norm)
        trend = self.trend_head(trend)

        weights = F.softmax(self.branch_logits, dim=0)
        residual_gate = self.gate(x_norm[:, -1, :]).unsqueeze(1)
        dec_out = weights[0] * backbone + residual_gate * (
            weights[1] * multi_scale
            + weights[2] * frequency
            + weights[3] * channel
            + weights[4] * trend
        )

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        raise ValueError('MSF_PatchTST_v2 currently supports forecasting tasks only')


class Model(MSFPatchTSTV2Base):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(
            configs,
            patch_len=patch_len,
            stride=stride,
            use_freq_topk=True,
            use_decomp=True,
            use_scale_fusion=True,
        )
