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


class MultiScalePatchHead(nn.Module):
    def __init__(self, seq_len, pred_len, dropout):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool1d(kernel_size=scale, stride=scale, ceil_mode=True),
                nn.Flatten(start_dim=-1),
                nn.LazyLinear(pred_len),
            )
            for scale in (2, 4, 8)
        ])
        self.mix = nn.Sequential(
            nn.Linear(pred_len * 3, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len),
        )

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        outs = [branch(x) for branch in self.branches]
        out = self.mix(torch.cat(outs, dim=-1))
        return out.permute(0, 2, 1)


class FrequencyHead(nn.Module):
    def __init__(self, seq_len, pred_len, dropout, keep_ratio=0.25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.keep_bins = max(2, int((seq_len // 2 + 1) * keep_ratio))
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len),
        )

    def forward(self, x):
        # Keep dominant low-frequency structure and extrapolate it with a light head.
        x = x.permute(0, 2, 1)
        spectrum = torch.fft.rfft(x, dim=-1)
        filtered = torch.zeros_like(spectrum)
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


class Model(nn.Module):
    """
    Multi-Scale Frequency enhanced PatchTST for long-term forecasting.
    The PatchTST backbone remains the main path; multi-scale, frequency, and
    channel-mixing heads provide gated residual forecasts in normalized space.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
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

        self.multi_scale_head = MultiScalePatchHead(configs.seq_len, configs.pred_len, configs.dropout)
        self.frequency_head = FrequencyHead(configs.seq_len, configs.pred_len, configs.dropout)
        self.channel_head = ChannelMixingHead(configs.seq_len, configs.pred_len, configs.enc_in, configs.dropout)
        self.gate = nn.Sequential(
            nn.Linear(configs.enc_in, configs.enc_in),
            nn.Sigmoid()
        )
        self.branch_logits = nn.Parameter(torch.tensor([2.0, -2.0, -2.0, -2.0]))

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
        multi_scale = self.multi_scale_head(x_norm)
        frequency = self.frequency_head(x_norm)
        channel = self.channel_head(x_norm)

        weights = F.softmax(self.branch_logits, dim=0)
        residual_gate = self.gate(x_norm[:, -1, :]).unsqueeze(1)
        dec_out = weights[0] * backbone + residual_gate * (
            weights[1] * multi_scale + weights[2] * frequency + weights[3] * channel
        )

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        raise ValueError('MSF_PatchTST currently supports forecasting tasks only')
