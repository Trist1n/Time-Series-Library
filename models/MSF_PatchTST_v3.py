import math

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


class TokenLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # [B, C, D, P] -> normalize D
        return self.norm(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


class DynamicMultiScaleTokenMixer(nn.Module):
    def __init__(self, d_model, dropout, kernels=(3, 5, 7)):
        super().__init__()
        self.kernels = kernels
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1),
            )
            for k in kernels
        ])
        self.scale_router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, len(kernels)),
        )
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = TokenLayerNorm(d_model)

    def forward(self, x):
        # x: [B, C, D, P]
        b, c, d, p = x.shape
        x_flat = x.reshape(b * c, d, p)
        branch_outs = [branch(x_flat).reshape(b, c, d, p) for branch in self.branches]

        context = x.mean(dim=(1, 3))
        weights = F.softmax(self.scale_router(context), dim=-1)
        mixed = sum(
            weights[:, i].view(b, 1, 1, 1) * branch_outs[i]
            for i in range(len(branch_outs))
        )
        mixed = self.proj(mixed.reshape(b * c, d, p)).reshape(b, c, d, p)
        return self.norm(x + self.dropout(mixed))


class PeriodLagMixer(nn.Module):
    def __init__(self, d_model, seq_len, patch_num, stride, top_k, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.patch_num = patch_num
        self.stride = stride
        self.top_k = max(1, top_k)
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = TokenLayerNorm(d_model)

    def forward(self, x, seasonal):
        # x: [B, C, D, P], seasonal: [B, L, C]
        b, c, d, p = x.shape
        if p <= 1:
            return x

        spectrum = torch.fft.rfft(seasonal, dim=1)
        magnitudes = spectrum.abs().mean(dim=-1)
        if magnitudes.shape[-1] <= 1:
            return x

        k = min(self.top_k, magnitudes.shape[-1] - 1)
        top_values, top_idx = torch.topk(magnitudes[:, 1:], k=k, dim=-1)
        freq_idx = top_idx + 1
        periods = self.seq_len / freq_idx.float()
        lags = torch.clamp(torch.round(periods / self.stride).long(), min=1, max=p - 1)

        patch_idx = torch.arange(p, device=x.device).view(1, 1, p)
        gather_idx = (patch_idx - lags.unsqueeze(-1)) % p
        gather_idx = gather_idx.view(b, k, 1, 1, p).expand(b, k, c, d, p)

        x_expand = x.unsqueeze(1).expand(b, k, c, d, p)
        shifted = torch.gather(x_expand, dim=-1, index=gather_idx)
        weights = F.softmax(top_values, dim=-1).view(b, k, 1, 1, 1)
        mixed = torch.sum(shifted * weights, dim=1)
        mixed = self.proj(mixed.reshape(b * c, d, p)).reshape(b, c, d, p)
        return self.norm(x + self.dropout(mixed))


class SpectralTokenGate(nn.Module):
    def __init__(self, d_model, dropout, keep_ratio=0.5):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = TokenLayerNorm(d_model)

    def forward(self, x):
        # x: [B, C, D, P]
        b, c, d, p = x.shape
        spectrum = torch.fft.rfft(x, dim=-1)
        bins = spectrum.shape[-1]
        filtered = torch.zeros_like(spectrum)
        filtered[..., :1] = spectrum[..., :1]

        if bins > 1:
            k = max(1, min(bins - 1, int(math.ceil((bins - 1) * self.keep_ratio))))
            energy = spectrum.abs().mean(dim=2)
            top_idx = torch.topk(energy[..., 1:], k=k, dim=-1).indices + 1
            gather_idx = top_idx.unsqueeze(2).expand(b, c, d, k)
            filtered.scatter_(-1, gather_idx, spectrum.gather(-1, gather_idx))

        spectral = torch.fft.irfft(filtered, n=p, dim=-1)
        token_gate = self.gate(x.mean(dim=-1)).unsqueeze(-1)
        spectral = self.proj((spectral * token_gate).reshape(b * c, d, p)).reshape(b, c, d, p)
        return self.norm(x + self.dropout(spectral))


class FrequencyGuidedChannelAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.value_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.out_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = TokenLayerNorm(d_model)

    def forward(self, x, source):
        # x: [B, C, D, P], source: [B, L, C]
        b, c, d, p = x.shape
        freq = torch.fft.rfft(source, dim=1).abs().permute(0, 2, 1)
        freq = F.normalize(freq, p=2, dim=-1)
        scores = torch.matmul(freq, freq.transpose(1, 2))
        scores = scores / torch.clamp(self.temperature.abs(), min=1e-3)
        attn = F.softmax(scores, dim=-1)

        value = self.value_proj(x.reshape(b * c, d, p)).reshape(b, c, d, p)
        mixed = torch.einsum('bij,bjdp->bidp', attn, value)
        mixed = self.out_proj(mixed.reshape(b * c, d, p)).reshape(b, c, d, p)
        return self.norm(x + self.dropout(mixed))


class AdvancedMSFBlock(nn.Module):
    def __init__(
        self,
        d_model,
        seq_len,
        patch_num,
        stride,
        top_k,
        dropout,
        use_period=True,
        use_channel_graph=True,
        use_spectral_gate=True,
    ):
        super().__init__()
        self.use_period = use_period
        self.use_channel_graph = use_channel_graph
        self.use_spectral_gate = use_spectral_gate

        self.multi_scale = DynamicMultiScaleTokenMixer(d_model, dropout)
        self.period_lag = PeriodLagMixer(d_model, seq_len, patch_num, stride, top_k, dropout)
        self.spectral_gate = SpectralTokenGate(d_model, dropout)
        self.channel_graph = FrequencyGuidedChannelAttention(d_model, dropout)
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.norm = TokenLayerNorm(d_model)

    def forward(self, x, seasonal, source):
        residual = x
        x = self.multi_scale(x)
        if self.use_period:
            x = self.period_lag(x, seasonal)
        if self.use_spectral_gate:
            x = self.spectral_gate(x)
        if self.use_channel_graph:
            x = self.channel_graph(x, source)

        b, c, d, p = x.shape
        fused = self.fusion(x.reshape(b * c, d, p)).reshape(b, c, d, p)
        return self.norm(residual + fused)


class TrendResidualHead(nn.Module):
    def __init__(self, seq_len, pred_len, dropout):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, trend):
        out = self.projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
        return torch.tanh(self.alpha) * out


class MSFPatchTSTV3Base(nn.Module):
    """
    Representation-level MSF-PatchTST.

    Multi-scale, period-lag, spectral, and frequency-guided channel operations
    refine encoder patch tokens before the single forecasting head.
    """

    def __init__(
        self,
        configs,
        patch_len=16,
        stride=8,
        use_period=True,
        use_channel_graph=True,
        use_spectral_gate=True,
    ):
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

        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

        moving_avg = getattr(configs, 'moving_avg', 25)
        top_k = min(max(1, getattr(configs, 'top_k', 5)), 5)
        self.decomposition = MovingAverageDecomposition(moving_avg)
        self.msf_block = AdvancedMSFBlock(
            configs.d_model,
            configs.seq_len,
            self.patch_num,
            stride,
            top_k,
            configs.dropout,
            use_period=use_period,
            use_channel_graph=use_channel_graph,
            use_spectral_gate=use_spectral_gate,
        )
        self.trend_residual = TrendResidualHead(configs.seq_len, configs.pred_len, configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        seasonal, trend = self.decomposition(x_norm)

        x_patch = x_norm.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_patch)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        enc_out = self.msf_block(enc_out, seasonal, x_norm)

        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out + self.trend_residual(trend)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        raise ValueError('MSF_PatchTST_v3 currently supports forecasting tasks only')


class Model(MSFPatchTSTV3Base):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(
            configs,
            patch_len=patch_len,
            stride=stride,
            use_period=True,
            use_channel_graph=True,
            use_spectral_gate=True,
        )
