import torch
import torch.nn as nn
import math

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DecoderBlock(nn.Module):
    def __init__(self, text_channels, kernel_size, kernel_stride, num_heads, downsample=False, residual=False):
        super(DecoderBlock, self).__init__()

        self.downsample = downsample
        self.residual = residual
        out_channels = text_channels // 2 if downsample else text_channels

        self.conv1 = nn.Conv1d(text_channels, out_channels, kernel_size, stride=kernel_stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=kernel_stride, padding=kernel_size//2)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.gelu = GELU()
        self.positional_encoding = PositionalEncoding(out_channels, 0)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=num_heads, batch_first=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bnorm(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bnorm(out)
        out = self.gelu(out)

        out = out.transpose(1, 2)  # N, L, C
        out = self.positional_encoding(out)
        out = out.transpose(1, 2)  # N, C, L

        out = self.transformer_layer(out.transpose(1, 2)).transpose(1, 2)  # N, C, L

        if self.residual and not self.downsample:
            out += x

        out = self.pool(out)

        return out


class Conv1DTransformerDecoder(nn.Module):
    def __init__(self, text_channels, kernel_size, kernel_stride, num_blocks, num_classes, num_heads):
        super(Conv1DTransformerDecoder, self).__init__()

        self.blocks = nn.ModuleList([
            DecoderBlock(
                text_channels if i == 0 else text_channels // (2 ** (i // 2)),
                kernel_size,
                kernel_stride,
                num_heads,
                downsample=(i+1) % 2 == 0,
                residual=True
            ) for i in range(num_blocks)
        ])

        final_channels = text_channels // (2 ** (num_blocks // 2))
        self.proj = nn.Linear(final_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        out = x
        for block in self.blocks:
            out = block(out)

        out = out.transpose(1, 2)  # N, L, C
        out = self.proj(out)
        out[:, :, 2] = self.sigmoid(out[:, :, 2])  # Apply sigmoid to a specific channel if needed

        return out