import torch
import torch.nn as nn

import math


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

# pulled from Jiachen's conformer implementation
def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.
    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.
    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)

class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


# full decoder
class Conv1DTransformerDecoder(nn.Module):
    def __init__(
        self,
        text_channels,
        kernel_size,
        kernel_stride,
        num_blocks,
        num_classes,  ##4 
        # add params for transformer mechanism
        num_transformer_layers,
        num_heads,
    ):
        super(Conv1DTransformerDecoder, self).__init__()

        self.blocks = []
        for i in range(num_blocks):
            downsample = False
            if (i+1) % 2 == 0:
                downsample = True
            self.blocks += [
                DecoderBlock(
                    text_channels,
                    kernel_size,
                    kernel_stride,
                    downsample,
                    True,
                )
            ]
            if (i+1) % 2 == 0:
                text_channels = int(text_channels // 2)
        self.blocks = nn.ModuleList(self.blocks)

        # add transformer encoder
        self.positional_encoding = PositionalEncoding(text_channels, 0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_channels, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # final learned feature space projection
        self.proj = nn.Linear(
            text_channels, 3 + num_classes
        )  # feature dim is 4 - start, end, exists, [type - 3 classes]

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        out = x
        for block in self.blocks:
            out = block(out)
        
        out = self.positional_encoding(out)
        out = self.transformer_encoder(out, src_key_padding_mask=mask)

        out = self.proj(out)

        # activation functions for proper prediction bounds
        out[:, :, 2] = self.sigmoid(out[:, :, 2])

        return out


# decoder block
class DecoderBlock(nn.Module):
    def __init__(
        self,
        text_channels,
        kernel_size,
        kernel_stride,
        downsample=False,
        residual=False,
    ):
        super(DecoderBlock, self).__init__()

        self.downsample = downsample
        self.residual = residual

        out_channels = text_channels
        if downsample:
            out_channels = int(text_channels // 2)

        # downsample during conv
        self.conv1 = nn.Conv1d(
            text_channels,
            out_channels,
            kernel_size,
            kernel_stride,
            padding="same",
            groups = out_channels
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            kernel_stride,
            padding="same",
            groups = out_channels
        )
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.gelu = GELU()

        self.poolx2 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x - N, L, C
        out = self.conv1(torch.transpose(x, -1, -2))  # N, C, L
        out = self.bnorm(out)  # N, C, L

        out = self.gelu(out) 
        out = self.conv2(out)

        out = torch.transpose(out, -1, -2) # N, L, C

        if self.residual and not self.downsample:
            out = out + x

        out = torch.transpose(out, -1, -2) # N, C, L

        out = self.poolx2(out)
        out = self.gelu(out) 

        out = torch.transpose(out, -1, -2)  # N, L, C

        return out


# decoder = Conv1DTransformerDecoder(text_channels=768, kernel_size=3, kernel_stride=1, num_blocks=4, num_classes=3, num_transformer_layers=4, num_heads=4)
# sample_input = torch.randn((32, 1024, 768)) # B, L, C
# output = decoder(sample_input)
# print(output.shape)