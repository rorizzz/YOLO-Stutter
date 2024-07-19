from multi_block import Conv1DTransformerDecoder as Decoder
# from conv1d_transformer import Conv1DTransformerDecoder as Decoder
# from multi_block import MultiheadAttentionBlock as MA
import torch
import torch.nn as nn


def print_shapes(model, x):

    print("Input Shape:", x.shape)
    
    # x = torch.transpose(x, 0, 1)
    # print("After Transposing:", x.shape)

    for block in model.blocks:
        x = block(x)
        print("After DecoderBlock:", x.shape)
    
    x = model.positional_encoding(x)
    print("After positional encoding: ", x.shape)


    # x = nn.functional.adaptive_avg_pool2d(x, (1, int(x.shape[-2] / model.downsample_factor)))
    # print("After adaptive_avg_pool2d:", x.shape)

    # x = torch.transpose(x, 1, -1).squeeze(2)
    # print("before transformer:", x.shape)

    # # Add transformer prints
    x = model.transformer_encoder(x)
    print("After Transformer Encoder:", x.shape)

    x = model.proj(x)

    x[:, :, 2] = model.sigmoid(x[:, :, 2])

    # x = model.proj(x)
    # print("After Linear Projection:", x.shape)

    # x = x.squeeze(-2)
    # print("After Squeezing:", x.shape)
    # x = model(x)

    return x



text_channels=768
kernel_size=3
kernel_stride=1
num_blocks=4
num_classes=4
downsample_factor=16
num_transformer_layers=8
n_heads=8


decoder = Decoder(
        text_channels,
        kernel_size,
        kernel_stride,
        num_blocks,
        num_classes,
        #num_transformer_layers,
        n_heads
)


# Create a random tensor input
soft_attention = torch.randn(64, 1024, 768)
mask = torch.ones((64, 64), dtype=torch.bool)

# Pass the input tensor through the decoder and print shapes of intermediate tensors
output = decoder(soft_attention)
# output = print_shapes(decoder, soft_attention)
print("Final Output Shape:", output.shape)

# multi_attention = MA(channels=768, num_heads=4)
# ma_output = multi_attention(soft_attention)
# print(ma_output.shape)
