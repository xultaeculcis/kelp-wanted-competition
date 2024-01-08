"""Code credit: https://github.com/jlcsilva/segmentation_models.pytorch"""
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import Attention
from torch import Tensor

from kelp.models.modules import PreActivatedConv2dReLU


class ASPP(nn.Module):
    """
    ASPP described in https://arxiv.org/pdf/1706.05587.pdf but without the concatenation of 1x1,
    original feature maps and global average pooling
    """

    def __init__(self, in_channels: int, out_channels: int, rate: Tuple[int, int, int] = (6, 12, 18)) -> None:
        super(ASPP, self).__init__()

        # Dilation rates of 6, 12 and 18 for the Atrous Spatial Pyramid Pooling blocks
        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        self.aspp_block4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

        self.output = nn.Conv2d((len(rate) + 1) * out_channels, out_channels, kernel_size=1)
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        x4 = self.aspp_block4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)

        return self.output(out)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionBlock(nn.Module):
    def __init__(self, skip_channels: int, in_channels: int, out_channels: int) -> None:
        super(AttentionBlock, self).__init__()

        if skip_channels != 0:
            self.encoder_conv = nn.Sequential(
                nn.BatchNorm2d(skip_channels),
                nn.ReLU(),
                nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
                # Attention is used before upsampling, so the encoder feature maps need to be downsampled
            )

        self.decoder_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(), nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.attn_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        # Apply BN, ReLU and 3x3 conv to incoming feature maps to obtain the desired number of feature maps
        # and be able to sum them
        if skip is not None:
            out = self.encoder_conv(skip) + self.decoder_conv(x)
        else:
            out = self.decoder_conv(x)
        out = self.attn_conv(out)  # Compute a BCHW attention mask
        return out * x  # Apply the attention mask to the input coming from the decoder


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.attention0 = AttentionBlock(skip_channels, in_channels, in_channels)
        self.conv1 = PreActivatedConv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = PreActivatedConv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        self.identity_conv = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        x = self.attention0(x, skip)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            identity = x
            x = self.attention1(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        identity = self.identity_conv(identity)
        return x + identity


class ResUnetPlusPlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [2 * head_channels] + [i * 2 for i in decoder_channels[:-1]]
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = [i * 2 for i in decoder_channels]  # decoder_channels

        self.center = ASPP(head_channels, in_channels[0])

        blocks = [
            DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_batchnorm=use_batchnorm,
                attention_type=attention_type,
            )
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.final_aspp = ASPP(out_channels[-1], out_channels[-1] // 2)

    def forward(self, *features: Any) -> Tensor:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        x = self.final_aspp(x)

        return x
