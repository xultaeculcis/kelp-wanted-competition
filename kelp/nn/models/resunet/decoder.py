"""Code credit: https://github.com/jlcsilva/segmentation_models.pytorch"""
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import Attention
from torch import Tensor

from kelp.nn.models.modules import PreActivatedConv2dReLU


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


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        conv1 = PreActivatedConv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = PreActivatedConv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class ResUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        center: bool = False,
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
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

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

    def forward(self, *features: Any) -> Tensor:
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
