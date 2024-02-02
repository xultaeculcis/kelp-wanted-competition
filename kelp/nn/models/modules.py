from torch import nn as nn


class PreActivatedConv2dReLU(nn.Sequential):
    """
    Pre-activated 2D convolution, as proposed in https://arxiv.org/pdf/1603.05027.pdf.
    Feature maps are processed by a normalization layer,  followed by a ReLU activation and a 3x3 convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True,
    ) -> None:
        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        relu = nn.ReLU(inplace=True)

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        super(PreActivatedConv2dReLU, self).__init__(conv, bn, relu)


class DepthWiseConv2d(nn.Conv2d):
    """Depth-wise convolution operation"""

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__(channels, channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=channels)


class PointWiseConv2d(nn.Conv2d):
    """Point-wise (1x1) convolution operation"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1)
