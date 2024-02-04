"""
Simple fully convolutional neural network (FCN) implementations. Code credit: https://github.com/microsoft/torchgeo/
"""

from torch import Tensor, nn


class FCN(nn.Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, in_channels: int, classes: int, num_filters: int = 256) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super().__init__()

        conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        conv5 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.backbone = nn.Sequential(
            conv1,
            nn.LeakyReLU(inplace=True),
            conv2,
            nn.LeakyReLU(inplace=True),
            conv3,
            nn.LeakyReLU(inplace=True),
            conv4,
            nn.LeakyReLU(inplace=True),
            conv5,
            nn.LeakyReLU(inplace=True),
        )

        self.last = nn.Conv2d(num_filters, classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.last(x)
        return x
