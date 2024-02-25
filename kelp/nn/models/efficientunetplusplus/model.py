"""Code credit: https://github.com/jlcsilva/segmentation_models.pytorch"""

from typing import Any, Callable, Dict, List, Optional, Union

from segmentation_models_pytorch.base import ClassificationHead, SegmentationHead, SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder

from kelp.nn.models.efficientunetplusplus.decoder import EfficientUnetPlusPlusDecoder


class EfficientUnetPlusPlus(SegmentationModel):
    """The EfficientUNet++ is a fully convolutional neural network for ordinary and medical image semantic segmentation.
    Consists of an *encoder* and a *decoder*, connected by *skip connections*. The encoder extracts features of
    different spatial resolutions, which are fed to the decoder through skip connections. The decoder combines its
    own feature maps with the ones from skip connections to produce accurate segmentations masks.  The EfficientUNet++
    decoder architecture is based on the UNet++, a model composed of nested U-Net-like decoder sub-networks. To
    increase performance and computational efficiency, the EfficientUNet++ replaces the UNet++'s blocks with
    inverted residual blocks with depthwise convolutions and embedded spatial and channel attention mechanisms.
    Synergizes well with EfficientNet encoders. Due to their efficient visual representations (i.e., using few channels
    to represent extracted features), EfficientNet encoders require few computation from the decoder.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Reference:
        [Silva et al. 2021](https://arxiv.org/abs/2106.11447)

    """

    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: Optional[List[int]] = None,
        squeeze_ratio: int = 1,
        expansion_ratio: int = 1,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable[[Any], Any]]] = None,
        aux_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        self.classes = classes

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = EfficientUnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "EfficientUNet++-{}".format(encoder_name)
        self.initialize()
