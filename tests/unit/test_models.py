import pytest
import torch

from kelp.nn.models.efficientunetplusplus.model import EfficientUnetPlusPlus
from kelp.nn.models.resunet.model import ResUnet
from kelp.nn.models.resunetplusplus.model import ResUnetPlusPlus

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.requires_gpu
def test_efficientunet_plus_plus() -> None:
    # Arrange
    model = EfficientUnetPlusPlus().to(device=_device)
    model.eval()
    x = torch.rand((32, 3, 256, 256)).to(device=_device)

    # Act
    with torch.no_grad():
        out = model(x)

    # Assert
    assert out.shape == (32, 1, 256, 256)


@pytest.mark.requires_gpu
def test_resunet() -> None:
    # Arrange
    model = ResUnet().to(device=_device)
    model.eval()
    x = torch.rand((32, 3, 256, 256)).to(device=_device)

    # Act
    with torch.no_grad():
        out = model(x)

    # Assert
    assert out.shape == (32, 1, 256, 256)


@pytest.mark.requires_gpu
def test_resunet_plus_plus() -> None:
    # Arrange
    model = ResUnetPlusPlus().to(device=_device)
    model.eval()
    x = torch.rand((32, 3, 256, 256)).to(device=_device)

    # Act
    with torch.no_grad():
        out = model(x)

    # Assert
    assert out.shape == (32, 1, 256, 256)
