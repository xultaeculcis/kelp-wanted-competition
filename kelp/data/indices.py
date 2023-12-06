# mypy: disable-error-code="override"
import abc
from typing import Any

import torch
from torch import Tensor, nn
from torchgeo.transforms import AppendNDVI, AppendNDWI

_EPSILON = 1e-10


class AppendIndex(nn.Module, abc.ABC):
    def __init__(self, **band_kwargs: Any) -> None:
        super().__init__()
        assert all(k.startswith("index_") for k in band_kwargs), (
            "Passed keyword arguments must start with 'index_' followed by band name! "
            f"Found following keys: {list(band_kwargs.keys())}"
        )
        self.dim = -3
        self.band_kwargs = band_kwargs

    def _append_index(self, sample: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
        index = index.unsqueeze(self.dim)
        sample["image"] = torch.cat([sample["image"], index], dim=self.dim)
        return sample

    @abc.abstractmethod
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if "image" in sample:
            compute_kwargs: dict[str, Tensor] = {
                k.replace("index_", ""): sample["image"][..., v, :, :] for k, v in self.band_kwargs.items()
            }
            index = self._compute_index(**compute_kwargs)
            self.append_index(sample=sample, index=index)
        return sample


class AppendATSAVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 1.22 * (nir - 1.22 * red - 0.03) / (1.22 * nir + red - 1.22 * 0.03 + 0.08 * (1 + 1.22**2))


class AppendAFRI1600(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir))


class AppendAVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2 * nir - red


class AppendARVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return -0.18 + 1.17 * ((nir - red) / (nir + red))


class AppendBWDRVI(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir))


class AppendBWDRV(AppendIndex):
    def _compute_index(self, nir: Tensor, blue: Tensor) -> Tensor:
        return (0.1 * nir - blue) / (0.1 * nir + blue)


class AppendClGreen(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / green - 1


class AppendCVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir * (red / (green**2))


class AppendDEMWM(AppendIndex):
    def _compute_index(self, dem: Tensor) -> Tensor:
        return torch.maximum(dem, torch.zeros_like(dem))


class AppendWDRVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (0.1 * nir - red) / (0.1 * nir + red)


class AppendVARIGreen(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (green - red) / (green + red - blue)


class AppendTVI(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return torch.sqrt(((red - green) / (red + green)) + 0.5)


class AppendTNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt((nir - red) / (nir + red) + 0.5)


class AppendSQRTNIRR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt(nir / red)


class AppendRBNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + blue)) / (nir + red + blue)


class AppendSRSWIRNIR(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return swir / nir


class AppendSRNIRSWIR(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir / swir


class AppendSRNIRR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return nir / red


class AppendSRNIRG(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / green


class AppendSRGR(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return green / red


class AppendPNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + green + blue)) / (nir + red + green + blue)


class AppendNormR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return red / (nir + red + green)


class AppendNormNIR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir / (nir + red + green)


class AppendNormG(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return green / (nir + red + green)


class AppendNDWIWM(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.maximum((nir - red) / (nir + red), torch.zeros_like(nir))


class AppendNDVIWM(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendNLI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendMSAVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendMSRNirRed(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendMCARI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendMVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendMCRIG(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendLogR(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendH(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendIPVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendGVMI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendGBNDVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendGRNDVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendGNDVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendGARI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendEVI22(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendEVI2(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendEVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendDVIMSS(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendGDVI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


class AppendCI(AppendIndex):
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass


INDICES = {
    "ATSAVI": AppendATSAVI(index_nir=1, index_red=2),  #
    "AFRI1600": AppendAFRI1600(index_swir=0, index_nir=1),  #
    "AVI": AppendAVI(index_nir=1, index_red=2),  #
    "ARVI": AppendARVI(index_nir=1, index_red=2),  #
    "BWDRVI": AppendBWDRV(index_nir=1, index_blue=4),
    "ClGreen": AppendClGreen(index_nir=1, index_green=3),
    "CVI": AppendCVI(index_nir=1, index_red=2, index_green=3),
    "CI": AppendCI(index_red=2, index_blue=4),
    "GDVI": AppendGDVI(index_nir=1, index_green=3),
    "DVIMSS": AppendDVIMSS(index_nir=1, index_red=2),
    "EVI": AppendEVI(index_nir=1, index_red=2, index_blue=4),
    "EVI2": AppendEVI2(index_nir=1, index_red=2),
    "EVI22": AppendEVI22(index_nir=1, index_red=2),
    "GARI": AppendGARI(index_nir=1, index_red=2, index_green=3, index_blue=4),
    "GNDVI": AppendGNDVI(index_nir=1, index_green=3),
    "GRNDVI": AppendGRNDVI(index_nir=1, index_green=3, index_blue=4),
    "GBNDVI": AppendGBNDVI(index_nir=1, index_red=2),
    "GVMI": AppendGVMI(index_swir=0, index_nir=1),
    "IPVI": AppendIPVI(index_nir=1, index_red=2, index_green=3),
    "I": AppendI(index_red=2, index_green=3, index_blue=4),
    "H": AppendH(index_red=2, index_green=3, index_blue=4),
    "LogR": AppendLogR(index_nir=1, index_red=2),
    "mCRIG": AppendMCRIG(index_nir=1, index_green=3, index_blue=4),
    "MVI": AppendMVI(index_swir=0, index_nir=1),
    "MCARI": AppendMCARI(index_nir=1, index_red=2, index_green=3),
    "MSRNirRed": AppendMSRNirRed(index_nir=1, index_red=2),
    "MSAVI": AppendMSAVI(index_nir=1, index_red=2),
    "NLI": AppendNLI(index_nir=1, index_red=2),
    "NDVI": AppendNDVI(index_nir=1, index_red=2),
    "NDVIWM": AppendNDVIWM(index_nir=1, index_red=2),
    "NDWI": AppendNDWI(index_nir=1, index_green=3),
    "NDWIWM": AppendNDWIWM(index_nir=1, index_green=3),
    "NormG": AppendNormG(index_nir=1, index_red=2, index_green=3),
    "NormNIR": AppendNormNIR(index_nir=1, index_red=2, index_green=3),
    "NormR": AppendNormR(index_nir=1, index_red=2, index_green=3),
    "PNDVI": AppendPNDVI(index_nir=1, index_red=2, index_green=3, index_blue=4),
    "SRGR": AppendSRGR(index_red=2, index_green=3),
    "SRNIRG": AppendSRNIRG(index_nir=1, index_green=3),
    "SRNIRR": AppendSRNIRR(index_nir=1, index_red=2),
    "SRNIRSWIR": AppendSRNIRSWIR(index_swir=0, index_nir=1),
    "SRSWIRNIR": AppendSRSWIRNIR(index_swir=0, index_nir=1),
    "RBNDVI": AppendRBNDVI(index_nir=1, index_red=2, index_blue=4),
    "SQRTNIRR": AppendSQRTNIRR(index_nir=1, index_red=2),
    "TNDVI": AppendTNDVI(index_nir=1, index_red=2),
    "TVI": AppendTVI(index_red=2, index_green=3),
    "VARIGreen": AppendVARIGreen(index_red=2, index_green=3, index_blue=4),
    "WDRVI": AppendWDRVI(index_nir=1, index_red=2),
    "DEMWM": AppendDEMWM(index_dem=7),
}
