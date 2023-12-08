# mypy: disable-error-code="override"
import abc
from typing import Any

import torch
from torch import Tensor, nn
from torchgeo.transforms import AppendNDVI, AppendNDWI

_EPSILON = 1e-10


class AppendIndex(nn.Module, abc.ABC):
    def __init__(
        self,
        index_qa: int = 5,
        normalize: bool = True,
        normalize_percentile_low: float = 0.01,
        normalize_percentile_high: float = 0.99,
        **band_kwargs: Any,
    ) -> None:
        super().__init__()
        assert all(k.startswith("index_") for k in band_kwargs), (
            "Passed keyword arguments must start with 'index_' followed by band name! "
            f"Found following keys: {list(band_kwargs.keys())}"
        )
        self.index_qa = index_qa
        self.dim = -3
        self.normalize = normalize
        self.normalize_percentile_low = normalize_percentile_low
        self.normalize_percentile_high = normalize_percentile_high
        self.band_kwargs = band_kwargs

    def _append_index(self, sample: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
        index = index.unsqueeze(self.dim)
        sample["image"] = torch.cat([sample["image"], index], dim=self.dim)
        return sample

    def _mask_using_qa_band(self, index: Tensor, sample: dict[str, Tensor]) -> Tensor:
        qa_band = sample["image"][..., self.index_qa, :, :]
        qa_band = torch.where(qa_band == 0, 1, torch.nan)
        index = index * qa_band
        return index

    def _maybe_normalize(self, index: Tensor) -> Tensor:
        if not self.normalize:
            return index
        min_val = torch.nanquantile(index, self.normalize_percentile_low)
        max_val = torch.nanquantile(index, self.normalize_percentile_high)
        index = torch.clamp(index, min_val, max_val)
        return (index - min_val) / (max_val - min_val)

    @abc.abstractmethod
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if "image" in sample:
            compute_kwargs: dict[str, Tensor] = {
                k.replace("index_", ""): sample["image"][..., v, :, :] for k, v in self.band_kwargs.items()
            }
            index = self._compute_index(**compute_kwargs)
            index = self._mask_using_qa_band(index, sample)
            index = self._maybe_normalize(index)
            self._append_index(sample=sample, index=index)
        return sample


class AppendATSAVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 1.22 * (nir - 1.22 * red - 0.03) / (1.22 * nir + red - 1.22 * 0.03 + 0.08 * (1 + 1.22**2) + _EPSILON)


class AppendAFRI1600(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir + _EPSILON))


class AppendAVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2 * nir - red


class AppendARVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return -0.18 + 1.17 * ((nir - red) / (nir + red + _EPSILON))


class AppendBWDRVI(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir + _EPSILON))


class AppendBWDRV(AppendIndex):
    def _compute_index(self, nir: Tensor, blue: Tensor) -> Tensor:
        return (0.1 * nir - blue) / (0.1 * nir + blue + _EPSILON)


class AppendClGreen(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / (green + _EPSILON) - 1


class AppendCVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir * (red / (green**2 + _EPSILON))


class AppendDEMWM(AppendIndex):
    def _compute_index(self, dem: Tensor) -> Tensor:
        return torch.maximum(dem, torch.zeros_like(dem))


class AppendWDRVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (0.1 * nir - red) / (0.1 * nir + red + _EPSILON)


class AppendVARIGreen(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (green - red) / (green + red - blue + _EPSILON)


class AppendTVI(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return torch.sqrt(((red - green) / (red + green + _EPSILON)) + 0.5)


class AppendTNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt((nir - red) / (nir + red + _EPSILON) + 0.5)


class AppendSQRTNIRR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt(nir / (red + _EPSILON))


class AppendRBNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + blue)) / (nir + red + blue + _EPSILON)


class AppendSRSWIRNIR(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return swir / (nir + _EPSILON)


class AppendSRNIRSWIR(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir / (swir + _EPSILON)


class AppendSRNIRR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return nir / (red + _EPSILON)


class AppendSRNIRG(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / (green + _EPSILON)


class AppendSRGR(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return green / (red + _EPSILON)


class AppendPNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + green + blue)) / (nir + red + green + blue + _EPSILON)


class AppendNormR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return red / (nir + red + green + _EPSILON)


class AppendNormNIR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir / (nir + red + green + _EPSILON)


class AppendNormG(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return green / (nir + red + green + _EPSILON)


class AppendNDWIWM(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return torch.maximum((nir - green) / (nir + green + _EPSILON), torch.zeros_like(nir))


class AppendNDVIWM(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.maximum((nir - red) / (nir + red + _EPSILON), torch.zeros_like(nir))


class AppendNLI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (nir**2 - red) / (nir**2 + red + _EPSILON)


class AppendMSAVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (2 * nir + 1 - torch.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2


class AppendMSRNirRed(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return ((nir / red) - 1) / torch.sqrt((nir / (red + _EPSILON)) + 1)


class AppendMCARI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return ((nir - red) - 0.2 * (nir - green)) * (nir / (red + _EPSILON))


class AppendMVI(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir / (swir + _EPSILON)


class AppendMCRIG(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (blue**-1 - green**-1) * nir


class AppendLogR(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.log(nir / (red + _EPSILON) + _EPSILON)


class AppendH(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.arctan(((2 * red - green - blue) / 30.5) * (green - blue))


class AppendI(AppendIndex):
    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (1 / 30.5) * (red + green + blue + _EPSILON)


class AppendIPVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return (nir / (nir + red + _EPSILON) / 2) * (((red - green) / red + green) + 1 + _EPSILON)


class AppendGVMI(AppendIndex):
    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return ((nir + 0.1) - (swir + 0.02)) / ((nir + 0.1) + (swir + 0.02) + _EPSILON)


class AppendGBNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (green + blue)) / (nir + green + blue + _EPSILON)


class AppendGRNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return (nir - (red + green)) / (nir + red + green + _EPSILON)


class AppendGNDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return (nir - green) / (nir + green + _EPSILON)


class AppendGARI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (green - (blue - red))) / (nir - (green + (blue - red)) + _EPSILON)


class AppendEVI22(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.5 * (nir - red) / (nir + 2.4 * red + 1 + _EPSILON)


class AppendEVI2(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.4 * (nir - red) / (nir + red + 1 + _EPSILON)


class AppendEVI(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor, blue: Tensor) -> Tensor:
        return torch.clamp(
            2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + _EPSILON)),
            min=-20_000,
            max=20_000,
        )


class AppendDVIMSS(AppendIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.4 * nir - red


class AppendGDVI(AppendIndex):
    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir - green


class AppendCI(AppendIndex):
    def _compute_index(self, red: Tensor, blue: Tensor) -> Tensor:
        return (red - blue) / (red + _EPSILON)


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
    "GRNDVI": AppendGRNDVI(index_nir=1, index_red=2, index_green=3),
    "GBNDVI": AppendGBNDVI(index_nir=1, index_green=3, index_blue=4),
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
