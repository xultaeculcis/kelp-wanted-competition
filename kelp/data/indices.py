import abc

import torch
from torch import Tensor, nn
from torchgeo.transforms import AppendNDVI, AppendNDWI

_EPSILON = 1e-10


class AppendIndex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = -3

    def append_index(self, sample: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
        index = index.unsqueeze(self.dim)
        sample["image"] = torch.cat([sample["image"], index], dim=self.dim)
        return sample


class NirRedBandIndex(nn.Module, abc.ABC):
    def __init__(self, index_nir: int, index_red: int) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red

    @abc.abstractmethod
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        pass

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if "image" in sample:
            index = self._compute_index(
                nir=sample["image"][..., self.index_nir, :, :],
                red=sample["image"][..., self.index_red, :, :],
            )
            self.append_index(sample=sample, index=index)

        return sample


class AppendATSAVI(NirRedBandIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 1.22 * (nir - 1.22 * red - 0.03) / (1.22 * nir + red - 1.22 * 0.03 + 0.08 * (1 + 1.22 * 1.22))


class AppendAFRI1600(AppendIndex):
    def __init__(self, index_swir: int, index_nir: int) -> None:
        super().__init__()
        self.index_swir = index_swir
        self.index_nir = index_nir

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir))

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if "image" in sample:
            index = self._compute_index(
                swir=sample["image"][..., self.index_swir, :, :],
                nir=sample["image"][..., self.index_nir, :, :],
            )
            self.append_index(sample=sample, index=index)

        return sample


class AppendAVI(NirRedBandIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2 * nir - red


class AppendARVI(NirRedBandIndex):
    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return -0.18 + 1.17 * ((nir - red) / (nir + red))


class AppendBWDRVI(AppendIndex):
    def __init__(self, index_swir: int, index_nir: int) -> None:
        super().__init__()
        self.index_swir = index_swir
        self.index_nir = index_nir

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir))

    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if "image" in sample:
            index = self._compute_index(
                swir=sample["image"][..., self.index_swir, :, :],
                nir=sample["image"][..., self.index_nir, :, :],
            )
            self.append_index(sample=sample, index=index)

        return sample


class AppendBWDRV:
    def __init__(self, index_nir: int, index_blue: int) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_blue = index_blue


class AppendClGreen:
    def __init__(
        self,
        index_nir: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_green = index_green


class AppendCVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green


class AppendDEMWM:
    def __init__(self, index_dem: int) -> None:
        super().__init__()
        self.index_dem = index_dem


class AppendWDRVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendVARIGreen:
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_red = index_red
        self.index_green = index_green
        self.index_blue = index_blue


class AppendTVI:
    def __init__(
        self,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_red = index_red
        self.index_green = index_green


class AppendTNDVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendSQRTNIRR:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendRBNDVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_blue = index_blue


class AppendSRSWIRNIR:
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
    ) -> None:
        super().__init__()
        self.index_swir = index_swir
        self.index_nir = index_nir


class AppendSRNIRSWIR:
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
    ) -> None:
        super().__init__()
        self.index_swir = index_swir
        self.index_nir = index_nir


class AppendSRNIRR:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendSRNIRG:
    def __init__(
        self,
        index_nir: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_green = index_green


class AppendSRGR:
    def __init__(
        self,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_red = index_red
        self.index_green = index_green


class AppendPNDVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green
        self.index_blue = index_blue


class AppendNormR:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green


class AppendNormNIR:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green


class AppendNormG:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green


class AppendNDWIWM:
    def __init__(
        self,
        index_nir: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_green = index_green


class AppendNDVIWM:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendNLI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendMSAVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendMSRNirRed:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendMCARI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green


class AppendMVI:
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
    ) -> None:
        super().__init__()
        self.index_swir = index_swir
        self.index_nir = index_nir


class AppendMCRIG:
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_green = index_green
        self.index_blue = index_blue


class AppendLogR:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendH:
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_red = index_red
        self.index_green = index_green
        self.index_blue = index_blue


class AppendI:
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_red = index_red
        self.index_green = index_green
        self.index_blue = index_blue


class AppendIPVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green


class AppendGVMI:
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
    ) -> None:
        super().__init__()
        self.index_swir = index_swir
        self.index_nir = index_nir


class AppendGBNDVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendGRNDVI:
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_green = index_green
        self.index_blue = index_blue


class AppendGNDVI:
    def __init__(
        self,
        index_nir: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_green = index_green


class AppendGARI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_green = index_green
        self.index_blue = index_blue


class AppendEVI22:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendEVI2:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendEVI:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_blue = index_blue


class AppendDVIMSS:
    def __init__(
        self,
        index_nir: int,
        index_red: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_red = index_red


class AppendGDVI:
    def __init__(
        self,
        index_nir: int,
        index_green: int,
    ) -> None:
        super().__init__()
        self.index_nir = index_nir
        self.index_green = index_green


class AppendCI:
    def __init__(
        self,
        index_red: int,
        index_blue: int,
    ) -> None:
        super().__init__()
        self.index_red = index_red
        self.index_blue = index_blue


INDICES = {
    "ATSAVI": AppendATSAVI(index_nir=1, index_red=2),
    "AFRI1600": AppendAFRI1600(index_swir=0, index_nir=1),
    "AVI": AppendAVI(index_nir=1, index_red=2),
    "ARVI": AppendARVI(index_nir=1, index_red=2),
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
