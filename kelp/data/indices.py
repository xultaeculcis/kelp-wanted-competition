import torch
from torch import nn
from torchgeo.transforms import AppendNDVI, AppendNDWI

_EPSILON = 1e-10


class AppendIndex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = -3

    def append_index(self, sample: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
        if "image" in sample:
            index = index.unsqueeze(self.dim)
            sample["image"] = torch.cat([sample["image"], index], dim=self.dim)
        return sample


INDICES = {
    "ASAVI": AppendIndex(),
    "AFRI1600": AppendIndex(),
    "AVI": AppendIndex(),
    "ARVI": AppendIndex(),
    "BWDRVI": AppendIndex(),
    "Clgreen": AppendIndex(),
    "CVI": AppendIndex(),
    "CI": AppendIndex(),
    "GDVI": AppendIndex(),
    "DVIMSS": AppendIndex(),
    "EVI": AppendIndex(),
    "EVI2": AppendIndex(),
    "EVI22": AppendIndex(),
    "GARI": AppendIndex(),
    "GNDVI": AppendIndex(),
    "GRNDVI": AppendIndex(),
    "GBNDVI": AppendIndex(),
    "GVMI": AppendIndex(),
    "IPVI": AppendIndex(),
    "I": AppendIndex(),
    "LogR": AppendIndex(),
    "mCRIG": AppendIndex(),
    "MVI": AppendIndex(),
    "MNDVI": AppendIndex(),
    "MCARI": AppendIndex(),
    "MSRNir_Red": AppendIndex(),
    "MSAVI": AppendIndex(),
    "NLI": AppendIndex(),
    "NDVI": AppendNDVI(index_nir=1, index_red=2),
    "NDVIWM": AppendIndex(),
    "NDWI": AppendNDWI(index_nir=1, index_green=3),
    "NDWIWM": AppendIndex(),
    "Norm_G": AppendIndex(),
    "Norm_NIR": AppendIndex(),
    "Norm_B": AppendIndex(),
    "PNDVI": AppendIndex(),
    "SRGR": AppendIndex(),
    "SRNIRG": AppendIndex(),
    "SRNIRR": AppendIndex(),
    "SRNIRSWIR": AppendIndex(),
    "SRSWIRNIR": AppendIndex(),
    "RBNDVI": AppendIndex(),
    "SQRTNIRR": AppendIndex(),
    "TNDVI": AppendIndex(),
    "TVI": AppendIndex(),
    "VARIgreen": AppendIndex(),
    "WDRVI": AppendIndex(),
    "DEMWM": AppendIndex(),
}
