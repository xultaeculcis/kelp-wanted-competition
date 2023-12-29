# mypy: disable-error-code="override"
from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import kornia.augmentation as K
import torch
from torch import Tensor

from kelp import consts


class AppendIndex(K.IntensityAugmentationBase2D, abc.ABC):
    def __init__(
        self,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **band_kwargs: Any,
    ) -> None:
        super().__init__(p=1, same_on_batch=True)
        assert all(k.startswith("index_") for k in band_kwargs), (
            "Passed keyword arguments must start with 'index_' followed by band name! "
            f"Found following keys: {list(band_kwargs.keys())}"
        )
        self.index_qa = index_qa
        self.dim = -3
        self.mask_using_qa = mask_using_qa
        self.band_kwargs = band_kwargs

    def _append_index(self, sample: Tensor, index: Tensor) -> Tensor:
        index = index.unsqueeze(self.dim)
        sample = torch.cat([sample, index], dim=self.dim)
        return sample

    def _mask_using_qa_band(self, index: Tensor, sample: Tensor) -> Tensor:
        if not self.mask_using_qa:
            return index
        min_val = index.min()
        qa_band = sample[..., self.index_qa, :, :]
        index = torch.where(qa_band == 0, index, min_val)
        return index

    @abc.abstractmethod
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        compute_kwargs: Dict[str, Tensor] = {
            k.replace("index_", ""): input[..., v, :, :] for k, v in self.band_kwargs.items()
        }
        index = self._compute_index(**compute_kwargs)
        index = self._mask_using_qa_band(index=index, sample=input)
        input = self._append_index(sample=input, index=index)
        return input


class AppendNDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (nir - red) / (nir + red + consts.data.EPS)


class AppendNDWI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return (nir - green) / (nir + green + consts.data.EPS)


class AppendATSAVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (
            1.22
            * (nir - 1.22 * red - 0.03)
            / (1.22 * nir + red - 1.22 * 0.03 + 0.08 * (1 + 1.22**2) + consts.data.EPS)
        )


class AppendAFRI1600(AppendIndex):
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir + consts.data.EPS))


class AppendAVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2 * nir - red


class AppendARVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return -0.18 + 1.17 * ((nir - red) / (nir + red + consts.data.EPS))


class AppendBWDRVI(AppendIndex):
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir + consts.data.EPS))


class AppendBWDRV(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, blue: Tensor) -> Tensor:
        return (0.1 * nir - blue) / (0.1 * nir + blue + consts.data.EPS)


class AppendClGreen(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / (green + consts.data.EPS) - 1


class AppendCVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir * (red / (green**2 + consts.data.EPS))


class AppendDEMWM(AppendIndex):
    def __init__(
        self,
        index_dem: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_dem=index_dem,
        )

    def _compute_index(self, dem: Tensor) -> Tensor:
        return torch.maximum(dem, torch.zeros_like(dem))


class AppendWDRVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (0.1 * nir - red) / (0.1 * nir + red + consts.data.EPS)


class AppendVARIGreen(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (green - red) / (green + red - blue + consts.data.EPS)


class AppendTVI(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return torch.sqrt(((red - green) / (red + green + consts.data.EPS)) + 0.5)


class AppendTNDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt((nir - red) / (nir + red + consts.data.EPS) + 0.5)


class AppendSQRTNIRR(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt(nir / (red + consts.data.EPS))


class AppendRBNDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + blue)) / (nir + red + blue + consts.data.EPS)


class AppendSRSWIRNIR(AppendIndex):
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return swir / (nir + consts.data.EPS)


class AppendSRNIRSWIR(AppendIndex):
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir / (swir + consts.data.EPS)


class AppendSRNIRR(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return nir / (red + consts.data.EPS)


class AppendSRNIRG(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / (green + consts.data.EPS)


class AppendSRGR(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return green / (red + consts.data.EPS)


class AppendPNDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + green + blue)) / (nir + red + green + blue + consts.data.EPS)


class AppendNormR(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return red / (nir + red + green + consts.data.EPS)


class AppendNormNIR(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir / (nir + red + green + consts.data.EPS)


class AppendNormG(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return green / (nir + red + green + consts.data.EPS)


class AppendNDWIWM(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return torch.maximum((nir - green) / (nir + green + consts.data.EPS), torch.zeros_like(nir))


class AppendNDVIWM(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.maximum((nir - red) / (nir + red + consts.data.EPS), torch.zeros_like(nir))


class AppendNLI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (nir**2 - red) / (nir**2 + red + consts.data.EPS)


class AppendMSAVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (2 * nir + 1 - torch.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2


class AppendMSRNirRed(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return ((nir / red + consts.data.EPS) - 1) / torch.sqrt((nir / (red + consts.data.EPS)) + 1)


class AppendMCARI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return ((nir - red) - 0.2 * (nir - green)) * (nir / (red + consts.data.EPS))


class AppendMVI(AppendIndex):
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir / (swir + consts.data.EPS)


class AppendMCRIG(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (blue**-1 - green**-1) * nir


class AppendLogR(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.log(nir / (red + consts.data.EPS) + consts.data.EPS)


class AppendH(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.arctan(((2 * red - green - blue) / 30.5) * (green - blue))


class AppendI(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (1 / 30.5) * (red + green + blue + consts.data.EPS)


class AppendIPVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return (nir / (nir + red + consts.data.EPS) / 2) * (((red - green) / red + green) + 1 + consts.data.EPS)


class AppendGVMI(AppendIndex):
    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return ((nir + 0.1) - (swir + 0.02)) / ((nir + 0.1) + (swir + 0.02) + consts.data.EPS)


class AppendGBNDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (green + blue)) / (nir + green + blue + consts.data.EPS)


class AppendGRNDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return (nir - (red + green)) / (nir + red + green + consts.data.EPS)


class AppendGNDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return (nir - green) / (nir + green + consts.data.EPS)


class AppendGARI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (green - (blue - red))) / (nir - (green + (blue - red)) + consts.data.EPS)


class AppendEVI22(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.5 * (nir - red) / (nir + 2.4 * red + 1 + consts.data.EPS)


class AppendEVI2(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.4 * (nir - red) / (nir + red + 1 + consts.data.EPS)


class AppendEVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, blue: Tensor) -> Tensor:
        return torch.clamp(
            2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + consts.data.EPS)),
            min=-20_000,
            max=20_000,
        )


class AppendDVIMSS(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.4 * nir - red


class AppendGDVI(AppendIndex):
    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir - green


class AppendCI(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, blue: Tensor) -> Tensor:
        return (red - blue) / (red + consts.data.EPS)


class AppendCHLA(AppendIndex):
    def __init__(
        self,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, blue: Tensor, green: Tensor) -> Tensor:
        # Yes, I know we should use coastal band here instead of blue, but we don't get to have coastal band
        return 4.23 * torch.pow((green / (blue + consts.data.EPS)), 3.94)


class AppendCYA(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return 115_530.31 * torch.pow(((green * blue) / (red + consts.data.EPS)), 2.38)


class AppendTURB(AppendIndex):
    def __init__(
        self,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, blue: Tensor, green: Tensor) -> Tensor:
        return 8.93 * (green / (blue + consts.data.EPS)) - 6.39


class AppendCDOM(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return 537 * torch.exp(-2.93 * green / (red + consts.data.EPS))


class AppendDOC(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return 432 * torch.exp(-2.24 * green / (red + consts.data.EPS))


class AppendWaterColor(AppendIndex):
    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        mask_using_qa: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            mask_using_qa=mask_using_qa,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return 25366 * torch.exp(-4.53 * green / (red + consts.data.EPS))


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
    "GRNDVI": AppendGRNDVI(index_nir=1, index_red=2, index_green=3),
    "GBNDVI": AppendGBNDVI(index_nir=1, index_green=3, index_blue=4),
    "GVMI": AppendGVMI(index_swir=0, index_nir=1),
    # "IPVI": AppendIPVI(index_nir=1, index_red=2, index_green=3),  # Do not use - produces nan and/or inf vals
    "I": AppendI(index_red=2, index_green=3, index_blue=4),
    "H": AppendH(index_red=2, index_green=3, index_blue=4),
    "LogR": AppendLogR(index_nir=1, index_red=2),
    # "mCRIG": AppendMCRIG(index_nir=1, index_green=3, index_blue=4),  # Do not use - produces nan and/or inf vals
    "MVI": AppendMVI(index_swir=0, index_nir=1),
    # "MCARI": AppendMCARI(index_nir=1, index_red=2, index_green=3),  # Do not use - produces nan and/or inf vals
    # "MSRNirRed": AppendMSRNirRed(index_nir=1, index_red=2),  # Do not use - produces nan and/or inf vals
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
    "CHLA": AppendCHLA(index_green=3, index_blue=4),
    "CYA": AppendCYA(index_red=2, index_green=3, index_blue=4),
    "TURB": AppendTURB(index_green=3, index_blue=4),
    "CDOM": AppendCDOM(index_red=2, index_green=3),
    "DOC": AppendDOC(index_red=2, index_green=3),
    "WATERCOLOR": AppendWaterColor(index_red=2, index_green=3),
}

SPECTRAL_INDEX_LOOKUP = {
    "ATSAVI": AppendATSAVI,
    "AFRI1600": AppendAFRI1600,
    "AVI": AppendAVI,
    "ARVI": AppendARVI,
    "BWDRVI": AppendBWDRV,
    "ClGreen": AppendClGreen,
    "CVI": AppendCVI,
    "CI": AppendCI,
    "GDVI": AppendGDVI,
    "DVIMSS": AppendDVIMSS,
    "EVI": AppendEVI,
    "EVI2": AppendEVI2,
    "EVI22": AppendEVI22,
    "GARI": AppendGARI,
    "GNDVI": AppendGNDVI,
    "GRNDVI": AppendGRNDVI,
    "GBNDVI": AppendGBNDVI,
    "GVMI": AppendGVMI,
    # "IPVI": AppendIPVI,  # Do not use - produces nan and/or inf vals
    "I": AppendI,
    "H": AppendH,
    "LogR": AppendLogR,
    # "mCRIG": AppendMCRIG,  # Do not use - produces nan and/or inf vals
    "MVI": AppendMVI,
    # "MCARI": AppendMCARI,  # Do not use - produces nan and/or inf vals
    # "MSRNirRed": AppendMSRNirRed,  # Do not use - produces nan and/or inf vals
    "MSAVI": AppendMSAVI,
    "NLI": AppendNLI,
    "NDVI": AppendNDVI,
    "NDVIWM": AppendNDVIWM,
    "NDWI": AppendNDWI,
    "NDWIWM": AppendNDWIWM,
    "NormG": AppendNormG,
    "NormNIR": AppendNormNIR,
    "NormR": AppendNormR,
    "PNDVI": AppendPNDVI,
    "SRGR": AppendSRGR,
    "SRNIRG": AppendSRNIRG,
    "SRNIRR": AppendSRNIRR,
    "SRNIRSWIR": AppendSRNIRSWIR,
    "SRSWIRNIR": AppendSRSWIRNIR,
    "RBNDVI": AppendRBNDVI,
    "SQRTNIRR": AppendSQRTNIRR,
    "TNDVI": AppendTNDVI,
    "TVI": AppendTVI,
    "VARIGreen": AppendVARIGreen,
    "WDRVI": AppendWDRVI,
    "DEMWM": AppendDEMWM,
    "CHLA": AppendCHLA,
    "CYA": AppendCYA,
    "TURB": AppendTURB,
    "CDOM": AppendCDOM,
    "DOC": AppendDOC,
    "WATERCOLOR": AppendWaterColor,
}
