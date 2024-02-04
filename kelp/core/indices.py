# mypy: disable-error-code="override"
from __future__ import annotations

import abc
from typing import Any, Dict

import torch
from torch import Tensor, nn

from kelp import consts


class AppendIndex(nn.Module, abc.ABC):
    """
    Base class for appending spectral indices to the input Tensor.
    """

    def __init__(
        self,
        index_qa: int = 5,
        index_water_mask: int = 7,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **band_kwargs: Any,
    ) -> None:
        super().__init__()
        assert all(k.startswith("index_") for k in band_kwargs), (
            "Passed keyword arguments must start with 'index_' followed by band name! "
            f"Found following keys: {list(band_kwargs.keys())}"
        )
        self.dim = -3
        self.index_qa = index_qa
        self.index_water_mask = index_water_mask
        self.mask_using_qa = mask_using_qa
        self.mask_using_water_mask = mask_using_water_mask
        self.fill_val = fill_val
        self.band_kwargs = band_kwargs

    def _append_index(self, sample: Tensor, index: Tensor) -> Tensor:
        index = index.unsqueeze(self.dim)
        sample = torch.cat([sample, index], dim=self.dim)
        return sample

    def _mask_index(self, index: Tensor, sample: Tensor, masking_band_index: int, apply_mask: bool) -> Tensor:
        if not apply_mask:
            return index
        mask = sample[..., masking_band_index, :, :]
        index = torch.where(mask == 0, index, self.fill_val)
        return index

    @abc.abstractmethod
    def _compute_index(self, **kwargs: Any) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        # Compute the index
        compute_kwargs: Dict[str, Tensor] = {
            k.replace("index_", ""): x[..., v, :, :] for k, v in self.band_kwargs.items()
        }
        index = self._compute_index(**compute_kwargs)

        # Mask using QA band if requested
        index = self._mask_index(
            index=index,
            sample=x,
            masking_band_index=self.index_qa,
            apply_mask=self.mask_using_qa,
        )

        # Mask using Water Mask if requested
        index = self._mask_index(
            index=index,
            sample=x,
            masking_band_index=self.index_water_mask,
            apply_mask=self.mask_using_water_mask,
        )

        # Append to the input tensor
        x = self._append_index(sample=x, index=index)

        return x


class AppendNDVI(AppendIndex):
    """
    Normalized Difference Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (nir - red) / (nir + red + consts.data.EPS)


class AppendNDWI(AppendIndex):
    """
    Normalized Difference Water Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return (nir - green) / (nir + green + consts.data.EPS)


class AppendATSAVI(AppendIndex):
    """
    Adjusted transformed soil-adjusted VI

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
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
    """
    Aerosol free vegetation index 1600

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir - 0.66 * (swir / (nir + 0.66 * swir + consts.data.EPS))


class AppendAVI(AppendIndex):
    """
    Ashburn Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2 * nir - red


class AppendARVI(AppendIndex):
    """
    Atmospherically Resistant Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return -0.18 + 1.17 * ((nir - red) / (nir + red + consts.data.EPS))


class AppendBWDRVI(AppendIndex):
    """
    Blue-wide dynamic range vegetation index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, blue: Tensor) -> Tensor:
        return (0.1 * nir - blue) / (0.1 * nir + blue + consts.data.EPS)


class AppendClGreen(AppendIndex):
    """
    Chlorophyll Index Green

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / (green + consts.data.EPS) - 1


class AppendCVI(AppendIndex):
    """
    Chlorophyll vegetation index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir * (red / (green**2 + consts.data.EPS))


class AppendDEMWM(AppendIndex):
    """
    DEM Water Mask

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_dem: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_dem=index_dem,
        )

    def _compute_index(self, dem: Tensor) -> Tensor:
        return torch.maximum(dem, torch.zeros_like(dem))


class AppendWDRVI(AppendIndex):
    """
    Wide Dynamic Range Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (0.1 * nir - red) / (0.1 * nir + red + consts.data.EPS)


class AppendVARIGreen(AppendIndex):
    """
    Visible Atmospherically Resistant Index Green

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (green - red) / (green + red - blue + consts.data.EPS)


class AppendTVI(AppendIndex):
    """
    Transformed Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return torch.sqrt(((red - green) / (red + green + consts.data.EPS)) + 0.5)


class AppendTNDVI(AppendIndex):
    """
    Transformed NDVI

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt((nir - red) / (nir + red + consts.data.EPS) + 0.5)


class AppendSQRTNIRR(AppendIndex):
    """
    SQRT(IR/R)

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.sqrt(nir / (red + consts.data.EPS))


class AppendRBNDVI(AppendIndex):
    """
    Red-Blue NDVI

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + blue)) / (nir + red + blue + consts.data.EPS)


class AppendSRSWIRNIR(AppendIndex):
    """
    Simple Ratio SWIR/NIR

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return swir / (nir + consts.data.EPS)


class AppendSRNIRSWIR(AppendIndex):
    """
    Simple Ratio NIR/SWIR

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir / (swir + consts.data.EPS)


class AppendSRNIRR(AppendIndex):
    """
    Simple Ratio NIR/Red

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return nir / (red + consts.data.EPS)


class AppendSRNIRG(AppendIndex):
    """
    Simple Ratio NIR/Green

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir / (green + consts.data.EPS)


class AppendSRGR(AppendIndex):
    """
    Simple Ratio NIR/Blue

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return green / (red + consts.data.EPS)


class AppendPNDVI(AppendIndex):
    """
        Pan NDVI

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (red + green + blue)) / (nir + red + green + blue + consts.data.EPS)


class AppendNormR(AppendIndex):
    """
    Norm R

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return red / (nir + red + green + consts.data.EPS)


class AppendNormNIR(AppendIndex):
    """
    Norm NIR

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return nir / (nir + red + green + consts.data.EPS)


class AppendNormG(AppendIndex):
    """
    Norm Green

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return green / (nir + red + green + consts.data.EPS)


class AppendNDWIWM(AppendIndex):
    """
    NDWI Water Mask

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return torch.maximum((nir - green) / (nir + green + consts.data.EPS), torch.zeros_like(nir))


class AppendNDVIWM(AppendIndex):
    """
    NDVI Water Mask

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.maximum((nir - red) / (nir + red + consts.data.EPS), torch.zeros_like(nir))


class AppendNLI(AppendIndex):
    """
    Nonlinear vegetation index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (nir**2 - red) / (nir**2 + red + consts.data.EPS)


class AppendMSAVI(AppendIndex):
    """
    Modified Soil Adjusted Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return (2 * nir + 1 - torch.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2


class AppendMSRNirRed(AppendIndex):
    """
    Modified Simple Ratio NIR/RED

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return ((nir / red + consts.data.EPS) - 1) / torch.sqrt((nir / (red + consts.data.EPS)) + 1)


class AppendMCARI(AppendIndex):
    """
    Modified Chlorophyll Absorption in Reflectance Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return ((nir - red) - 0.2 * (nir - green)) * (nir / (red + consts.data.EPS))


class AppendMVI(AppendIndex):
    """
    Mid-infrared vegetation index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return nir / (swir + consts.data.EPS)


class AppendMCRIG(AppendIndex):
    """
    mCRIG

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (blue**-1 - green**-1) * nir


class AppendLogR(AppendIndex):
    """
    Log Ratio

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return torch.log(nir / (red + consts.data.EPS) + consts.data.EPS)


class AppendH(AppendIndex):
    """
    Hue

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return torch.arctan(((2 * red - green - blue) / 30.5) * (green - blue))


class AppendI(AppendIndex):
    """
    Intensity

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (1 / 30.5) * (red + green + blue + consts.data.EPS)


class AppendIPVI(AppendIndex):
    """
    Infrared percentage vegetation index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return (nir / (nir + red + consts.data.EPS) / 2) * (
            ((red - green) / (red + consts.data.EPS) + green) + 1 + consts.data.EPS
        )


class AppendGVMI(AppendIndex):
    """
    Global Vegetation Moisture Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_swir: int,
        index_nir: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_swir=index_swir,
            index_nir=index_nir,
        )

    def _compute_index(self, swir: Tensor, nir: Tensor) -> Tensor:
        return ((nir + 0.1) - (swir + 0.02)) / ((nir + 0.1) + (swir + 0.02) + consts.data.EPS)


class AppendGBNDVI(AppendIndex):
    """
    Green-Blue NDVI

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (green + blue)) / (nir + green + blue + consts.data.EPS)


class AppendGRNDVI(AppendIndex):
    """
    Green-Red NDVI

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor) -> Tensor:
        return (nir - (red + green)) / (nir + red + green + consts.data.EPS)


class AppendGNDVI(AppendIndex):
    """
    Green Normalized Difference Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return (nir - green) / (nir + green + consts.data.EPS)


class AppendGARI(AppendIndex):
    """
    Green atmospherically resistant vegetation index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - (green - (blue - red))) / (nir - (green + (blue - red)) + consts.data.EPS)


class AppendEVI22(AppendIndex):
    """
    Enhanced Vegetation Index 2 -2

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.5 * (nir - red) / (nir + 2.4 * red + 1 + consts.data.EPS)


class AppendEVI2(AppendIndex):
    """
    Enhanced Vegetation Index 2

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.4 * (nir - red) / (nir + red + 1 + consts.data.EPS)


class AppendEVI(AppendIndex):
    """
    Enhanced Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
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
    """
    Differenced Vegetation Index MSS

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
        )

    def _compute_index(self, nir: Tensor, red: Tensor) -> Tensor:
        return 2.4 * nir - red


class AppendGDVI(AppendIndex):
    """
    Difference NIR/Green Green Difference Vegetation Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_nir: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_green=index_green,
        )

    def _compute_index(self, nir: Tensor, green: Tensor) -> Tensor:
        return nir - green


class AppendCI(AppendIndex):
    """
    Coloration Index

    Source: https://www.indexdatabase.de/
    """

    def __init__(
        self,
        index_red: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, blue: Tensor) -> Tensor:
        return (red - blue) / (red + consts.data.EPS)


class AppendCHLA(AppendIndex):
    """
    Chlorophyll-a

    Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/
    """

    def __init__(
        self,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, blue: Tensor, green: Tensor) -> Tensor:
        # Yes, I know we should use coastal band here instead of blue, but we don't get to have coastal band
        blue = blue / 65_535
        green = green / 65_535
        return 4.23 * torch.pow((green / (blue + consts.data.EPS)), 3.94)


class AppendCYA(AppendIndex):
    """
    Cyanobacteria density

    Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        red = red / 65_535
        green = green / 65_535
        blue = blue / 65_535
        return torch.pow(((green * red + consts.data.EPS) / (blue + consts.data.EPS)), 2.38)  # * 115_530.31


class AppendTURB(AppendIndex):
    """
    Water Turbidity

    Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/
    """

    def __init__(
        self,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, green: Tensor, blue: Tensor) -> Tensor:
        return 8.93 * (green / (blue + consts.data.EPS)) - 6.39


class AppendCDOM(AppendIndex):
    """
    Colored Dissolved Organic Matter

    Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return 537 * torch.exp(-2.93 * green / (red + consts.data.EPS))


class AppendDOC(AppendIndex):
    """
    Dissolved Organic Carbon index

    Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return 432 * torch.exp(-2.24 * green / (red + consts.data.EPS))


class AppendWaterColor(AppendIndex):
    """
    Water color index

    Source: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/se2waq/
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
        )

    def _compute_index(self, red: Tensor, green: Tensor) -> Tensor:
        return 25366 * torch.exp(-4.53 * green / (red + consts.data.EPS))


class AppendSABI(AppendIndex):
    """
    Surface Algal Bloom Index

    Source: https://doi.org/10.1002/eap.1708
    """

    def __init__(
        self,
        index_nir: int,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (nir - red) / (green + blue + consts.data.EPS)


class AppendKIVU(AppendIndex):
    """
    KIVU

    Source: https://doi.org/10.1002/eap.1708
    """

    def __init__(
        self,
        index_red: int,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_red=index_red,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, red: Tensor, green: Tensor, blue: Tensor) -> Tensor:
        return (blue - red) / (green + consts.data.EPS)


class AppendKab1(AppendIndex):
    """
    Kabbara Index 1

    Source: https://doi.org/10.1002/eap.1708
    """

    def __init__(
        self,
        index_green: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_green=index_green,
            index_blue=index_blue,
        )

    def _compute_index(self, green: Tensor, blue: Tensor) -> Tensor:
        return 1.67 - 3.94 * torch.log(blue + consts.data.EPS) + 3.78 * torch.log(green + consts.data.EPS)


class AppendNDAVI(AppendIndex):
    """
    Normalized Difference Aquatic Vegetation Index

    Source: https://doi.org/10.1016/j.jag.2014.01.017
    """

    def __init__(
        self,
        index_nir: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, blue: Tensor) -> Tensor:
        return (nir - blue) / (nir + blue + consts.data.EPS)


class AppendWAVI(AppendIndex):
    """
    Water Adjusted Vegetation Index

    Source: https://doi.org/10.1016/j.jag.2014.01.017
    """

    def __init__(
        self,
        index_nir: int,
        index_blue: int,
        index_qa: int = 5,
        index_water_mask: int = 8,
        mask_using_qa: bool = False,
        mask_using_water_mask: bool = False,
        fill_val: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            index_qa=index_qa,
            index_water_mask=index_water_mask,
            mask_using_qa=mask_using_qa,
            mask_using_water_mask=mask_using_water_mask,
            fill_val=fill_val,
            index_nir=index_nir,
            index_blue=index_blue,
        )

    def _compute_index(self, nir: Tensor, blue: Tensor) -> Tensor:
        return 1.5 * (nir - blue) / (nir + blue + 0.5 + consts.data.EPS)


SPECTRAL_INDEX_LOOKUP = {
    "ATSAVI": AppendATSAVI,
    "AFRI1600": AppendAFRI1600,
    "AVI": AppendAVI,
    "ARVI": AppendARVI,
    "BWDRVI": AppendBWDRVI,
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
    "IPVI": AppendIPVI,
    "I": AppendI,
    "H": AppendH,
    "LogR": AppendLogR,
    "mCRIG": AppendMCRIG,  # Do not use - produces nan and/or inf vals
    "MVI": AppendMVI,
    "MCARI": AppendMCARI,  # Do not use - produces nan and/or inf vals
    "MSRNirRed": AppendMSRNirRed,  # Do not use - produces nan and/or inf vals
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
    "CHLA": AppendCHLA,  # Do not use - produces nan and/or inf vals
    "CYA": AppendCYA,  # Do not use - produces nan and/or inf vals
    "TURB": AppendTURB,
    "CDOM": AppendCDOM,
    "DOC": AppendDOC,
    "WATERCOLOR": AppendWaterColor,
    "SABI": AppendSABI,
    "KIVU": AppendKIVU,
    "Kab1": AppendKab1,
    "NDAVI": AppendNDAVI,
    "WAVI": AppendWAVI,
}
ALL_INDICES = list(SPECTRAL_INDEX_LOOKUP.keys())
BASE_BANDS = list(consts.data.ORIGINAL_BANDS) + ["DEMWM"]
BAND_INDEX_LOOKUP = {band_name: index for index, band_name in enumerate(BASE_BANDS)}
