"""Spectral index calculations for glacier delineation."""

from __future__ import annotations

import numpy as np


def normalized_difference(
    band_a: np.ndarray,
    band_b: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute a normalized difference index using floating-point arrays."""

    a = band_a.astype("float32", copy=False)
    b = band_b.astype("float32", copy=False)
    return (a - b) / (a + b + eps)


def ndsi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Snow Index from green and SWIR bands."""

    return normalized_difference(green, swir)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Water Index from green and NIR bands."""

    return normalized_difference(green, nir)


def add_ee_indices(image: object) -> object:
    """Add NDSI and NDWI bands to an Earth Engine Sentinel-2 image.

    The function accepts ``object`` so this module remains importable before
    Earth Engine is installed. At runtime ``image`` must be an ``ee.Image``.
    """

    ndsi_img = image.normalizedDifference(["B3", "B11"]).rename("NDSI")
    ndwi_img = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands([ndsi_img, ndwi_img])
