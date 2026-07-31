# """Tests for spectral index calculations."""

# from __future__ import annotations

# import numpy as np

# from igcd.spectral import ndsi, ndwi, normalized_difference


# def test_normalized_difference() -> None:
#     a = np.array([4.0, 2.0], dtype="float32")
#     b = np.array([2.0, 2.0], dtype="float32")
#     result = normalized_difference(a, b)
#     np.testing.assert_allclose(result, np.array([1 / 3, 0], dtype="float32"))


# def test_ndsi_and_ndwi_delegate_to_normalized_difference() -> None:
#     green = np.array([[8.0, 4.0]], dtype="float32")
#     swir = np.array([[2.0, 4.0]], dtype="float32")
#     nir = np.array([[6.0, 2.0]], dtype="float32")
#     np.testing.assert_allclose(ndsi(green, swir), [[0.6, 0.0]], atol=1e-6)
#     np.testing.assert_allclose(ndwi(green, nir), [[1 / 7, 1 / 3]], atol=1e-6)

"""Tests for spectral index calculations."""

from __future__ import annotations

import numpy as np

from igcd.spectral import ndsi, ndwi, normalized_difference


def test_normalized_difference() -> None:
    a = np.array([4.0, 2.0], dtype="float32")
    b = np.array([2.0, 2.0], dtype="float32")
    result = normalized_difference(a, b)
    np.testing.assert_allclose(
        result, np.array([1 / 3, 0], dtype="float32"), atol=1e-6
    )


def test_ndsi_and_ndwi_delegate_to_normalized_difference() -> None:
    green = np.array([[8.0, 4.0]], dtype="float32")
    swir = np.array([[2.0, 4.0]], dtype="float32")
    nir = np.array([[6.0, 2.0]], dtype="float32")
    np.testing.assert_allclose(ndsi(green, swir), [[0.6, 0.0]], atol=1e-6)
    np.testing.assert_allclose(ndwi(green, nir), [[1 / 7, 1 / 3]], atol=1e-6)