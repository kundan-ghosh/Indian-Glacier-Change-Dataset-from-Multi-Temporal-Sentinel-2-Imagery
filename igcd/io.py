"""Raster and tabular I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_raster(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a raster and return array plus profile."""

    import rasterio

    with rasterio.open(path) as src:
        return src.read(), dict(src.profile)


def write_single_band_raster(
    array: np.ndarray,
    reference_profile: dict[str, Any],
    path: str | Path,
    dtype: str = "uint8",
    nodata: int | float | None = 0,
) -> Path:
    """Write a single-band raster aligned to a reference profile."""

    import rasterio

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(reference_profile)
    profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=2 if dtype.startswith("float") else 1,
    )
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    return output


def assert_raster_alignment(reference: str | Path, candidate: str | Path) -> None:
    """Raise if two rasters do not share CRS, transform, width, and height."""

    import rasterio

    with rasterio.open(reference) as ref, rasterio.open(candidate) as cand:
        checks = {
            "crs": ref.crs == cand.crs,
            "transform": ref.transform == cand.transform,
            "width": ref.width == cand.width,
            "height": ref.height == cand.height,
        }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            f"Raster alignment failed for {candidate}: {', '.join(failed)}"
        )

