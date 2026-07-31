"""Copernicus GLO-30 DEM helpers."""

from __future__ import annotations

from typing import Any

from igcd.config import IGCDConfig
from igcd.ee_utils import require_ee


def dem_image(config: IGCDConfig, roi: Any) -> Any:
    """Return a clipped Copernicus GLO-30 DEM image."""

    ee = require_ee()
    collection = ee.ImageCollection(config.raw["earth_engine"]["dem_collection"])
    band = config.raw["dem"]["elevation_band"]
    native_projection = collection.first().select(band).projection()
    return (
        collection.select(band)
        .mosaic()
        .setDefaultProjection(native_projection)
        .clip(roi)
    )


def terrain_features(config: IGCDConfig, roi: Any) -> Any:
    """Return elevation and slope bands clipped to an ROI."""

    ee = require_ee()
    elevation = dem_image(config, roi).rename("elevation")
    slope = ee.Terrain.slope(elevation).rename("slope")
    return elevation.addBands(slope)
