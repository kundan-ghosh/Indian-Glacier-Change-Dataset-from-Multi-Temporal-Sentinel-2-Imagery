"""Interactive GeoTIFF preview helpers for notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


StretchMode = Literal["percentile", "minmax", "stddev"]


@dataclass(frozen=True)
class RasterPreview:
    """Downsampled raster data and display metadata."""

    path: Path
    data: np.ndarray
    metadata: dict[str, Any]
    band_labels: list[str]

    @property
    def band_count(self) -> int:
        return int(self.data.shape[0])


def open_raster_preview(
    path: str | Path,
    max_size: int = 1400,
    indexes: list[int] | None = None,
) -> RasterPreview:
    """Read a downsampled preview from a GeoTIFF without loading full scenes."""

    import rasterio

    raster_path = Path(path).expanduser()
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        scale = min(1.0, max_size / max(src.width, src.height))
        out_width = max(1, int(src.width * scale))
        out_height = max(1, int(src.height * scale))
        selected = indexes or list(range(1, src.count + 1))
        data = src.read(
            indexes=selected,
            out_shape=(len(selected), out_height, out_width),
            masked=True,
        )
        descriptions = list(src.descriptions or [])
        labels = [
            _band_label(
                index,
                descriptions[index - 1] if index - 1 < len(descriptions) else None,
            )
            for index in selected
        ]
        metadata = {
            "path": str(raster_path),
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "preview_width": out_width,
            "preview_height": out_height,
            "count": src.count,
            "crs": str(src.crs) if src.crs else None,
            "transform": tuple(src.transform),
            "bounds": tuple(src.bounds),
            "dtypes": list(src.dtypes),
            "nodata": src.nodata,
        }
    return RasterPreview(
        path=raster_path,
        data=np.ma.filled(data, np.nan).astype("float32"),
        metadata=metadata,
        band_labels=labels,
    )


def _band_label(index: int, description: str | None) -> str:
    """Return a compact band label for widgets."""

    if description:
        return f"{index}: {description}"
    return f"Band {index}"


def stretch_band(
    band: np.ndarray,
    mode: StretchMode = "percentile",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Normalize a raster band to 0-1 for display."""

    values = band.astype("float32", copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype="float32")

    if mode == "percentile":
        low, high = np.nanpercentile(
            finite,
            [lower_percentile, upper_percentile],
        )
    elif mode == "minmax":
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    elif mode == "stddev":
        mean = float(np.nanmean(finite))
        std = float(np.nanstd(finite))
        low, high = mean - 2 * std, mean + 2 * std
    else:
        raise ValueError(f"Unsupported stretch mode: {mode}")

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype="float32")
    normalized = np.clip((values - low) / (high - low), 0, 1)
    if gamma <= 0:
        gamma = 1.0
    return np.power(normalized, 1.0 / gamma).astype("float32")


def rgb_composite(
    preview: RasterPreview,
    red_band: int,
    green_band: int,
    blue_band: int,
    mode: StretchMode = "percentile",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Create an RGB display array from selected 1-based band numbers."""

    channels = []
    for band_index in (red_band, green_band, blue_band):
        channels.append(
            stretch_band(
                preview.data[band_index - 1],
                mode=mode,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                gamma=gamma,
            )
        )
    return np.dstack(channels)


def single_band_display(
    preview: RasterPreview,
    band: int,
    mode: StretchMode = "percentile",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Return a normalized single-band display array."""

    return stretch_band(
        preview.data[band - 1],
        mode=mode,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        gamma=gamma,
    )


def overlay_mask(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.1, 0.1),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a boolean mask over an RGB image."""

    output = np.array(base_rgb, dtype="float32", copy=True)
    valid = np.asarray(mask).astype(bool)
    overlay_color = np.array(color, dtype="float32")
    output[valid] = (1 - alpha) * output[valid] + alpha * overlay_color
    return np.clip(output, 0, 1)


def threshold_mask(
    preview: RasterPreview,
    band: int,
    threshold: float,
    direction: Literal["greater_equal", "less_equal"] = "greater_equal",
) -> np.ndarray:
    """Create a boolean threshold mask from one preview band."""

    values = preview.data[band - 1]
    if direction == "greater_equal":
        return values >= threshold
    if direction == "less_equal":
        return values <= threshold
    raise ValueError(f"Unsupported threshold direction: {direction}")


def band_statistics(preview: RasterPreview) -> list[dict[str, float | int | str]]:
    """Compute robust per-band statistics from preview data."""

    stats: list[dict[str, float | int | str]] = []
    for idx, label in enumerate(preview.band_labels, start=1):
        band = preview.data[idx - 1]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            stats.append({"band": idx, "label": label, "valid_pixels": 0})
            continue
        stats.append(
            {
                "band": idx,
                "label": label,
                "valid_pixels": int(finite.size),
                "min": float(np.nanmin(finite)),
                "p2": float(np.nanpercentile(finite, 2)),
                "mean": float(np.nanmean(finite)),
                "p98": float(np.nanpercentile(finite, 98)),
                "max": float(np.nanmax(finite)),
            }
        )
    return stats


def default_rgb_bands(band_count: int) -> tuple[int, int, int]:
    """Choose sensible default RGB bands for common optical rasters."""

    if band_count >= 4:
        return 4, 3, 2
    if band_count >= 3:
        return 3, 2, 1
    return 1, 1, 1
