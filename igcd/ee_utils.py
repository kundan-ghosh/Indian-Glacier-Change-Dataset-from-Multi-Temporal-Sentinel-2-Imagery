"""Google Earth Engine initialization, validation, and export helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from igcd.config import IGCDConfig

ACTIVE_TASK_STATES = {"READY", "RUNNING"}


@dataclass(frozen=True)
class EEDatasetStatus:
    """Availability status for a configured Earth Engine dataset."""

    asset_id: str
    available: bool
    error: str | None = None


def require_ee() -> Any:
    """Import Earth Engine lazily with a clear error message."""

    try:
        import ee
    except ImportError as exc:
        raise RuntimeError(
            "earthengine-api is required. Install requirements.txt first."
        ) from exc
    return ee


def authenticate_and_initialize(
    project: str | None = None,
    force_authenticate: bool = False,
) -> None:
    """Authenticate and initialize Google Earth Engine."""

    ee = require_ee()
    if force_authenticate:
        ee.Authenticate()
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def validate_ee_access(config: IGCDConfig) -> list[EEDatasetStatus]:
    """Check whether configured Earth Engine assets are readable."""

    ee = require_ee()
    datasets = [
        config.raw["earth_engine"]["glims_asset"],
        config.raw["earth_engine"]["sentinel_collection"],
        config.raw["earth_engine"]["dem_collection"],
    ]
    statuses: list[EEDatasetStatus] = []
    for asset_id in datasets:
        try:
            if "/" in asset_id and asset_id != "GLIMS/current":
                ee.ImageCollection(asset_id).limit(1).size().getInfo()
            else:
                ee.FeatureCollection(asset_id).limit(1).size().getInfo()
            statuses.append(EEDatasetStatus(asset_id=asset_id, available=True))
        except Exception as exc:
            statuses.append(
                EEDatasetStatus(
                    asset_id=asset_id,
                    available=False,
                    error=str(exc),
                )
            )
    return statuses


def bbox_geometry(bbox: list[float]) -> Any:
    """Build an Earth Engine rectangle geometry from ``[xmin, ymin, xmax, ymax]``."""

    ee = require_ee()
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)


def start_drive_export(
    image: Any,
    description: str,
    folder: str,
    file_name_prefix: str,
    region: Any,
    scale: int,
    crs: str,
    max_pixels: int,
) -> Any:
    """Start a Google Drive image export task."""

    ee = require_ee()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def task_state_counts() -> dict[str, int]:
    """Return Earth Engine task counts grouped by state."""

    ee = require_ee()
    counts: dict[str, int] = {}
    for task in ee.batch.Task.list():
        state = str(task.status().get("state", "UNKNOWN"))
        counts[state] = counts.get(state, 0) + 1
    return counts


def active_task_count() -> int:
    """Return count of Earth Engine tasks occupying active queue slots."""

    counts = task_state_counts()
    return sum(counts.get(state, 0) for state in ACTIVE_TASK_STATES)


def active_task_descriptions() -> set[str]:
    """Return descriptions for Earth Engine tasks in active queue states."""

    ee = require_ee()
    descriptions: set[str] = set()
    for task in ee.batch.Task.list():
        status = task.status()
        if str(status.get("state", "UNKNOWN")) in ACTIVE_TASK_STATES:
            description = status.get("description")
            if description:
                descriptions.add(str(description))
    return descriptions


def task_state(task: Any) -> str:
    """Return the current Earth Engine task state."""

    return str(task.status().get("state", "UNKNOWN"))


def log_task(task: Any, logger: logging.Logger, label: str) -> None:
    """Write a concise Earth Engine task status line."""

    status = task.status()
    logger.info("%s | %s | %s", label, status.get("state"), status.get("id"))


def expected_export_path(
    export_dir: str | Path,
    glacier_id: str,
    year: int,
    suffix: str,
) -> Path:
    """Build a deterministic local export path used by manifests and QC."""

    return Path(export_dir) / str(year) / f"{glacier_id}_{year}_{suffix}.tif"
