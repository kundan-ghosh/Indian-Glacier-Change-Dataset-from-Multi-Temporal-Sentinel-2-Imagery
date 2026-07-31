"""Sentinel-2 acquisition and cloud-free compositing using Earth Engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from tqdm.auto import tqdm

from igcd.config import IGCDConfig
from igcd.ee_utils import (
    active_task_count,
    active_task_descriptions,
    require_ee,
    start_drive_export,
)
from igcd.spectral import add_ee_indices
from igcd.utils import file_exists_and_nonempty


@dataclass(frozen=True)
class AcquisitionRecord:
    """Manifest row for a requested Sentinel-2 composite export."""

    glacier_id: str
    year: int
    status: str
    task_id: str | None
    image_count: int
    output_prefix: str
    message: str = ""


def date_window(config: IGCDConfig, year: int) -> tuple[str, str]:
    """Return the configured seasonal acquisition window for a year."""

    temporal = config.raw["temporal"]
    start = f"{year:04d}-{int(temporal['start_month']):02d}-"
    start += f"{int(temporal['start_day']):02d}"
    end = f"{year:04d}-{int(temporal['end_month']):02d}-"
    end += f"{int(temporal['end_day']):02d}"
    return start, end


def mask_sentinel_clouds(image: Any, config: IGCDConfig) -> Any:
    """Mask Sentinel-2 cloud, shadow, saturated, and snow-coded pixels."""

    invalid_values = config.raw["sentinel"]["invalid_scl_values"]
    scl = image.select("SCL")
    valid = None
    for value in invalid_values:
        current = scl.neq(int(value))
        valid = current if valid is None else valid.And(current)
    return image.updateMask(valid)


def sentinel_collection_for_roi(
    roi: Any,
    year: int,
    config: IGCDConfig,
) -> Any:
    """Build a filtered Sentinel-2 SR Harmonized collection for an ROI."""

    ee = require_ee()
    start, end = date_window(config, year)
    sentinel_cfg = config.raw["sentinel"]
    collection = (
        ee.ImageCollection(config.raw["earth_engine"]["sentinel_collection"])
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(
            ee.Filter.lte(
                sentinel_cfg["cloud_cover_property"],
                sentinel_cfg["max_cloud_percent"],
            )
        )
        .sort(sentinel_cfg["cloud_cover_property"])
        .limit(int(sentinel_cfg["max_images_per_composite"]))
        .map(lambda image: mask_sentinel_clouds(image, config))
        .select(sentinel_cfg["bands"])
    )
    return collection


def build_cloud_free_composite(roi: Any, year: int, config: IGCDConfig) -> Any:
    """Create a cloud-masked seasonal Sentinel-2 composite."""

    collection = sentinel_collection_for_roi(roi, year, config)
    reducer = config.raw["sentinel"]["composite_reducer"]
    if reducer == "median":
        image = collection.median()
    elif reducer == "mosaic":
        image = collection.mosaic()
    else:
        raise ValueError(f"Unsupported composite reducer: {reducer}")
    image = add_ee_indices(image.clip(roi))
    dtype = str(config.raw["export"].get("sentinel_dtype", "float32")).lower()
    if dtype in {"float32", "float"}:
        return image.toFloat()
    if dtype in {"int16", "short"}:
        return image.toInt16()
    raise ValueError(f"Unsupported export.sentinel_dtype: {dtype}")


def export_sentinel_for_inventory(
    inventory: pd.DataFrame,
    config: IGCDConfig,
    logger: logging.Logger,
    manifest_path: str | Path,
    years: Sequence[int] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Request Sentinel composite exports for all glaciers and configured years."""

    ee = require_ee()
    records: list[AcquisitionRecord] = []
    export_cfg = config.raw["export"]
    buffer_m = int(config.raw["sentinel"]["roi_buffer_m"])
    selected_years = [int(year) for year in (years or config.years)]
    task_limit = max_tasks
    if task_limit is None:
        task_limit = export_cfg.get("max_tasks_per_run")
    max_queued_tasks = int(export_cfg.get("max_queued_tasks", 3000))
    flush_interval = int(export_cfg.get("manifest_flush_interval", 25))
    check_existing_queue = bool(export_cfg.get("check_existing_queue", False))
    active_descriptions: set[str] = set()
    if check_existing_queue:
        active_descriptions = active_task_descriptions()
        active_at_start = active_task_count()
        available_queue_slots = max(0, max_queued_tasks - active_at_start)
        if task_limit is None:
            task_limit = available_queue_slots
        else:
            task_limit = min(int(task_limit), available_queue_slots)
        logger.info(
            "Earth Engine active queue slots: %s/%s; Sentinel exports allowed: %s",
            active_at_start,
            max_queued_tasks,
            task_limit,
        )
    elif task_limit is None:
        task_limit = max_queued_tasks
    if task_limit <= 0:
        logger.warning("No Earth Engine queue capacity available for Sentinel exports.")
        manifest = pd.DataFrame([record.__dict__ for record in records])
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        return manifest
    submitted_tasks = 0
    rows = inventory.to_dict("records")
    for year in selected_years:
        for row in tqdm(
            rows,
            desc=f"Requesting Sentinel exports {year}",
            unit="glacier",
        ):
            if task_limit is not None and submitted_tasks >= int(task_limit):
                logger.warning(
                    "Reached Sentinel export task limit for this run: %s",
                    task_limit,
                )
                manifest = pd.DataFrame([record.__dict__ for record in records])
                Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
                manifest.to_csv(manifest_path, index=False)
                return manifest
            glacier_id = str(row["glacier_id"])
            geometry = ee.Geometry(row["geometry"]).buffer(buffer_m)
            output_prefix = f"{glacier_id}_{year}_sentinel"
            description = f"IGCD Sentinel {glacier_id} {year}"
            expected = Path(config.paths["exports"]) / str(year)
            expected = expected / f"{output_prefix}.tif"
            if export_cfg["skip_existing"] and file_exists_and_nonempty(expected):
                records.append(
                    AcquisitionRecord(
                        glacier_id=glacier_id,
                        year=year,
                        status="skipped_existing",
                        task_id=None,
                        image_count=-1,
                        output_prefix=output_prefix,
                    )
                )
                continue
            if description in active_descriptions:
                records.append(
                    AcquisitionRecord(
                        glacier_id=glacier_id,
                        year=year,
                        status="skipped_active_task",
                        task_id=None,
                        image_count=-1,
                        output_prefix=output_prefix,
                    )
                )
                continue
            collection = sentinel_collection_for_roi(geometry, year, config)
            image_count = int(collection.size().getInfo())
            if image_count == 0:
                records.append(
                    AcquisitionRecord(
                        glacier_id=glacier_id,
                        year=year,
                        status="no_images",
                        task_id=None,
                        image_count=0,
                        output_prefix=output_prefix,
                    )
                )
                continue
            image = build_cloud_free_composite(geometry, year, config)
            try:
                task = start_drive_export(
                    image=image,
                    description=description,
                    folder=export_cfg["drive_folder"],
                    file_name_prefix=output_prefix,
                    region=geometry,
                    scale=int(export_cfg["scale_m"]),
                    crs=export_cfg["crs"],
                    max_pixels=int(export_cfg["max_pixels"]),
                )
            except Exception as exc:
                message = str(exc)
                records.append(
                    AcquisitionRecord(
                        glacier_id=glacier_id,
                        year=year,
                        status="export_start_failed",
                        task_id=None,
                        image_count=image_count,
                        output_prefix=output_prefix,
                        message=message,
                    )
                )
                if "Too many tasks already in the queue" in message:
                    logger.warning(
                        "Earth Engine queue is full; stopping Sentinel "
                        "submission after %s tasks.",
                        submitted_tasks,
                    )
                    manifest = pd.DataFrame(
                        [record.__dict__ for record in records]
                    )
                    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
                    manifest.to_csv(manifest_path, index=False)
                    return manifest
                raise
            task_id = task.status().get("id")
            submitted_tasks += 1
            active_descriptions.add(description)
            logger.info(
                "Started Sentinel export %s year=%s task=%s",
                glacier_id,
                year,
                task_id,
            )
            records.append(
                AcquisitionRecord(
                    glacier_id=glacier_id,
                    year=year,
                    status="submitted",
                    task_id=task_id,
                    image_count=image_count,
                    output_prefix=output_prefix,
                )
            )
            if flush_interval > 0 and len(records) % flush_interval == 0:
                manifest = pd.DataFrame([record.__dict__ for record in records])
                Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
                manifest.to_csv(manifest_path, index=False)
    manifest = pd.DataFrame([record.__dict__ for record in records])
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest
