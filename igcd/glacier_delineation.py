"""Automatic glacier delineation from Sentinel-2 and DEM features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from skimage.filters import threshold_otsu
except ImportError:  # pragma: no cover - exercised only in minimal runtimes.
    threshold_otsu = None

from igcd.config import IGCDConfig
from igcd.dem import terrain_features
from igcd.ee_utils import (
    active_task_count,
    active_task_descriptions,
    require_ee,
    start_drive_export,
)
from igcd.morphology import clean_binary_mask
from igcd.sentinel import build_cloud_free_composite, sentinel_collection_for_roi
from igcd.spectral import ndsi, ndwi
from igcd.utils import file_exists_and_nonempty


@dataclass(frozen=True)
class DelineationRecord:
    """Manifest row for a requested glacier-mask export."""

    glacier_id: str
    year: int
    status: str
    task_id: str | None
    image_count: int
    output_prefix: str
    message: str = ""


def otsu_or_fallback(
    values: np.ndarray,
    fallback: float,
    min_samples: int = 128,
) -> float:
    """Compute Otsu threshold for finite values, with a safe fallback."""

    valid = values[np.isfinite(values)]
    if valid.size < min_samples:
        return float(fallback)
    if threshold_otsu is not None:
        try:
            return float(threshold_otsu(valid))
        except ValueError:
            return float(fallback)
    return _otsu_numpy(valid, fallback)


def delineate_glacier_mask(
    green: np.ndarray,
    swir: np.ndarray,
    nir: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    prior_mask: np.ndarray,
    config: IGCDConfig,
) -> np.ndarray:
    """Create a cleaned binary glacier mask from local raster arrays."""

    del_cfg = config.raw["delineation"]
    dem_cfg = config.raw["dem"]
    morph_cfg = config.raw["morphology"]
    ndsi_values = ndsi(green, swir)
    ndwi_values = ndwi(green, nir)
    prior = np.asarray(prior_mask).astype(bool)
    threshold = otsu_or_fallback(
        ndsi_values[prior],
        fallback=float(del_cfg["fallback_ndsi_threshold"]),
        min_samples=int(del_cfg["otsu_min_samples"]),
    )
    candidate = (
        (ndsi_values >= threshold)
        & (ndwi_values <= float(del_cfg["fallback_ndwi_max"]))
        & (elevation >= float(dem_cfg["min_glacier_elevation_m"]))
        & (elevation <= float(dem_cfg["max_glacier_elevation_m"]))
        & (slope >= float(del_cfg["min_slope_deg"]))
        & (slope <= float(del_cfg["max_slope_deg"]))
        & prior
    )
    return clean_binary_mask(
        candidate,
        opening_radius=int(morph_cfg["opening_radius"]),
        closing_radius=int(morph_cfg["closing_radius"]),
        min_object_size=int(morph_cfg["min_object_size_pixels"]),
        min_component_area=int(morph_cfg["min_component_area_pixels"]),
        fill_holes=bool(morph_cfg["fill_holes"]),
    )


def build_ee_delineation_image(
    sentinel_image: Any,
    roi: Any,
    glims_prior: Any,
    config: IGCDConfig,
) -> Any:
    """Build an Earth Engine glacier-mask image using spectral and DEM rules."""

    del_cfg = config.raw["delineation"]
    dem_cfg = config.raw["dem"]
    terrain = terrain_features(config, roi)
    ndsi_img = sentinel_image.select("NDSI")
    ndwi_img = sentinel_image.select("NDWI")
    prior = glims_prior.gt(0).selfMask()
    mask = (
        ndsi_img.gte(float(del_cfg["fallback_ndsi_threshold"]))
        .And(ndwi_img.lte(float(del_cfg["fallback_ndwi_max"])))
        .And(
            terrain.select("elevation").gte(
                float(dem_cfg["min_glacier_elevation_m"])
            )
        )
        .And(
            terrain.select("elevation").lte(
                float(dem_cfg["max_glacier_elevation_m"])
            )
        )
        .And(terrain.select("slope").gte(float(del_cfg["min_slope_deg"])))
        .And(terrain.select("slope").lte(float(del_cfg["max_slope_deg"])))
        .And(prior)
    )
    return mask.rename("glacier_mask").uint8().clip(roi)


def build_glims_prior_mask(geometry: Any, roi: Any) -> Any:
    """Rasterize a glacier geometry as an Earth Engine spatial prior mask."""

    ee = require_ee()
    feature = ee.Feature(geometry)
    prior = ee.Image(0).byte().paint(ee.FeatureCollection([feature]), 1)
    return prior.rename("glims_prior").clip(roi)


def export_delineation_for_inventory(
    inventory: pd.DataFrame,
    config: IGCDConfig,
    logger: logging.Logger,
    manifest_path: str | Path,
    years: Sequence[int] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    """Request Earth Engine exports for year-specific glacier masks."""

    ee = require_ee()
    records: list[DelineationRecord] = []
    export_cfg = config.raw["export"]
    roi_buffer_m = int(config.raw["sentinel"]["roi_buffer_m"])
    prior_buffer_m = int(config.raw["delineation"]["prior_buffer_m"])
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
            "Earth Engine active queue slots: %s/%s; mask exports allowed: %s",
            active_at_start,
            max_queued_tasks,
            task_limit,
        )
    elif task_limit is None:
        task_limit = max_queued_tasks
    if task_limit <= 0:
        logger.warning("No Earth Engine queue capacity available for mask exports.")
        manifest = pd.DataFrame([record.__dict__ for record in records])
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        return manifest
    submitted_tasks = 0
    rows = inventory.to_dict("records")
    for year in selected_years:
        for row in tqdm(
            rows,
            desc=f"Requesting glacier-mask exports {year}",
            unit="glacier",
        ):
            if task_limit is not None and submitted_tasks >= int(task_limit):
                logger.warning(
                    "Reached mask export task limit for this run: %s",
                    task_limit,
                )
                manifest = pd.DataFrame([record.__dict__ for record in records])
                Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
                manifest.to_csv(manifest_path, index=False)
                return manifest
            glacier_id = str(row["glacier_id"])
            glacier_geometry = ee.Geometry(row["geometry"])
            roi = glacier_geometry.buffer(roi_buffer_m)
            prior_geometry = glacier_geometry.buffer(prior_buffer_m)
            prior = build_glims_prior_mask(prior_geometry, roi)
            output_prefix = f"{glacier_id}_{year}_mask"
            description = f"IGCD Mask {glacier_id} {year}"
            expected = mask_output_path(config.paths["processed"], glacier_id, year)
            if export_cfg["skip_existing"] and file_exists_and_nonempty(expected):
                records.append(
                    DelineationRecord(
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
                    DelineationRecord(
                        glacier_id=glacier_id,
                        year=year,
                        status="skipped_active_task",
                        task_id=None,
                        image_count=-1,
                        output_prefix=output_prefix,
                    )
                )
                continue
            collection = sentinel_collection_for_roi(roi, year, config)
            image_count = int(collection.size().getInfo())
            if image_count == 0:
                records.append(
                    DelineationRecord(
                        glacier_id=glacier_id,
                        year=year,
                        status="no_images",
                        task_id=None,
                        image_count=0,
                        output_prefix=output_prefix,
                    )
                )
                continue
            sentinel = build_cloud_free_composite(roi, year, config)
            mask = build_ee_delineation_image(sentinel, roi, prior, config)
            try:
                task = start_drive_export(
                    image=mask,
                    description=description,
                    folder=export_cfg["drive_folder"],
                    file_name_prefix=output_prefix,
                    region=roi,
                    scale=int(export_cfg["scale_m"]),
                    crs=export_cfg["crs"],
                    max_pixels=int(export_cfg["max_pixels"]),
                )
            except Exception as exc:
                message = str(exc)
                records.append(
                    DelineationRecord(
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
                        "Earth Engine queue is full; stopping mask "
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
                "Started mask export %s year=%s task=%s",
                glacier_id,
                year,
                task_id,
            )
            records.append(
                DelineationRecord(
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


def mask_output_path(base_dir: str | Path, glacier_id: str, year: int) -> Path:
    """Return the standard path for a yearly glacier mask."""

    return Path(base_dir) / "masks" / str(year) / f"{glacier_id}_{year}_mask.tif"


def _otsu_numpy(values: np.ndarray, fallback: float, bins: int = 256) -> float:
    """Compute an Otsu threshold using NumPy when scikit-image is unavailable."""

    if values.size == 0 or np.nanmin(values) == np.nanmax(values):
        return float(fallback)
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight_1 = np.cumsum(hist)
    weight_2 = np.cumsum(hist[::-1])[::-1]
    mean_1 = np.cumsum(hist * centers) / np.maximum(weight_1, 1)
    mean_2 = (
        np.cumsum((hist * centers)[::-1])
        / np.maximum(weight_2[::-1], 1)
    )[::-1]
    variance = weight_1[:-1] * weight_2[1:] * (mean_1[:-1] - mean_2[1:]) ** 2
    if variance.size == 0:
        return float(fallback)
    return float(centers[:-1][np.argmax(variance)])
