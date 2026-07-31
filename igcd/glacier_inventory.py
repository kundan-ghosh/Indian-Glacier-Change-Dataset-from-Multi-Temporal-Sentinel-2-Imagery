"""GLIMS glacier inventory loading, cleaning, and export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
from tqdm.auto import tqdm

from igcd.config import IGCDConfig
from igcd.ee_utils import bbox_geometry, require_ee


def load_glims_ee(config: IGCDConfig) -> Any:
    """Load the configured GLIMS inventory as an Earth Engine collection."""

    ee = require_ee()
    collection = ee.FeatureCollection(config.raw["earth_engine"]["glims_asset"])
    return collection.filterBounds(bbox_geometry(config.raw["study_area"]["bbox"]))


def ee_feature_collection_to_geodataframe(
    collection: Any,
    batch_size: int = 1000,
    max_features: int | None = None,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Download an Earth Engine feature collection into GeoPandas in pages.

    Earth Engine aborts a single ``FeatureCollection.getInfo()`` call once the
    response accumulates too many elements. Paging through ``toList`` keeps each
    request below that limit while preserving a simple GeoPandas output for the
    inventory-cleaning stage.
    """

    if batch_size <= 0 or batch_size >= 5000:
        raise ValueError("batch_size must be between 1 and 4999.")

    total = int(collection.size().getInfo())
    if max_features is not None:
        total = min(total, int(max_features))
    if logger:
        logger.info("GLIMS collection contains %s features to download", total)

    features: list[dict[str, Any]] = []
    for offset in tqdm(
        range(0, total, batch_size),
        desc="Downloading GLIMS batches",
        unit="batch",
    ):
        count = min(batch_size, total - offset)
        batch = collection.toList(count, offset).getInfo()
        features.extend(batch)

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def repair_geometry(geometry: BaseGeometry) -> BaseGeometry | None:
    """Repair invalid geometries using ``make_valid`` when available."""

    if geometry is None or geometry.is_empty:
        return None
    if geometry.is_valid:
        return geometry
    try:
        from shapely.validation import make_valid

        repaired = make_valid(geometry)
    except Exception:
        repaired = geometry.buffer(0)
    if repaired is None or repaired.is_empty:
        return None
    return repaired


def clean_inventory(
    gdf: gpd.GeoDataFrame,
    config: IGCDConfig,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """Validate, deduplicate, enrich, and identify GLIMS glacier polygons."""

    if gdf.empty:
        raise ValueError("GLIMS inventory is empty after study-area filtering.")
    cleaned = gdf.copy()
    cleaned = cleaned.set_geometry("geometry")
    cleaned["geometry"] = cleaned.geometry.apply(repair_geometry)
    cleaned = cleaned[cleaned.geometry.notna() & ~cleaned.geometry.is_empty]
    cleaned = cleaned.drop_duplicates(subset=["geometry"]).reset_index(drop=True)

    metric_crs = config.raw["export"]["crs"]
    metric = cleaned.to_crs(metric_crs)
    cleaned["area_m2"] = metric.geometry.area.astype(float)
    metric_centroids = metric.geometry.centroid
    centroids = gpd.GeoSeries(metric_centroids, crs=metric_crs).to_crs(
        "EPSG:4326"
    )
    cleaned["centroid_lon"] = centroids.x.to_numpy()
    cleaned["centroid_lat"] = centroids.y.to_numpy()
    cleaned = cleaned.sort_values(["centroid_lon", "centroid_lat", "area_m2"])
    cleaned = cleaned.reset_index(drop=True)
    cleaned.insert(
        0,
        "glacier_id",
        [f"IGCD_{idx + 1:06d}" for idx in range(len(cleaned))],
    )
    if logger:
        logger.info("Prepared %s glacier inventory records", len(cleaned))
    return cleaned


def export_inventory(gdf: gpd.GeoDataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Export inventory as CSV, GeoPackage, GeoJSON, and statistics JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "glacier_inventory.csv"
    gpkg_path = output / "glacier_inventory.gpkg"
    geojson_path = output / "glacier_inventory.geojson"
    stats_path = output / "glacier_inventory_stats.json"

    table = pd.DataFrame(gdf.drop(columns="geometry"))
    table.to_csv(csv_path, index=False)
    gdf.to_file(gpkg_path, driver="GPKG")
    gdf.to_file(geojson_path, driver="GeoJSON")
    stats = inventory_statistics(gdf)
    stats_path.write_text(pd.Series(stats).to_json(indent=2), encoding="utf-8")
    return {
        "csv": csv_path,
        "gpkg": gpkg_path,
        "geojson": geojson_path,
        "stats": stats_path,
    }


def inventory_statistics(gdf: gpd.GeoDataFrame) -> dict[str, float | int]:
    """Compute publication-oriented inventory summary statistics."""

    areas = gdf["area_m2"].astype(float)
    return {
        "glacier_count": int(len(gdf)),
        "total_area_km2": float(areas.sum() / 1_000_000),
        "mean_area_km2": float(areas.mean() / 1_000_000),
        "median_area_km2": float(areas.median() / 1_000_000),
        "min_area_km2": float(areas.min() / 1_000_000),
        "max_area_km2": float(areas.max() / 1_000_000),
    }


def prepare_inventory_from_ee(
    config: IGCDConfig,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Load GLIMS from Earth Engine and return a cleaned GeoDataFrame."""

    collection = load_glims_ee(config)
    logger.info("Downloading GLIMS features from Earth Engine")
    inventory_cfg = config.raw.get("inventory", {})
    gdf = ee_feature_collection_to_geodataframe(
        collection,
        batch_size=int(inventory_cfg.get("download_batch_size", 1000)),
        max_features=inventory_cfg.get("max_features"),
        logger=logger,
    )
    for _ in tqdm(range(1), desc="Cleaning inventory"):
        return clean_inventory(gdf, config, logger)
    raise RuntimeError("Inventory cleaning did not complete.")
