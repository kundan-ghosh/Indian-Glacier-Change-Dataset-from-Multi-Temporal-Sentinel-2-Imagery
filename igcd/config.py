"""Configuration models and validation for the IGCD pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IGCDConfig:
    """Runtime configuration loaded from ``config/config.json``."""

    raw: dict[str, Any]
    project_root: Path

    @property
    def dataset_name(self) -> str:
        return str(self.raw["dataset"]["name"])

    @property
    def years(self) -> list[int]:
        return [int(year) for year in self.raw["temporal"]["years"]]

    @property
    def baseline_year(self) -> int:
        return int(self.raw["temporal"]["baseline_year"])

    @property
    def target_year(self) -> int:
        return int(self.raw["temporal"]["target_year"])

    @property
    def export_crs(self) -> str:
        return str(self.raw["export"]["crs"])

    @property
    def export_scale(self) -> int:
        return int(self.raw["export"]["scale_m"])

    @property
    def paths(self) -> dict[str, Path]:
        return {
            key: self.project_root / value
            for key, value in self.raw["paths"].items()
        }


def load_config(path: str | Path = "config/config.json") -> IGCDConfig:
    """Load and validate an IGCD JSON configuration file.

    Parameters
    ----------
    path:
        Path to the project configuration JSON file.

    Returns
    -------
    IGCDConfig
        Validated configuration wrapper.
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    validate_config(raw)
    return IGCDConfig(raw=raw, project_root=config_path.parents[1])


def validate_config(config: dict[str, Any]) -> None:
    """Validate required configuration sections and common constraints."""

    required_sections = {
        "dataset",
        "earth_engine",
        "study_area",
        "temporal",
        "sentinel",
        "dem",
        "export",
        "delineation",
        "morphology",
        "quality_control",
        "splits",
        "paths",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    years = config["temporal"].get("years", [])
    if len(years) < 2:
        raise ValueError("At least two temporal.years are required.")
    if config["temporal"]["baseline_year"] not in years:
        raise ValueError("baseline_year must be present in temporal.years.")
    if config["temporal"]["target_year"] not in years:
        raise ValueError("target_year must be present in temporal.years.")

    split_sum = sum(float(v) for v in config["splits"].values())
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError("Dataset split ratios must sum to 1.0.")

    scale = int(config["export"]["scale_m"])
    if scale <= 0:
        raise ValueError("export.scale_m must be positive.")


def write_default_config(path: str | Path) -> Path:
    """Write a complete default IGCD configuration JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": {
            "name": "Indian Glacier Change Dataset",
            "short_name": "IGCD",
            "version": "0.1.0",
            "description": (
                "Reproducible glacier change dataset generated from GLIMS, "
                "Sentinel-2 L2A, and Copernicus GLO-30 DEM."
            ),
            "license": "CC-BY-4.0",
        },
        "earth_engine": {
            "project": "replace-with-google-cloud-project-id",
            "glims_asset": "GLIMS/current",
            "sentinel_collection": "COPERNICUS/S2_SR_HARMONIZED",
            "cloud_probability_collection": "COPERNICUS/S2_CLOUD_PROBABILITY",
            "dem_collection": "COPERNICUS/DEM/GLO30_2024_1",
        },
        "study_area": {
            "name": "Indian Himalayan Region",
            "bbox": [72.0, 26.0, 98.0, 38.0],
            "country_names": ["India"],
        },
        "temporal": {
            "years": [2018, 2023],
            "baseline_year": 2018,
            "target_year": 2023,
            "start_month": 7,
            "start_day": 1,
            "end_month": 10,
            "end_day": 15,
        },
        "sentinel": {
            "bands": ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"],
            "reflectance_bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "cloud_cover_property": "CLOUDY_PIXEL_PERCENTAGE",
            "max_cloud_percent": 35,
            "cloud_probability_threshold": 45,
            "invalid_scl_values": [0, 1, 3, 8, 9, 10, 11],
            "composite_reducer": "median",
            "roi_buffer_m": 1000,
            "max_images_per_composite": 80,
        },
        "inventory": {
            "download_batch_size": 1000,
            "max_features": None,
        },
        "dem": {
            "source": "COPERNICUS/DEM/GLO30_2024_1",
            "elevation_band": "DEM",
            "min_glacier_elevation_m": 2500,
            "max_glacier_elevation_m": 7500,
        },
        "export": {
            "check_existing_queue": False,
            "crs": "EPSG:32643",
            "scale_m": 10,
            "max_pixels": 1000000000,
            "drive_folder": "IGCD_exports",
            "file_format": "GeoTIFF",
            "manifest_flush_interval": 25,
            "max_queued_tasks": 3000,
            "max_tasks_per_run": 3000,
            "skip_existing": True,
            "sentinel_dtype": "float32",
        },
        "processing": {
            "max_glaciers_per_export_batch": 1500,
        },
        "delineation": {
            "ndsi_band": "NDSI",
            "ndwi_band": "NDWI",
            "adaptive_threshold": True,
            "otsu_min_samples": 128,
            "fallback_ndsi_threshold": 0.42,
            "fallback_ndwi_max": 0.35,
            "min_slope_deg": 2,
            "max_slope_deg": 55,
            "prior_buffer_m": 250,
        },
        "morphology": {
            "opening_radius": 1,
            "closing_radius": 2,
            "min_object_size_pixels": 9,
            "min_component_area_pixels": 16,
            "fill_holes": True,
        },
        "quality_control": {
            "expected_resolution_m": 10,
            "max_cloud_fraction": 0.15,
            "max_area_change_fraction": 0.6,
            "min_mask_pixels": 10,
        },
        "splits": {
            "train": 0.70,
            "validation": 0.10,
            "test": 0.20,
        },
        "paths": {
            "logs": "logs",
            "raw": "data/raw",
            "interim": "data/interim",
            "processed": "data/processed",
            "exports": "exports",
            "reports": "reports",
            "dataset": "dataset",
        },
    }
    with output.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
        fp.write("\n")
    return output
