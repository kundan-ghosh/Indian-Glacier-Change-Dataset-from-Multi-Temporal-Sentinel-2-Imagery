"""Quality-control checks for generated IGCD rasters and metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QCRecord:
    """Single-sample quality-control outcome."""

    glacier_id: str
    year: int | str
    check: str
    passed: bool
    message: str


def validate_raster(path: str | Path) -> list[QCRecord]:
    """Check that a GeoTIFF can be opened and has sane dimensions."""

    import rasterio

    p = Path(path)
    parts = p.stem.split("_")
    glacier_id = "_".join(parts[:2]) if len(parts) >= 2 else p.stem
    records: list[QCRecord] = []
    if not p.exists():
        return [QCRecord(glacier_id, "unknown", "exists", False, str(p))]
    try:
        with rasterio.open(p) as src:
            records.append(
                QCRecord(glacier_id, "unknown", "open", True, "Readable")
            )
            records.append(
                QCRecord(
                    glacier_id,
                    "unknown",
                    "dimensions",
                    src.width > 0 and src.height > 0,
                    f"{src.width}x{src.height}",
                )
            )
            records.append(
                QCRecord(
                    glacier_id,
                    "unknown",
                    "crs",
                    src.crs is not None,
                    str(src.crs),
                )
            )
    except Exception as exc:
        records.append(QCRecord(glacier_id, "unknown", "open", False, str(exc)))
    return records


def validate_alignment(reference: str | Path, candidate: str | Path) -> QCRecord:
    """Validate CRS, transform, dimensions, and resolution alignment."""

    import rasterio

    glacier_id = Path(candidate).stem.split("_mask")[0]
    try:
        with rasterio.open(reference) as ref, rasterio.open(candidate) as cand:
            passed = (
                ref.crs == cand.crs
                and ref.transform == cand.transform
                and ref.width == cand.width
                and ref.height == cand.height
                and abs(abs(ref.transform.a) - abs(cand.transform.a)) < 1e-9
            )
            message = (
                f"ref=({ref.crs},{ref.transform},{ref.width},{ref.height}); "
                f"cand=({cand.crs},{cand.transform},{cand.width},{cand.height})"
            )
            return QCRecord(glacier_id, "unknown", "alignment", passed, message)
    except Exception as exc:
        return QCRecord(glacier_id, "unknown", "alignment", False, str(exc))


def validate_nonempty_mask(
    mask_path: str | Path,
    min_pixels: int,
) -> QCRecord:
    """Check whether a mask contains enough positive pixels."""

    import rasterio

    glacier_id = Path(mask_path).stem.split("_mask")[0]
    try:
        with rasterio.open(mask_path) as src:
            array = src.read(1)
        count = int(np.count_nonzero(array))
        return QCRecord(
            glacier_id,
            "unknown",
            "nonempty_mask",
            count >= min_pixels,
            f"positive_pixels={count}",
        )
    except Exception as exc:
        return QCRecord(glacier_id, "unknown", "nonempty_mask", False, str(exc))


def area_change_fraction(
    baseline_mask: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    """Compute absolute area-change fraction relative to baseline glacier area."""

    baseline_area = int(np.count_nonzero(baseline_mask))
    target_area = int(np.count_nonzero(target_mask))
    if baseline_area == 0:
        return float("inf")
    return abs(target_area - baseline_area) / baseline_area


def write_quality_reports(
    records: Iterable[QCRecord],
    csv_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Write detailed CSV and aggregate JSON quality reports."""

    rows = [record.__dict__ for record in records]
    df = pd.DataFrame(rows)
    csv_output = Path(csv_path)
    summary_output = Path(summary_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_output, index=False)
    summary = {
        "total_checks": int(len(df)),
        "passed_checks": int(df["passed"].sum()) if not df.empty else 0,
        "failed_checks": int((~df["passed"]).sum()) if not df.empty else 0,
        "failed_by_check": (
            df.loc[~df["passed"]].groupby("check").size().to_dict()
            if not df.empty
            else {}
        ),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_output, summary_output

