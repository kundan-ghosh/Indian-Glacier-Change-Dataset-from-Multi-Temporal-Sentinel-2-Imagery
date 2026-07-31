"""Organize completed Google Drive Earth Engine exports into IGCD folders."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from igcd.config import IGCDConfig

EXPORT_RE = re.compile(
    r"^(?P<glacier_id>IGCD_\d{6})_(?P<year>\d{4})_"
    r"(?P<kind>sentinel|mask)(?:-\d+-\d+)?\.tif$"
)


def organize_drive_exports(
    drive_export_dir: str | Path,
    config: IGCDConfig,
    logger: logging.Logger | None = None,
    move: bool = False,
) -> list[Path]:
    """Copy or move completed Earth Engine exports into standard folders.

    Earth Engine Drive exports are usually written as flat files inside the
    configured Drive folder. This helper places Sentinel composites under
    ``exports/<year>/`` and glacier masks under
    ``data/processed/masks/<year>/`` using the project layout expected by the
    downstream change-mask, quality-control, and packaging stages.
    """

    source_dir = Path(drive_export_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Drive export directory not found: {source_dir}")

    organized: list[Path] = []
    for source in sorted(source_dir.glob("*.tif")):
        match = EXPORT_RE.match(source.name)
        if not match:
            if logger:
                logger.debug("Ignoring unrecognized export filename: %s", source)
            continue

        glacier_id = match.group("glacier_id")
        year = match.group("year")
        kind = match.group("kind")
        if kind == "sentinel":
            target = (
                config.paths["exports"]
                / year
                / f"{glacier_id}_{year}_sentinel.tif"
            )
        else:
            target = (
                config.paths["processed"]
                / "masks"
                / year
                / f"{glacier_id}_{year}_mask.tif"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            organized.append(target)
            continue
        if move:
            shutil.move(str(source), target)
        else:
            shutil.copy2(source, target)
        organized.append(target)
        if logger:
            logger.info("Organized export %s -> %s", source.name, target)
    return organized

