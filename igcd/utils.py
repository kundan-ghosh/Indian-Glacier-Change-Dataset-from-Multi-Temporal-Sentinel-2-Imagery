"""Shared utilities for logging, manifests, and deterministic processing."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def setup_logging(log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure structured console and file logging for the pipeline."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("igcd")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        log_path / f"igcd_{utc_now_slug()}.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def utc_now_slug() -> str:
    """Return a stable UTC timestamp suitable for filenames."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_directories(paths: Iterable[str | Path]) -> list[Path]:
    """Create directories and return normalized paths."""

    created = []
    for item in paths:
        path = Path(item)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def stable_id(prefix: str, values: Sequence[Any], width: int = 6) -> str:
    """Build a deterministic identifier from ordered values."""

    text = "|".join(str(value) for value in values)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    number = int(digest[:10], 16) % (10**width)
    return f"{prefix}_{number:0{width}d}"


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size slices from a sequence."""

    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON object with stable formatting."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return output


def file_exists_and_nonempty(path: str | Path) -> bool:
    """Return whether a path exists and has at least one byte."""

    candidate = Path(path)
    return candidate.exists() and candidate.stat().st_size > 0

