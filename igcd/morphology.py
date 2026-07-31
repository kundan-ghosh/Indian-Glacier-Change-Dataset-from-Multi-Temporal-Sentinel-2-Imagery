"""Morphological post-processing for glacier masks."""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover - exercised only in minimal runtimes.
    ndi = None

try:
    from skimage import measure, morphology
except ImportError:  # pragma: no cover - exercised only in minimal runtimes.
    measure = None
    morphology = None


def clean_binary_mask(
    mask: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 2,
    min_object_size: int = 9,
    min_component_area: int = 16,
    fill_holes: bool = True,
) -> np.ndarray:
    """Apply standard morphology and connected-component filtering.

    Parameters
    ----------
    mask:
        Boolean or binary array where non-zero values represent glacier pixels.
    opening_radius, closing_radius:
        Disk radius used for binary opening and closing.
    min_object_size:
        Minimum object size passed to scikit-image.
    min_component_area:
        Minimum connected-component area retained after morphology.
    fill_holes:
        Whether to fill interior holes.
    """

    result = np.asarray(mask).astype(bool)
    if morphology is None:
        if any((opening_radius, closing_radius, min_object_size)):
            raise RuntimeError(
                "scikit-image is required for opening, closing, and small "
                "object removal. Install requirements.txt first."
            )
        if fill_holes and ndi is None:
            result = _fill_holes_fallback(result)
        elif fill_holes:
            result = ndi.binary_fill_holes(result)
        if min_component_area > 0:
            result = keep_large_components(result, min_component_area)
        return result.astype("uint8")

    if opening_radius > 0:
        result = morphology.binary_opening(
            result, morphology.disk(opening_radius)
        )
    if closing_radius > 0:
        result = morphology.binary_closing(
            result, morphology.disk(closing_radius)
        )
    if fill_holes and ndi is not None:
        result = ndi.binary_fill_holes(result)
    elif fill_holes:
        result = _fill_holes_fallback(result)
    if min_object_size > 0:
        result = morphology.remove_small_objects(result, min_object_size)
    if min_component_area > 0:
        result = keep_large_components(result, min_component_area)
    return result.astype("uint8")


def keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Retain connected components with area greater than or equal to a limit."""

    if measure is None:
        return _keep_large_components_fallback(mask, min_area)
    labels = measure.label(np.asarray(mask).astype(bool), connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero(counts >= min_area)
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def _keep_large_components_fallback(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Pure NumPy connected-component fallback for minimal environments."""

    source = np.asarray(mask).astype(bool)
    visited = np.zeros(source.shape, dtype=bool)
    output = np.zeros(source.shape, dtype=bool)
    rows, cols = source.shape
    for row in range(rows):
        for col in range(cols):
            if visited[row, col] or not source[row, col]:
                continue
            stack = [(row, col)]
            component: list[tuple[int, int]] = []
            visited[row, col] = True
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < rows
                            and 0 <= nx < cols
                            and not visited[ny, nx]
                            and source[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) >= min_area:
                for y, x in component:
                    output[y, x] = True
    return output


def _fill_holes_fallback(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes using border flood fill without SciPy."""

    source = np.asarray(mask).astype(bool)
    background = ~source
    visited = np.zeros(source.shape, dtype=bool)
    rows, cols = source.shape
    stack: list[tuple[int, int]] = []
    for row in range(rows):
        for col in (0, cols - 1):
            if background[row, col]:
                stack.append((row, col))
    for col in range(cols):
        for row in (0, rows - 1):
            if background[row, col]:
                stack.append((row, col))
    while stack:
        y, x = stack.pop()
        if visited[y, x] or not background[y, x]:
            continue
        visited[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                stack.append((ny, nx))
    holes = background & ~visited
    return source | holes
