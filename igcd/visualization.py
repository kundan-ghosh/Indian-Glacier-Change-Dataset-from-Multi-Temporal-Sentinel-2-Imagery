"""Visualization helpers for inventory, masks, and change products."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CHANGE_COLORS = {
    0: (0, 0, 0),
    1: (0, 170, 255),
    2: (255, 80, 80),
    3: (80, 220, 120),
}

CHANGE_LABELS = {
    0: "Background",
    1: "Stable Glacier",
    2: "Glacier Retreat",
    3: "Glacier Advance",
}


def save_change_preview(change_mask: np.ndarray, output_path: str | Path) -> Path:
    """Save an RGB PNG preview for a four-class change mask."""

    import matplotlib.pyplot as plt

    rgb = change_mask_to_rgb(change_mask)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, rgb)
    return output


def change_mask_to_rgb(change_mask: np.ndarray) -> np.ndarray:
    """Convert a four-class change mask to an RGB display image."""

    mask = change_mask.astype("uint8")
    rgb = np.zeros((*mask.shape, 3), dtype="uint8")
    for label, color in CHANGE_COLORS.items():
        rgb[mask == label] = color
    return rgb


def blend_change_overlay(
    base_rgb: np.ndarray,
    change_mask: np.ndarray,
    alpha: float = 0.45,
    include_background: bool = False,
) -> np.ndarray:
    """Blend change classes over an RGB image for visual inspection."""

    base = np.clip(base_rgb.astype("float32", copy=True), 0, 1)
    colors = change_mask_to_rgb(change_mask).astype("float32") / 255.0
    valid = np.isin(change_mask, [1, 2, 3])
    if include_background:
        valid = np.isin(change_mask, [0, 1, 2, 3])
    base[valid] = (1 - alpha) * base[valid] + alpha * colors[valid]
    return np.clip(base, 0, 1)


def change_class_counts(change_mask: np.ndarray) -> dict[str, int]:
    """Return pixel counts for each change-map class."""

    labels, counts = np.unique(change_mask.astype("uint8"), return_counts=True)
    raw = dict(zip(labels.tolist(), counts.tolist()))
    return {
        CHANGE_LABELS[label]: int(raw.get(label, 0))
        for label in sorted(CHANGE_LABELS)
    }


def plot_inventory_overview(gdf: object, output_path: str | Path) -> Path:
    """Save a simple glacier inventory footprint map."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(ax=ax, linewidth=0.2, edgecolor="navy", facecolor="skyblue")
    ax.set_title("IGCD Glacier Inventory")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output
