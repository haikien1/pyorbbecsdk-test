"""Depth sampling and pinhole camera geometry helpers."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MIN_DEPTH_M = 1.0
MAX_DEPTH_M = 5.0


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def deproject_pixel_to_3d(u: float, v: float, depth: float, intrinsics: Intrinsics) -> np.ndarray:
    """Convert aligned color-image pixel and depth in meters to camera XYZ meters.

    Gemini camera coordinates follow the SDK pinhole convention: +X right,
    +Y down, +Z forward from the camera.
    """
    z = float(depth)
    return np.array([(u - intrinsics.cx) * z / intrinsics.fx,
                     (v - intrinsics.cy) * z / intrinsics.fy, z], dtype=np.float64)


def robust_depth_m(depth_m: np.ndarray, box: Tuple[int, int, int, int],
                   min_m: float = MIN_DEPTH_M, max_m: float = MAX_DEPTH_M) -> Optional[float]:
    """Median valid depth from the central 50% of a box; rejects zeros, NaNs and tails."""
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    if x2 <= x1 or y2 <= y1:
        return None
    # Restrict sampling to the center so pixels on the object's silhouette/background
    # do not dominate the measurement, including when a box touches an image edge.
    padx, pady = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
    roi = depth_m[y1 + pady:y2 - pady or y2, x1 + padx:x2 - padx or x2]
    values = roi[np.isfinite(roi) & (roi >= min_m) & (roi <= max_m)]
    if values.size < 5:
        return None
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad > 0:
        values = values[np.abs(values - med) <= max(3.5 * 1.4826 * mad, 0.03)]
    return float(np.median(values)) if values.size >= 5 else None
