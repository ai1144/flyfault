"""Predeclared behavior metrics from decision-boundary poses."""

from __future__ import annotations

import numpy as np


def behavior_summary(
    time_s: np.ndarray,
    xy_mm: np.ndarray,
    heading_rad: np.ndarray,
    source_xy_mm: tuple[float, float],
    source_radius_mm: float,
    terminal: str,
) -> dict:
    delta = np.asarray(source_xy_mm) - xy_mm
    distance = np.linalg.norm(delta, axis=1)
    path_length = float(np.linalg.norm(np.diff(xy_mm, axis=0), axis=1).sum())
    improvement = float(distance[0] - distance[-1])
    source_bearing = np.arctan2(delta[:, 1], delta[:, 0])
    heading_error = np.angle(np.exp(1j * (source_bearing - heading_rad)))
    reached = terminal == "source_reached"
    return {
        "duration_s": float(time_s[-1]),
        "reached": reached,
        "latency_s": float(time_s[-1]) if reached else None,
        "terminal": terminal,
        "flipped": terminal == "flipped",
        "out_of_bounds": terminal == "out_of_bounds",
        "start_distance_mm": float(distance[0]),
        "end_distance_mm": float(distance[-1]),
        "closest_distance_mm": float(distance.min()),
        "source_radius_mm": source_radius_mm,
        "distance_improvement_mm": improvement,
        "path_length_mm": path_length,
        "path_efficiency": improvement / path_length if path_length else None,
        "mean_abs_heading_error_deg": float(np.degrees(np.abs(heading_error)).mean()),
        "signed_turn_deg": float(np.degrees(np.unwrap(heading_rad)[-1] - heading_rad[0])),
    }
