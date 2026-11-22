"""Validated experiment parameters and stimulus mirroring."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import yaml

CONDITIONS = (
    "intact", "lesion_left", "lesion_right", "sham_random", "rescue", "no_neural_control"
)
STIMULUS_SIDES = ("left", "right")


def grid_steps(interval: float, dt: float, label: str) -> int:
    steps = round(interval / dt)
    if steps < 1 or not math.isclose(interval, steps * dt, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{label} must be a positive integer multiple of {dt:g} s")
    return steps


def positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Timing:
    duration_s: float
    decision_s: float
    neural_dt_s: float

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            positive(value, key)
        grid_steps(self.duration_s, self.decision_s, "duration_s")
        grid_steps(self.decision_s, self.neural_dt_s, "decision_s")


@dataclass(frozen=True)
class Arena:
    start_xy_mm: tuple[float, float]
    start_yaw_rad: float
    source_xy_mm: tuple[float, float]
    sigma_mm: float
    source_radius_mm: float
    limit_mm: float

    def __post_init__(self) -> None:
        for key in ("sigma_mm", "source_radius_mm", "limit_mm"):
            positive(getattr(self, key), key)
        for xy in (self.start_xy_mm, self.source_xy_mm):
            if len(xy) != 2 or any(not math.isfinite(p) or abs(p) >= self.limit_mm for p in xy):
                raise ValueError("start and source must be finite XY pairs inside the arena")
        if not math.isfinite(self.start_yaw_rad):
            raise ValueError("start_yaw_rad must be finite")
        if math.dist(self.start_xy_mm, self.source_xy_mm) <= self.source_radius_mm:
            raise ValueError("fly must start outside the source radius")

    def for_side(self, side: str) -> Arena:
        if side not in STIMULUS_SIDES:
            raise ValueError(f"Unknown stimulus side {side!r}")
        x, y = self.source_xy_mm
        return replace(self, source_xy_mm=(x, y if side == "left" else 2 * self.start_xy_mm[1] - y))


@dataclass(frozen=True)
class Circuit:
    lesion_fraction: float

    def __post_init__(self) -> None:
        if not 0 <= self.lesion_fraction <= 1:
            raise ValueError("lesion_fraction must be between zero and one")


@dataclass(frozen=True)
class Media:
    fps: int
    playback_speed: float

    def __post_init__(self) -> None:
        if type(self.fps) is not int or not 1 <= self.fps <= 60:
            raise ValueError("fps must be an integer between 1 and 60")
        positive(self.playback_speed, "playback_speed")
        if self.playback_speed > 1:
            raise ValueError("playback_speed must be at most 1")


@dataclass(frozen=True)
class Protocol:
    name: str
    seeds: tuple[int, ...]
    stimulus_sides: tuple[str, ...]
    conditions: tuple[str, ...]
    timing: Timing
    arena: Arena
    circuit: Circuit
    media: Media

    def __post_init__(self) -> None:
        if not self.name or not self.name.isascii() or not self.name.replace("-", "").isalnum():
            raise ValueError("name must contain only ASCII letters, digits and hyphens")
        if not self.seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("seeds must be nonempty unsigned 32-bit integers")
        for values, permitted in ((self.stimulus_sides, STIMULUS_SIDES), (self.conditions, CONDITIONS)):
            if not values or any(value not in permitted for value in values):
                raise ValueError(f"Expected one or more of {permitted}")
        for values in (self.seeds, self.stimulus_sides, self.conditions):
            if len(set(values)) != len(values):
                raise ValueError("protocol entries must be unique")
        if self.arena.source_xy_mm[1] <= self.arena.start_xy_mm[1]:
            raise ValueError("canonical source y must be above start y for stimulus mirroring")
        for side in self.stimulus_sides:
            self.arena.for_side(side)


def load_protocol(path: Path) -> tuple[Protocol, str]:
    raw = path.read_bytes()
    values = yaml.safe_load(raw)
    if not isinstance(values, dict):
        raise ValueError("configuration must be a YAML mapping")
    for key in ("seeds", "stimulus_sides", "conditions"):
        values[key] = tuple(values[key])
    arena = values["arena"]
    for key in ("start_xy_mm", "source_xy_mm"):
        arena[key] = tuple(arena[key])
    values.update(
        timing=Timing(**values["timing"]), arena=Arena(**arena),
        circuit=Circuit(**values["circuit"]), media=Media(**values["media"]),
    )
    return Protocol(**values), sha256(raw).hexdigest()
