from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

BackgroundMode = Literal["threshold", "subtract"]


@dataclass(frozen=True)
class Stage:
    name: str
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class Crop:
    x0: int
    y0: int
    x1: int
    y1: int

    def validate(self, width: int, height: int) -> None:
        if not (0 <= self.x0 < self.x1 <= width and 0 <= self.y0 < self.y1 <= height):
            raise ValueError("Crop coordinates must be inside the image and non-empty")


@dataclass(frozen=True)
class PipelineSettings:
    crop: Crop | None = None
    background_mode: BackgroundMode = "threshold"
    threshold: float = 0.0
    background_value: float = 0.0
    enable_combine: bool = True
    enable_crop: bool = False
    enable_background: bool = True
    enable_connected_region: bool = True
    region_mask: np.ndarray | None = None
    background_mask: np.ndarray | None = None
    enable_spatial_background: bool = False
    reference_background: np.ndarray | None = None
    enable_reference_background: bool = False
    zero_outside_crop: bool = False


@dataclass(frozen=True)
class StageResult:
    stage: str
    image: np.ndarray
    brightness: float
    active_pixels: int
    maximum: float
    mean_intensity: float
    settings: PipelineSettings


@dataclass
class Experiment:
    stages: list[Stage] = field(default_factory=list)
    combined: dict[str, np.ndarray] = field(default_factory=dict)
    results: dict[str, StageResult] = field(default_factory=dict)
