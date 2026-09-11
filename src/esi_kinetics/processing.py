from collections.abc import Sequence

import numpy as np

from .models import Crop, PipelineSettings, StageResult


def combine_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Sum frames in float64 without integer overflow."""
    if not frames:
        raise ValueError("At least one frame is required")
    first_shape = frames[0].shape
    if len(first_shape) != 2:
        raise ValueError("Frames must be two-dimensional grayscale arrays")
    if any(frame.shape != first_shape for frame in frames):
        raise ValueError("All frames in a stage must have matching dimensions")
    return np.sum(np.stack([frame.astype(np.float64, copy=False) for frame in frames]), axis=0, dtype=np.float64)


def crop_image(image: np.ndarray, crop: Crop) -> np.ndarray:
    crop.validate(image.shape[1], image.shape[0])
    return image[crop.y0:crop.y1, crop.x0:crop.x1].copy()


def remove_background(image: np.ndarray, settings: PipelineSettings) -> np.ndarray:
    if not settings.enable_background:
        return image.copy()
    if settings.background_mode == "threshold":
        return np.where(image > settings.threshold, image, 0.0)
    if settings.background_mode == "subtract":
        return np.maximum(image - settings.background_value, 0.0)
    raise ValueError(f"Unsupported background mode: {settings.background_mode}")


def process_image(stage: str, combined: np.ndarray, settings: PipelineSettings) -> StageResult:
    image = combined.astype(np.float64, copy=True)
    if settings.enable_crop:
        if settings.crop is None:
            raise ValueError("Crop is enabled but no crop coordinates were provided")
        image = crop_image(image, settings.crop)
    image = remove_background(image, settings)
    return StageResult(
        stage=stage,
        image=image,
        brightness=float(np.sum(image, dtype=np.float64)),
        active_pixels=int(np.count_nonzero(image)),
        maximum=float(np.max(image)) if image.size else 0.0,
        settings=settings,
    )
