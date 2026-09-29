from collections.abc import Sequence

import numpy as np
from scipy import ndimage

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
    combined = np.zeros(first_shape, dtype=np.float64)
    for frame in frames:
        combined += frame
    return combined


def crop_image(image: np.ndarray, crop: Crop) -> np.ndarray:
    crop.validate(image.shape[1], image.shape[0])
    return image[crop.y0:crop.y1, crop.x0:crop.x1].copy()


def estimate_local_background(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[1:-1, 0],
        image[1:-1, -1],
    ])
    return float(np.median(border)) if border.size else 0.0


def remove_background(image: np.ndarray, settings: PipelineSettings) -> np.ndarray:
    if not settings.enable_background:
        return image.copy()
    if settings.background_mode == "threshold":
        return np.where(image > settings.threshold, image, 0.0)
    if settings.background_mode == "subtract":
        background_value = settings.background_value
        if background_value == 0.0:
            background_value = estimate_local_background(image)
        return np.maximum(image - background_value, 0.0)
    raise ValueError(f"Unsupported background mode: {settings.background_mode}")


def subtract_spatial_background(image: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
    if background_mask.shape != image.shape:
        raise ValueError("Background mask must match the processed image")
    if np.count_nonzero(background_mask) < 3:
        raise ValueError("Select at least three background pixels")

    y_coords, x_coords = np.indices(image.shape, dtype=np.float64)
    design = np.column_stack((x_coords[background_mask], y_coords[background_mask], np.ones(np.count_nonzero(background_mask))))
    values = image[background_mask]
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank < 3:
        raise ValueError("Background pixels must cover enough area to estimate a spatial gradient")
    background = coefficients[0] * x_coords + coefficients[1] * y_coords + coefficients[2]
    return np.maximum(image - background, 0.0)


def estimate_reference_background(image: np.ndarray, signal_mask: np.ndarray) -> np.ndarray:
    if signal_mask.shape != image.shape:
        raise ValueError("Signal mask must match the reference image")
    background_mask = ~signal_mask
    if np.count_nonzero(background_mask) < 3:
        raise ValueError("The selected signal leaves too few background pixels")

    height, width = image.shape
    x_center = (width - 1) / 2.0
    y_center = (height - 1) / 2.0
    count = sum_x = sum_y = sum_value = 0.0
    sum_xx = sum_xy = sum_yy = sum_x_value = sum_y_value = 0.0
    for row_index in range(height):
        x_coords = np.flatnonzero(background_mask[row_index]).astype(np.float64) - x_center
        if not x_coords.size:
            continue
        values = image[row_index, background_mask[row_index]]
        y_coord = row_index - y_center
        count += x_coords.size
        sum_x += float(np.sum(x_coords))
        sum_y += y_coord * x_coords.size
        sum_value += float(np.sum(values))
        sum_xx += float(np.dot(x_coords, x_coords))
        sum_xy += y_coord * float(np.sum(x_coords))
        sum_yy += y_coord * y_coord * x_coords.size
        sum_x_value += float(np.dot(x_coords, values))
        sum_y_value += y_coord * float(np.sum(values))

    normal_matrix = np.array([
        [sum_xx, sum_xy, sum_x],
        [sum_xy, sum_yy, sum_y],
        [sum_x, sum_y, count],
    ])
    if np.linalg.matrix_rank(normal_matrix) < 3:
        raise ValueError("The non-highlighted background must span both image directions")
    coefficients = np.linalg.solve(normal_matrix, [sum_x_value, sum_y_value, sum_value])
    x_coords = np.arange(width, dtype=np.float64) - x_center
    y_coords = np.arange(height, dtype=np.float64) - y_center
    fitted_background = (
        coefficients[0] * x_coords[None, :]
        + coefficients[1] * y_coords[:, None]
        + coefficients[2]
    )
    fitted_background[background_mask] = image[background_mask]
    return fitted_background


def keep_largest_connected_region(image: np.ndarray) -> np.ndarray:
    """Keep the brightest connected foreground region using 8-connectivity."""
    foreground = image > 0
    if not np.any(foreground):
        return np.zeros_like(image)

    height, width = foreground.shape
    visited = np.zeros_like(foreground)
    largest: list[tuple[int, int]] = []
    for start_y, start_x in zip(*np.nonzero(foreground)):
        if visited[start_y, start_x]:
            continue
        component = []
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if foreground[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        if len(component) > len(largest):
            largest = component

    connected = np.zeros_like(image)
    coordinates = tuple(zip(*largest))
    connected[coordinates] = image[coordinates]
    return connected


def connected_component_labels(image: np.ndarray, threshold: float) -> np.ndarray:
    """Label 8-connected pixels strictly brighter than threshold."""
    foreground = image > threshold
    labels, _ = ndimage.label(foreground, structure=np.ones((3, 3), dtype=bool))
    return labels.astype(np.int32, copy=False)


def process_image(stage: str, combined: np.ndarray, settings: PipelineSettings) -> StageResult:
    image = combined.astype(np.float64, copy=True)
    crop_image_output = settings.enable_crop and not settings.zero_outside_crop
    if settings.enable_reference_background:
        if settings.reference_background is None:
            raise ValueError("Reference background correction requires a saved background image")
        if settings.reference_background.shape != image.shape:
            raise ValueError("Saved background image must match the current image dimensions")
        image = np.maximum(image - settings.reference_background, 0.0)
    elif settings.enable_spatial_background:
        if settings.background_mask is None:
            raise ValueError("Spatial background correction requires a background mask")
        image = subtract_spatial_background(image, settings.background_mask)
    else:
        if crop_image_output:
            if settings.crop is None:
                raise ValueError("Crop is enabled but no crop coordinates were provided")
            image = crop_image(image, settings.crop)
        image = remove_background(image, settings)

    if settings.zero_outside_crop:
        if settings.crop is None:
            raise ValueError("Zero-outside-box mode requires box coordinates")
        settings.crop.validate(image.shape[1], image.shape[0])
        box_mask = np.zeros(image.shape, dtype=bool)
        box_mask[settings.crop.y0:settings.crop.y1, settings.crop.x0:settings.crop.x1] = True
        image = np.where(box_mask, image, 0.0)
        region_mask = settings.region_mask
    elif settings.enable_reference_background:
        if crop_image_output:
            if settings.crop is None:
                raise ValueError("Crop is enabled but no crop coordinates were provided")
            image = crop_image(image, settings.crop)
        region_mask = None
    elif crop_image_output:
        if settings.crop is None:
            raise ValueError("Crop is enabled but no crop coordinates were provided")
        if settings.enable_spatial_background:
            image = crop_image(image, settings.crop)
        region_mask = crop_image(settings.region_mask, settings.crop) if settings.region_mask is not None else None
    else:
        region_mask = settings.region_mask
    if region_mask is not None:
        if region_mask.shape != image.shape:
            raise ValueError("Selected region mask must match the processed image")
        image = np.where(region_mask, image, 0.0)
    elif settings.enable_connected_region and not settings.enable_reference_background:
        image = keep_largest_connected_region(image)
    nonzero_pixels = int(np.count_nonzero(image))
    mean_intensity = float(np.sum(image, dtype=np.float64) / nonzero_pixels) if nonzero_pixels else 0.0
    return StageResult(
        stage=stage,
        image=image,
        brightness=float(np.sum(image, dtype=np.float64)),
        active_pixels=nonzero_pixels,
        maximum=float(np.max(image)) if image.size else 0.0,
        mean_intensity=mean_intensity,
        settings=settings,
    )
