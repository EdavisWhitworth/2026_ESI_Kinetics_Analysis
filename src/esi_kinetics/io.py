from collections import defaultdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
import imageio_ffmpeg
import pandas as pd
from PIL import Image

from .models import Experiment, Stage, StageResult
from .processing import combine_frames

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _validate_reference_profile(signal_mask: np.ndarray, background_reference: np.ndarray) -> None:
    if signal_mask.ndim != 2 or background_reference.shape != signal_mask.shape:
        raise ValueError("Signal mask and background reference must have matching 2D shapes")
    if not np.any(signal_mask):
        raise ValueError("Select a target signal segment before saving the profile")
    if not np.all(np.isfinite(background_reference)):
        raise ValueError("Background reference must contain only finite pixel values")


def stage_key(path: Path) -> str:
    match = re.search(r"(?:stage|step|s)[_ -]?(\d+)", path.stem, re.IGNORECASE)
    return f"Stage {int(match.group(1)):02d}" if match else "Stage 01"


def discover_stages(folder: Path) -> list[Stage]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            grouped[stage_key(path)].append(path)
    return [
        Stage(name, tuple(paths))
        for name, paths in sorted(
            grouped.items(),
            key=lambda item: int(re.search(r"\d+", item[0]).group()),
        )
    ]


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64)


def load_experiment(folder: Path) -> Experiment:
    stages = discover_stages(folder)
    if not stages:
        raise ValueError("No supported image files were found")
    experiment = Experiment(stages=stages)
    for stage in stages:
        experiment.combined[stage.name] = combine_frames([read_image(path) for path in stage.paths])
    return experiment


def _bright_frame_indices(brightness: list[float]) -> list[int]:
    values = np.asarray(brightness, dtype=np.float64)
    unique_levels = np.unique(values)
    if unique_levels.size < 2:
        return list(range(values.size))

    level_gaps = np.diff(unique_levels)
    largest_gap_index = int(np.argmax(level_gaps))
    largest_gap = float(level_gaps[largest_gap_index])
    remaining_gaps = np.delete(level_gaps, largest_gap_index)
    typical_gap = float(np.median(remaining_gaps)) if remaining_gaps.size else 0.0
    brightness_range = float(unique_levels[-1] - unique_levels[0])
    if largest_gap < max(1.0, 3.0 * typical_gap, 0.05 * brightness_range):
        return list(range(values.size))

    cutoff = (unique_levels[largest_gap_index] + unique_levels[largest_gap_index + 1]) / 2.0
    return np.flatnonzero(values > cutoff).tolist()


def extract_video_frames(video_path: Path, output_folder: Path, frame_count: int = 100) -> list[Path]:
    if frame_count < 1:
        raise ValueError("Frame count must be at least one")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_folder.mkdir(parents=True, exist_ok=True)
    if any(output_folder.iterdir()):
        raise ValueError("Choose an empty folder for extracted video frames")

    total_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    if total_frames < 1:
        raise ValueError("The selected video contains no readable frames")
    sample_indices = np.rint(np.linspace(0, total_frames - 1, frame_count)).astype(np.int64)
    output_paths = [output_folder / f"stage_{index + 1:03d}.png" for index in range(frame_count)]
    with TemporaryDirectory(dir=output_folder) as temporary_folder:
        temporary_paths = [
            Path(temporary_folder) / f"stage_{index + 1:03d}.png"
            for index in range(frame_count)
        ]
        brightness: list[float] = []
        reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            width, height = metadata["size"]
            sample_number = 0
            for frame_index, frame_bytes in enumerate(reader):
                while sample_number < frame_count and sample_indices[sample_number] == frame_index:
                    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
                    grayscale = Image.fromarray(frame).convert("L")
                    grayscale.save(temporary_paths[sample_number])
                    brightness.append(float(np.asarray(grayscale, dtype=np.float64).mean()))
                    sample_number += 1
                if sample_number == frame_count:
                    break
            if sample_number != frame_count:
                raise ValueError(
                    f"Video decoding yielded only {sample_number} of {frame_count} requested frames"
                )
        finally:
            reader.close()

        selected_indices = _bright_frame_indices(brightness)
        selected_paths = []
        for index in selected_indices:
            temporary_paths[index].replace(output_paths[index])
            selected_paths.append(output_paths[index])
    return selected_paths


def save_reference_profile(signal_mask: np.ndarray, background_reference: np.ndarray, path: Path) -> Path:
    _validate_reference_profile(signal_mask, background_reference)
    with path.open("wb") as profile_file:
        np.savez_compressed(
            profile_file,
            signal_mask=signal_mask.astype(bool),
            background_reference=background_reference.astype(np.float64),
        )
    return path


def load_reference_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as profile:
        signal_mask = profile["signal_mask"].astype(bool)
        background_reference = profile["background_reference"].astype(np.float64)
    _validate_reference_profile(signal_mask, background_reference)
    return signal_mask, background_reference


def export_results(results: list[StageResult], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.stage)
        image = np.clip(result.image, 0, np.iinfo(np.uint32).max).astype(np.uint32)
        Image.fromarray(image).save(folder / f"{safe_name}_processed.tiff")
        rows.append({
            "stage": result.stage,
            "Integrated_Brightness": result.brightness,
            "Active_Pixels": result.active_pixels,
            "Peak_Intensity": result.maximum,
            "Mean_Intensity": result.mean_intensity,
            "Background_Correction": (
                "saved_reference_subtract"
                if result.settings.enable_reference_background
                else "spatial_plane"
                if result.settings.enable_spatial_background
                else result.settings.background_mode if result.settings.enable_background else "disabled"
            ),
        })
    csv_path = folder / "brightness_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    average_mean_intensity = float(np.mean([result.mean_intensity for result in results]))
    pd.DataFrame([{"Average_Mean_Intensity": average_mean_intensity}]).to_csv(
        folder / "mean_intensity_average.csv", index=False
    )
    return csv_path
