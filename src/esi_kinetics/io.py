from collections import defaultdict
from pathlib import Path
import re

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
    written_paths: list[Path] = []
    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        sample_number = 0
        for frame_index, frame_bytes in enumerate(reader):
            while sample_number < frame_count and sample_indices[sample_number] == frame_index:
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
                output_path = output_paths[sample_number]
                Image.fromarray(frame).convert("L").save(output_path)
                written_paths.append(output_path)
                sample_number += 1
            if sample_number == frame_count:
                break
        if sample_number != frame_count:
            raise ValueError(
                f"Video decoding yielded only {sample_number} of {frame_count} requested frames"
            )
    except Exception:
        for output_path in written_paths:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        reader.close()
    return output_paths


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
    return csv_path
