from collections import defaultdict
from pathlib import Path
import re

import numpy as np
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
    return [Stage(name, tuple(paths)) for name, paths in sorted(grouped.items())]


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
