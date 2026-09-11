from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image

from .models import Experiment, Stage, StageResult
from .processing import combine_frames

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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


def export_results(results: list[StageResult], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.stage)
        image = np.clip(result.image, 0, np.iinfo(np.uint32).max).astype(np.uint32)
        Image.fromarray(image).save(folder / f"{safe_name}_processed.tiff")
        rows.append({
            "stage": result.stage,
            "brightness": result.brightness,
            "active_pixels": result.active_pixels,
            "maximum": result.maximum,
        })
    csv_path = folder / "brightness_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path
