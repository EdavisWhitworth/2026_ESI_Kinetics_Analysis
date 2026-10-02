from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import imageio_ffmpeg
import pandas as pd
import xlsxwriter
from PIL import Image

from .models import Experiment, PipelineSettings, Stage, StageResult
from .processing import combine_frames, process_image

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DELAY_PATTERN = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(fs|ps|ns|us|µs|μs|ms|s)(?![A-Za-z])",
    re.IGNORECASE,
)
DELAY_TO_SECONDS = {
    "fs": 1e-15,
    "ps": 1e-12,
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
}


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


def video_frames_folder(video_path: Path) -> Path:
    base_folder = video_path.with_name(f"{video_path.stem} Frames")
    candidate = base_folder
    suffix = 2
    while candidate.exists() and (
        not candidate.is_dir() or any(candidate.iterdir())
    ):
        candidate = base_folder.with_name(f"{base_folder.name} ({suffix})")
        suffix += 1
    return candidate


def parse_time_delay(folder_name: str) -> tuple[float, str]:
    match = DELAY_PATTERN.search(folder_name)
    if match is None:
        raise ValueError(
            f"Folder name '{folder_name}' must include a delay value and unit, such as '2 ms'."
        )
    value = float(match.group(1))
    unit = match.group(2).lower()
    if not np.isfinite(value):
        raise ValueError(f"Folder name '{folder_name}' contains an invalid time delay.")
    return value * DELAY_TO_SECONDS[unit], unit


def analyze_delay_folders(parent_folder: Path, background_reference: np.ndarray) -> list[dict[str, object]]:
    if background_reference.ndim != 2 or not np.all(np.isfinite(background_reference)):
        raise ValueError("The loaded background reference must be a finite 2D image.")

    rows: list[dict[str, object]] = []
    for folder in parent_folder.iterdir():
        if not folder.is_dir():
            continue
        image_paths = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not image_paths:
            continue
        delay_seconds, delay_unit = parse_time_delay(folder.name)
        frame_means = []
        for image_path in image_paths:
            image = read_image(image_path)
            if image.shape != background_reference.shape:
                raise ValueError(
                    f"Image dimensions in '{folder.name}' do not match the loaded background."
                )
            result = process_image(
                image_path.name,
                image,
                PipelineSettings(
                    enable_background=False,
                    enable_connected_region=False,
                    reference_background=background_reference,
                    enable_reference_background=True,
                ),
            )
            frame_means.append(result.mean_intensity)
        rows.append({
            "Folder": folder.name,
            "Time_Delay": delay_seconds,
            "Time_Unit_In_Title": delay_unit,
            "Mean_Intensity": float(np.mean(frame_means, dtype=np.float64)),
            "Images_Averaged": len(frame_means),
        })

    if not rows:
        raise ValueError("No image folders were found in the selected parent folder.")
    return sorted(rows, key=lambda row: float(row["Time_Delay"]))


def export_delay_analysis(rows: list[dict[str, object]], workbook_path: Path) -> Path:
    if not rows:
        raise ValueError("There are no delay results to export.")
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Folder", "Time_Delay_s", "Time_Unit_In_Title", "Mean_Intensity", "Images_Averaged"
    ]
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        dataframe = pd.DataFrame(rows, columns=columns)
        dataframe.to_excel(writer, sheet_name="Mean Intensity", index=False)
        worksheet = writer.sheets["Mean Intensity"]
        worksheet.freeze_panes(1, 0)
        worksheet.set_column("A:A", 28)
        worksheet.set_column("B:B", 16)
        worksheet.set_column("C:C", 20)
        worksheet.set_column("D:D", 20)
        worksheet.set_column("E:E", 18)
        worksheet.autofilter(0, 0, len(rows), len(columns) - 1)

        chart = writer.book.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
        chart.add_series({
            "name": "Mean Intensity",
            "categories": ["Mean Intensity", 1, 1, len(rows), 1],
            "values": ["Mean Intensity", 1, 3, len(rows), 3],
            "marker": {"type": "circle", "size": 6},
            "line": {"none": True},
        })
        chart.set_title({"name": "Mean Intensity vs Time Delay"})
        chart.set_x_axis({"name": "Time Delay (s)", "num_format": "0.#######"})
        chart.set_y_axis({"name": "Mean Intensity"})
        chart.set_legend({"none": True})
        chart.set_size({"width": 760, "height": 440})
        worksheet.insert_chart("G2", chart)
    return workbook_path


def _bright_frame_indices(brightness: list[float]) -> list[int]:
    values = np.asarray(brightness, dtype=np.float64)
    if values.size < 2 or np.all(values == values[0]):
        return list(range(values.size))

    average_brightness = float(np.mean(values, dtype=np.float64))
    selected = np.flatnonzero(values > average_brightness).tolist()
    return selected or list(range(values.size))


def extract_video_frames(video_path: Path, output_folder: Path, frame_count: int = 100) -> list[Path]:
    if frame_count < 1:
        raise ValueError("Frame count must be at least one")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_folder.mkdir(parents=True, exist_ok=True)
    if any(output_folder.iterdir()):
        raise ValueError("Choose an empty folder for extracted video frames")
    all_frames_folder = output_folder / "All Frames"
    all_frames_folder.mkdir()

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
        for index, temporary_path in enumerate(temporary_paths):
            shutil.copy2(temporary_path, all_frames_folder / f"stage_{index + 1:03d}.png")
        selected_paths = []
        for index in selected_indices:
            temporary_paths[index].replace(output_paths[index])
            selected_paths.append(output_paths[index])
    return selected_paths


def extract_video_batch(
    video_paths: list[Path],
    frame_count: int = 100,
    progress_callback: Callable[[Path, int, int], None] | None = None,
) -> tuple[list[tuple[Path, int]], list[str]]:
    completed: list[tuple[Path, int]] = []
    failures: list[str] = []
    for index, video_path in enumerate(video_paths, start=1):
        if progress_callback is not None:
            progress_callback(video_path, index, len(video_paths))
        try:
            output_folder = video_frames_folder(video_path)
            frames = extract_video_frames(video_path, output_folder, frame_count)
            completed.append((output_folder, len(frames)))
        except Exception as error:
            failures.append(f"{video_path.name}: {error}")
    return completed, failures


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
