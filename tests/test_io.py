from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from esi_kinetics.io import (
    discover_stages,
    analyze_delay_folders,
    export_delay_analysis,
    _bright_frame_indices,
    export_results,
    extract_video_batch,
    extract_video_frames,
    load_experiment,
    load_reference_profile,
    save_reference_profile,
    video_frames_folder,
)
from esi_kinetics.models import Crop, PipelineSettings
from esi_kinetics.processing import process_image


def _write_test_video(path: Path, frame_count: int = 12) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(8, 6),
        fps=6,
        codec="mpeg4",
        macro_block_size=1,
    )
    writer.send(None)
    for frame_index in range(frame_count):
        frame = np.full((6, 8, 3), frame_index * 15, dtype=np.uint8)
        writer.send(frame)
    writer.close()


def _write_on_off_video(path: Path) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path), size=(8, 6), fps=6, codec="mpeg4", macro_block_size=1
    )
    writer.send(None)
    for frame_index in range(12):
        brightness = 0 if frame_index < 6 else 120
        writer.send(np.full((6, 8, 3), brightness, dtype=np.uint8))
    writer.close()


def test_discover_stages_from_filenames(tmp_path: Path):
    for name in ["sample_stage_02_a.png", "sample_stage_01_a.png", "sample_stage_02_b.png"]:
        Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(tmp_path / name)
    stages = discover_stages(tmp_path)
    assert [stage.name for stage in stages] == ["Stage 01", "Stage 02"]
    assert len(stages[1].paths) == 2


def test_video_frames_folder_uses_video_name_and_avoids_existing_data(tmp_path: Path):
    video_path = tmp_path / "experiment.mp4"
    expected = tmp_path / "experiment Frames"

    assert video_frames_folder(video_path) == expected

    expected.mkdir()
    (expected / "stage_001.png").touch()
    assert video_frames_folder(video_path) == tmp_path / "experiment Frames (2)"


def test_bright_frame_selection_keeps_only_values_above_the_mean():
    brightness = [100.0, 100.1, 100.2, 100.3]

    assert _bright_frame_indices(brightness) == [2, 3]


def test_bright_frame_selection_keeps_uniform_video_frames():
    assert _bright_frame_indices([42.0, 42.0, 42.0]) == [0, 1, 2]


def test_delay_folder_analysis_subtracts_shared_background_and_exports_chart(tmp_path: Path):
    parent = tmp_path / "delays"
    early_folder = parent / "2 ms delay"
    late_folder = parent / "500 us delay"
    early_folder.mkdir(parents=True)
    late_folder.mkdir()
    Image.fromarray(np.array([[12, 8], [0, 0]], dtype=np.uint8)).save(early_folder / "a.png")
    Image.fromarray(np.array([[20, 12], [0, 0]], dtype=np.uint8)).save(early_folder / "b.png")
    Image.fromarray(np.array([[10, 10], [0, 0]], dtype=np.uint8)).save(late_folder / "a.png")
    background = np.array([[2, 2], [2, 2]], dtype=np.float64)

    rows = analyze_delay_folders(parent, background)

    assert [row["Time_Delay"] for row in rows] == [0.0005, 0.002]
    assert rows[0]["Mean_Intensity"] == 8.0
    assert rows[0]["Images_Averaged"] == 1
    assert rows[1]["Mean_Intensity"] == 11.0
    workbook_path = export_delay_analysis(rows, tmp_path / "results.xlsx")

    import zipfile

    with zipfile.ZipFile(workbook_path) as workbook:
        assert "xl/charts/chart1.xml" in workbook.namelist()
        shared_strings = workbook.read("xl/sharedStrings.xml").decode("utf-8")
        assert "Mean_Intensity" in shared_strings


def test_parse_time_delay_requires_unit():
    from esi_kinetics.io import parse_time_delay

    with pytest.raises(ValueError, match="delay value and unit"):
        parse_time_delay("early delay")


def test_discover_stages_sorts_three_digit_video_stages_numerically(tmp_path: Path):
    for index in (100, 11, 2, 99, 10):
        Image.fromarray(np.full((2, 2), index, dtype=np.uint8)).save(
            tmp_path / f"stage_{index:03d}.png"
        )

    stages = discover_stages(tmp_path)

    assert [stage.name for stage in stages] == [
        "Stage 02", "Stage 10", "Stage 11", "Stage 99", "Stage 100"
    ]


def test_load_experiment_combines_each_stage(tmp_path: Path):
    for index in range(3):
        Image.fromarray(np.full((2, 2), index + 1, dtype=np.uint8)).save(tmp_path / f"stage_01_{index}.png")
    experiment = load_experiment(tmp_path)
    np.testing.assert_array_equal(experiment.combined["Stage 01"], [[6, 6], [6, 6]])


def test_export_results_uses_explicit_metric_column_names(tmp_path: Path):
    results = [
        process_image("Stage 01", np.array([[0, 12], [8, 0]], dtype=np.float64), PipelineSettings()),
        process_image("Stage 02", np.array([[0, 18], [12, 0]], dtype=np.float64), PipelineSettings()),
    ]
    csv_path = export_results(results, tmp_path)
    exported = pd.read_csv(csv_path)

    assert exported.columns.tolist() == [
        "stage",
        "Integrated_Brightness",
        "Active_Pixels",
        "Peak_Intensity",
        "Mean_Intensity",
        "Background_Correction",
    ]
    assert exported.loc[0, "Integrated_Brightness"] == 20
    assert exported.loc[0, "Mean_Intensity"] == 10
    assert exported.loc[0, "Background_Correction"] == "threshold"
    average = pd.read_csv(tmp_path / "mean_intensity_average.csv")
    assert average.columns.tolist() == ["Average_Mean_Intensity"]
    assert len(average) == 1
    assert average.loc[0, "Average_Mean_Intensity"] == np.mean(
        exported["Mean_Intensity"]
    )


def test_exported_box_mask_tiff_keeps_full_dimensions(tmp_path: Path):
    image = np.arange(30, dtype=np.float64).reshape(5, 6)
    result = process_image(
        "Stage 01",
        image,
        PipelineSettings(
            crop=Crop(2, 1, 5, 4),
            zero_outside_crop=True,
            enable_background=False,
            enable_connected_region=False,
        ),
    )

    export_results([result], tmp_path)
    with Image.open(tmp_path / "Stage_01_processed.tiff") as output:
        exported_image = np.asarray(output)

    assert exported_image.shape == image.shape
    assert np.all(exported_image[:1] == 0)
    assert np.all(exported_image[4:] == 0)
    assert np.all(exported_image[:, :2] == 0)
    assert np.all(exported_image[:, 5:] == 0)
    np.testing.assert_array_equal(exported_image[1:4, 2:5], image[1:4, 2:5])


def test_extract_video_frames_samples_evenly_and_creates_stages(tmp_path: Path):
    video_path = tmp_path / "test_video.mp4"
    output_folder = tmp_path / "extracted"
    _write_test_video(video_path)

    output_paths = extract_video_frames(video_path, output_folder)
    experiment = load_experiment(output_folder)

    assert 0 < len(output_paths) < 100
    assert output_paths[-1].name == "stage_100.png"
    assert len(experiment.stages) == len(output_paths)
    assert experiment.stages[-1].name == "Stage 100"
    assert all(image.shape == (6, 8) for image in experiment.combined.values())
    assert experiment.combined[experiment.stages[0].name].mean() < experiment.combined["Stage 100"].mean()


def test_extract_video_frames_discards_dimmer_group(tmp_path: Path):
    video_path = tmp_path / "on_off_video.mp4"
    output_folder = tmp_path / "extracted"
    _write_on_off_video(video_path)

    output_paths = extract_video_frames(video_path, output_folder, frame_count=12)

    assert len(output_paths) == 6
    assert all(int(path.stem.split("_")[1]) >= 7 for path in output_paths)
    assert all(np.asarray(Image.open(path)).mean() > 100 for path in output_paths)


def test_extract_video_batch_creates_sibling_folders_for_all_videos(tmp_path: Path):
    first_video = tmp_path / "sample_500us.mp4"
    second_video = tmp_path / "sample_2ms.mp4"
    _write_test_video(first_video)
    _write_on_off_video(second_video)
    progress = []

    completed, failures = extract_video_batch(
        [first_video, second_video],
        frame_count=12,
        progress_callback=lambda path, index, total: progress.append((path.name, index, total)),
    )

    assert failures == []
    assert [folder.name for folder, _ in completed] == [
        "sample_500us Frames", "sample_2ms Frames"
    ]
    assert all(count > 0 for _, count in completed)
    assert progress == [(first_video.name, 1, 2), (second_video.name, 2, 2)]


def test_extract_video_frames_rejects_nonempty_destination(tmp_path: Path):
    video_path = tmp_path / "test_video.mp4"
    output_folder = tmp_path / "extracted"
    _write_test_video(video_path)
    output_folder.mkdir()
    (output_folder / "keep.txt").write_text("untouched")

    with pytest.raises(ValueError, match="empty folder"):
        extract_video_frames(video_path, output_folder, frame_count=5)

    assert (output_folder / "keep.txt").read_text() == "untouched"


def test_reference_profile_round_trips_masks(tmp_path: Path):
    signal_mask = np.zeros((5, 6), dtype=bool)
    signal_mask[2, 2:4] = True
    background_reference = np.arange(30, dtype=np.float64).reshape(5, 6)
    profile_path = tmp_path / "background_reference.npz"

    save_reference_profile(signal_mask, background_reference, profile_path)
    loaded_signal, loaded_background = load_reference_profile(profile_path)

    np.testing.assert_array_equal(loaded_signal, signal_mask)
    np.testing.assert_array_equal(loaded_background, background_reference)


def test_reference_profile_rejects_non_finite_background_values(tmp_path: Path):
    signal_mask = np.zeros((5, 6), dtype=bool)
    signal_mask[2, 2] = True
    background_reference = np.ones((5, 6), dtype=np.float64)
    background_reference[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        save_reference_profile(signal_mask, background_reference, tmp_path / "invalid.npz")
