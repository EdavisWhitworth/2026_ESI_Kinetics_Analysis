from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from esi_kinetics.io import (
    discover_stages,
    export_results,
    load_experiment,
    load_reference_profile,
    save_reference_profile,
)
from esi_kinetics.models import Crop, PipelineSettings
from esi_kinetics.processing import process_image


def test_discover_stages_from_filenames(tmp_path: Path):
    for name in ["sample_stage_02_a.png", "sample_stage_01_a.png", "sample_stage_02_b.png"]:
        Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(tmp_path / name)
    stages = discover_stages(tmp_path)
    assert [stage.name for stage in stages] == ["Stage 01", "Stage 02"]
    assert len(stages[1].paths) == 2


def test_load_experiment_combines_each_stage(tmp_path: Path):
    for index in range(3):
        Image.fromarray(np.full((2, 2), index + 1, dtype=np.uint8)).save(tmp_path / f"stage_01_{index}.png")
    experiment = load_experiment(tmp_path)
    np.testing.assert_array_equal(experiment.combined["Stage 01"], [[6, 6], [6, 6]])


def test_export_results_uses_explicit_metric_column_names(tmp_path: Path):
    result = process_image("Stage 01", np.array([[0, 12], [8, 0]], dtype=np.float64), PipelineSettings())
    csv_path = export_results([result], tmp_path)
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
