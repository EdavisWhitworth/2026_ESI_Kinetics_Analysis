from pathlib import Path

import numpy as np
from PIL import Image

from esi_kinetics.io import discover_stages, load_experiment


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
