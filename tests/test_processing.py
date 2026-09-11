import numpy as np
import pytest

from esi_kinetics.models import Crop, PipelineSettings
from esi_kinetics.processing import combine_frames, process_image


def test_combine_frames_uses_float64_and_preserves_100_frame_sum():
    frames = [np.full((2, 2), 255, dtype=np.uint8) for _ in range(100)]
    result = combine_frames(frames)
    assert result.dtype == np.float64
    assert np.all(result == 25500)


def test_combine_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching dimensions"):
        combine_frames([np.zeros((2, 2)), np.zeros((3, 2))])


def test_threshold_crop_and_brightness():
    image = np.array([[1, 5, 10], [2, 20, 3]], dtype=np.float32)
    settings = PipelineSettings(
        crop=Crop(1, 0, 3, 2),
        enable_crop=True,
        threshold=5,
    )
    result = process_image("stage-1", image, settings)
    np.testing.assert_array_equal(result.image, [[0, 10], [20, 0]])
    assert result.brightness == 30
    assert result.active_pixels == 2


def test_background_subtraction_clips_negative_values():
    image = np.array([[2, 10]], dtype=np.float64)
    settings = PipelineSettings(background_mode="subtract", background_value=5)
    result = process_image("stage-1", image, settings)
    np.testing.assert_array_equal(result.image, [[0, 5]])
    assert result.brightness == 5
