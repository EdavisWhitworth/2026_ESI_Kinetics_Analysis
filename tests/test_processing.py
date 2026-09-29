import numpy as np
import pytest

from esi_kinetics.models import Crop, PipelineSettings
from esi_kinetics.processing import (
    combine_frames,
    connected_component_labels,
    estimate_reference_background,
    process_image,
)


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
    assert result.mean_intensity == 15.0


def test_background_subtraction_clips_negative_values():
    image = np.array([[2, 10]], dtype=np.float64)
    settings = PipelineSettings(background_mode="subtract", background_value=5)
    result = process_image("stage-1", image, settings)
    np.testing.assert_array_equal(result.image, [[0, 5]])
    assert result.brightness == 5
    assert result.active_pixels == 1
    assert result.mean_intensity == 5.0


def test_background_subtraction_uses_local_border_background():
    image = np.array([
        [10, 10, 10, 10, 10],
        [10, 20, 20, 20, 10],
        [10, 20, 80, 20, 10],
        [10, 20, 20, 20, 10],
        [10, 10, 10, 10, 10],
    ], dtype=np.float64)
    settings = PipelineSettings(background_mode="subtract", background_value=0)
    result = process_image("stage-1", image, settings)
    np.testing.assert_array_equal(result.image, np.array([
        [0, 0, 0, 0, 0],
        [0, 10, 10, 10, 0],
        [0, 10, 70, 10, 0],
        [0, 10, 10, 10, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float64))
    assert result.brightness == 150


def test_saved_background_mask_removes_spatial_gradient_from_target():
    y, x = np.mgrid[:7, :9]
    background = 10 + 2 * x + 3 * y
    image = background.astype(np.float64)
    image[2:5, 3:6] += 20
    signal_mask = np.zeros(image.shape, dtype=bool)
    signal_mask[2:5, 3:6] = True
    background_mask = np.ones(image.shape, dtype=bool)
    background_mask[1:6, 2:7] = False

    result = process_image(
        "Stage 01",
        image,
        PipelineSettings(
            crop=Crop(3, 2, 6, 5),
            enable_crop=True,
            enable_background=False,
            region_mask=signal_mask,
            background_mask=background_mask,
            enable_spatial_background=True,
        ),
    )

    np.testing.assert_allclose(result.image, np.full((3, 3), 20.0), atol=1e-10)
    assert result.brightness == pytest.approx(180.0)


def test_saved_background_reference_keeps_all_positive_excess_and_ignores_stale_target_mask():
    reference = np.array([[10, 20, 30], [15, 25, 35]], dtype=np.float64)
    current = np.array([[8, 24, 30], [20, 20, 45]], dtype=np.float64)
    signal_mask = np.array([[False, True, True], [True, True, False]])
    result = process_image(
        "Stage 02",
        current,
        PipelineSettings(
            enable_background=False,
            region_mask=signal_mask,
            reference_background=reference,
            enable_reference_background=True,
        ),
    )

    np.testing.assert_array_equal(result.image, [[0, 4, 0], [5, 0, 10]])
    assert result.brightness == 19


def test_zero_outside_box_preserves_full_image_and_zeros_external_pixels():
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

    assert result.image.shape == image.shape
    expected = np.zeros_like(image)
    expected[1:4, 2:5] = image[1:4, 2:5]
    np.testing.assert_array_equal(result.image, expected)
    assert result.brightness == pytest.approx(float(expected.sum()))


def test_zero_outside_box_does_not_change_corrected_pixels_inside_box():
    reference = np.full((5, 6), 10.0)
    image = reference.copy()
    image[1:4, 2:5] += np.arange(9, dtype=np.float64).reshape(3, 3)
    box = Crop(2, 1, 5, 4)
    full_result = process_image(
        "Stage 01",
        image,
        PipelineSettings(
            enable_background=False,
            reference_background=reference,
            enable_reference_background=True,
            enable_connected_region=False,
        ),
    )
    boxed_result = process_image(
        "Stage 01",
        image,
        PipelineSettings(
            crop=box,
            zero_outside_crop=True,
            enable_background=False,
            reference_background=reference,
            enable_reference_background=True,
            enable_connected_region=False,
        ),
    )

    np.testing.assert_array_equal(boxed_result.image[1:4, 2:5], full_result.image[1:4, 2:5])
    assert np.count_nonzero(boxed_result.image) == 8


def test_reference_background_is_fitted_from_all_non_selected_pixels():
    y, x = np.mgrid[:7, :9]
    expected_background = 10 + 2 * x + 3 * y
    first_image = expected_background.astype(np.float64)
    signal_mask = np.zeros(first_image.shape, dtype=bool)
    signal_mask[2:5, 3:6] = True
    first_image[signal_mask] += 100

    background_reference = estimate_reference_background(first_image, signal_mask)

    np.testing.assert_allclose(background_reference, expected_background, atol=1e-10)
    np.testing.assert_array_equal(background_reference[~signal_mask], first_image[~signal_mask])
    assert np.all(background_reference[signal_mask] < first_image[signal_mask])


def test_brightness_uses_only_largest_connected_bright_region():
    image = np.array([[10, 10, 0, 4], [0, 10, 0, 0]], dtype=np.float64)
    result = process_image("stage-1", image, PipelineSettings())
    np.testing.assert_array_equal(result.image, [[10, 10, 0, 0], [0, 10, 0, 0]])
    assert result.brightness == 30
    assert result.active_pixels == 3


def test_connected_region_filter_can_be_disabled():
    image = np.array([[10, 0, 4]], dtype=np.float64)
    settings = PipelineSettings(enable_connected_region=False)
    result = process_image("stage-1", image, settings)
    np.testing.assert_array_equal(result.image, image)
    assert result.brightness == 14


def test_connected_component_labels_find_regions_above_average():
    image = np.array([[0, 8, 8, 0], [0, 0, 0, 0], [0, 0, 0, 9]], dtype=np.float64)
    labels = connected_component_labels(image, image.mean())
    assert labels[0, 1] == labels[0, 2] > 0
    assert labels[2, 3] > 0
    assert labels[2, 3] != labels[0, 1]


def test_selected_region_mask_controls_brightness():
    image = np.array([[10, 10, 0, 4]], dtype=np.float64)
    settings = PipelineSettings(region_mask=np.array([[True, False, False, False]]))
    result = process_image("stage-1", image, settings)
    assert result.brightness == 10
