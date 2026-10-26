import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from esi_kinetics.models import Crop
from esi_kinetics.ui import MainWindow

APP = QApplication.instance() or QApplication([])


def test_saved_crop_overlay_uses_coordinates_of_cropped_preview():
    window = MainWindow()
    window.preview.set_image(np.zeros((60, 60), dtype=np.float64))
    window.crop_enabled.setChecked(True)
    window.crop_selection_complete = True
    window._crop_view_origin = (20, 30)
    window.batch_crop = Crop(20, 30, 80, 90)
    window.preview.set_crop_mode(False)
    window.show_crop_overlay.setChecked(True)

    assert window.preview._selection == (0, 0, 60, 60)
    assert window.preview._selection_visible
    assert not window.preview._crop_mode
    window.close()


def test_crop_reference_image_is_not_replaced_by_loaded_experiment_preview():
    window = MainWindow()
    image = np.zeros((30, 40), dtype=np.float64)
    window.preview.set_image(image)
    window._crop_reference_image_loaded = True
    window.experiment = object()

    window.apply_crop_selection(5, 6, 20, 25)

    assert window.crop_selection_complete
    assert window._crop_view_origin == (0, 0)
    assert window.preview._source_image.shape == image.shape
    window.close()


def test_hiding_saved_crop_overlay_reenables_drawing_without_changing_saved_crop():
    window = MainWindow()
    window.preview.set_image(np.zeros((40, 50), dtype=np.float64))
    window._crop_reference_image_loaded = True
    window.batch_crop = Crop(10, 8, 30, 28)

    window.show_crop_overlay.setChecked(True)
    assert window.preview._selection == (10, 8, 30, 28)
    assert window.preview._selection_visible
    assert not window.preview._crop_mode

    window.show_crop_overlay.setChecked(False)
    assert not window.preview._selection_visible
    assert window.preview._crop_mode
    assert window.batch_crop == Crop(10, 8, 30, 28)

    window.show_crop_overlay.setChecked(True)
    assert window.preview._selection == (10, 8, 30, 28)
    assert window.preview._selection_visible
    window.close()


def test_imported_reference_image_allows_crop_coordinates_above_spinbox_default(monkeypatch):
    image = np.zeros((300, 400), dtype=np.float64)
    monkeypatch.setattr(
        "esi_kinetics.ui.QFileDialog.getOpenFileName",
        lambda *args: ("reference.png", ""),
    )
    monkeypatch.setattr("esi_kinetics.ui.read_image", lambda path: image)
    window = MainWindow()

    window.import_selected_image()
    window.apply_crop_selection(120, 100, 350, 250)
    window.batch_crop = Crop(
        window.x0.value(), window.y0.value(), window.x1.value(), window.y1.value()
    )
    window.show_crop_overlay.setChecked(True)

    assert window.x0.value() == 120
    assert window.y0.value() == 100
    assert window.x1.value() == 350
    assert window.y1.value() == 250
    assert window.preview._selection == (120, 100, 350, 250)
    window.close()


def test_imported_reference_image_preview_preserves_png_brightness(monkeypatch):
    image = np.array([[32, 70], [128, 200]], dtype=np.float64)
    monkeypatch.setattr(
        "esi_kinetics.ui.QFileDialog.getOpenFileName",
        lambda *args: ("reference.png", ""),
    )
    monkeypatch.setattr("esi_kinetics.ui.read_image", lambda path: image)
    window = MainWindow()

    window.import_selected_image()

    assert window.preview._image.pixelColor(1, 0).red() == 70
    assert window.preview._image.pixelColor(1, 1).red() == 200
    window.close()


def test_saved_crop_zooms_preview_and_hiding_restores_full_photo():
    window = MainWindow()
    window.preview.set_image(np.zeros((200, 300), dtype=np.float64))
    window._crop_reference_image_loaded = True
    window.batch_crop = Crop(100, 60, 180, 140)

    window.show_crop_overlay.setChecked(True)
    zoomed = window.preview._view_rect
    assert zoomed.x() < 100
    assert zoomed.y() < 60
    assert zoomed.x() + zoomed.width() > 180
    assert zoomed.y() + zoomed.height() > 140
    assert zoomed.width() < 300
    assert zoomed.height() < 200

    window.show_crop_overlay.setChecked(False)
    assert window.preview._view_rect == window.preview._image.rect()
    assert window.preview._crop_mode
    window.close()


def test_reference_photo_segments_can_be_selected_for_final_analysis():
    window = MainWindow()
    image = np.ones((20, 20), dtype=np.float64)
    image[8:12, 8:12] = 100
    window.preview.set_image(image)
    window._crop_reference_image_loaded = True
    window.scan_segments_enabled.setChecked(True)

    assert window._segment_scan_labels is not None
    window.select_segment(9, 9)

    assert window._selected_signal_mask is not None
    assert int(window._selected_signal_mask.sum()) == 16
    assert window._selected_signal_mask[9, 9]
    window.close()


def test_final_analysis_requires_both_options_and_exports_averaged_folder(tmp_path, monkeypatch):
    window = MainWindow()
    image_folder = tmp_path / "Averaged Selected Frames"
    image_folder.mkdir()
    Image.fromarray(np.array([[10, 10], [10, 30]], dtype=np.uint8)).save(
        image_folder / "sample_500us.png"
    )
    workbook = tmp_path / "analysis.xlsx"
    dialogs = []
    monkeypatch.setattr(
        "esi_kinetics.ui.QFileDialog.getExistingDirectory",
        lambda *args: (dialogs.append("folder") or str(image_folder)),
    )
    monkeypatch.setattr(
        "esi_kinetics.ui.QFileDialog.getSaveFileName",
        lambda *args: (dialogs.append("save") or str(workbook), ""),
    )
    monkeypatch.setattr("esi_kinetics.ui.QMessageBox.information", lambda *args: None)

    window.apply_crop_enabled.setEnabled(True)
    window.batch_crop = Crop(0, 0, 2, 2)
    window.reference_background = np.full((2, 2), 10, dtype=np.float64)
    window._selected_signal_mask = np.zeros((2, 2), dtype=bool)
    window._selected_signal_mask[1, 1] = True

    window.analyze_cropped_delay_images()
    assert dialogs == []

    window.apply_crop_enabled.setChecked(True)
    window.subtract_saved_background_enabled.setEnabled(True)
    window.subtract_saved_background_enabled.setChecked(True)
    window.analyze_cropped_delay_images()

    assert workbook.is_file()
    assert dialogs == ["folder", "save"]
    assert "1 time delays" in window.summary.text()
    window.close()