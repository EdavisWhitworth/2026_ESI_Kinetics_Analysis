from pathlib import Path

import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QDoubleSpinBox, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
    QScrollArea,
)

from .io import (
    export_results,
    load_experiment,
    load_reference_profile,
    save_reference_profile,
)
from .models import Crop, PipelineSettings
from .processing import (
    connected_component_labels,
    crop_image,
    estimate_reference_background,
    process_image,
)


class ImagePreview(QWidget):
    cropSelected = Signal(int, int, int, int)
    segmentSelected = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self._image = QImage()
        self._source_image = np.empty((0, 0), dtype=np.float64)
        self._contrast = 1.0
        self._display_max = 1.0
        self._image_rect = None
        self._selection = None
        self._drag_start = None
        self._drag_current = None
        self._crop_mode = False
        self._segment_labels = None
        self._selected_segment = 0
        self._reference_signal_mask = None

    def set_image(self, image: np.ndarray, display_max: float | None = None) -> None:
        self._source_image = image.astype(np.float64, copy=True)
        self._display_max = max(float(display_max if display_max is not None else image.max()), 1.0)
        self._render_image()
        self._selection = None
        self.update()

    def set_segment_labels(self, labels: np.ndarray | None, selected: int = 0) -> None:
        self._segment_labels = labels
        self._selected_segment = selected
        if self._source_image.size:
            self._render_image()
        self.update()

    def set_reference_masks(self, signal_mask: np.ndarray | None) -> None:
        self._reference_signal_mask = signal_mask
        if self._source_image.size:
            self._render_image()
        self.update()

    def set_contrast(self, contrast: float) -> None:
        self._contrast = contrast
        if self._source_image.size:
            self._render_image()
            self.update()

    def _render_image(self) -> None:
        image = self._source_image
        scaled = np.clip(image / self._display_max, 0, 1)
        scaled = np.clip((scaled - 0.5) * self._contrast + 0.5, 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
        if self._segment_labels is not None and self._segment_labels.shape == scaled.shape:
            rgb = np.repeat(scaled[:, :, None], 3, axis=2)
            candidates = self._segment_labels > 0
            rgb[candidates, 0] = 255
            rgb[candidates, 1] = (scaled[candidates] // 2)
            rgb[candidates, 2] = (scaled[candidates] // 2)
            selected = (self._selected_segment > 0) & (self._segment_labels == self._selected_segment)
            rgb[selected, 0] = 255
            rgb[selected, 1] = 255
            rgb[selected, 2] = 0
            if self._reference_signal_mask is not None and self._reference_signal_mask.shape == scaled.shape:
                rgb[self._reference_signal_mask, 0] = 255
                rgb[self._reference_signal_mask, 1] = 255
                rgb[self._reference_signal_mask, 2] = 0
            self._image = QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888,
            ).copy()
        else:
            rgb = np.repeat(scaled[:, :, None], 3, axis=2)
            if self._reference_signal_mask is not None and self._reference_signal_mask.shape == scaled.shape:
                rgb[self._reference_signal_mask, 0] = 255
                rgb[self._reference_signal_mask, 1] = 255
                rgb[self._reference_signal_mask, 2] = 0
            self._image = QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888,
            ).copy()

    def set_selection(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._selection = (x0, y0, x1, y1)
        self.update()

    def set_crop_mode(self, enabled: bool) -> None:
        self._crop_mode = enabled
        self._selection = None
        self._drag_start = None
        self._drag_current = None
        self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image.isNull():
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Import an image folder to begin")
            self._image_rect = None
            return
        self._image_rect = self._display_rect()
        painter.drawImage(self._image_rect, self._image)
        selection = self._drag_current or (self._selection if self._crop_mode else None)
        if selection:
            x0, y0, x1, y1 = selection
            scale_x = self._image_rect.width() / self._image.width()
            scale_y = self._image_rect.height() / self._image.height()
            rectangle = self._image_rect.adjusted(
                round(x0 * scale_x), round(y0 * scale_y),
                -round((self._image.width() - x1) * scale_x),
                -round((self._image.height() - y1) * scale_y),
            )
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            painter.drawRect(rectangle)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        point = event.position().toPoint()
        image_rect = self._display_rect()
        if (event.button() == Qt.MouseButton.LeftButton and self._segment_labels is not None
                and image_rect.contains(point)):
            x, y = self._image_point(point)
            label = int(self._segment_labels[y, x])
            if label:
                self.segmentSelected.emit(x, y)
                event.accept()
                return
        if (self._crop_mode and event.button() == Qt.MouseButton.LeftButton
                and image_rect.contains(point)):
            self._image_rect = image_rect
            self._drag_start = self._image_point(point)
            self._drag_current = self._normalized_selection(
                self._drag_start,
                (self._drag_start[0] + 1, self._drag_start[1] + 1),
            )
            self.grabMouse()
            self.update()
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._crop_mode and self._drag_start:
            current = self._image_point(event.position().toPoint())
            self._drag_current = self._normalized_selection(self._drag_start, current)
            self.update()
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._crop_mode and event.button() == Qt.MouseButton.LeftButton and self._drag_start:
            selection = self._normalized_selection(self._drag_start, self._image_point(event.position().toPoint()))
            self._drag_start = None
            self._drag_current = None
            self.releaseMouse()
            if selection[2] > selection[0] and selection[3] > selection[1]:
                self._selection = selection
                self.cropSelected.emit(*selection)
            self.update()
            event.accept()
            return
        event.ignore()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._image.isNull():
            self._image_rect = self._display_rect()

    def _display_rect(self) -> QRect:
        display_size = self._image.size()
        display_size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - display_size.width()) // 2
        y = (self.height() - display_size.height()) // 2
        return QRect(x, y, display_size.width(), display_size.height())

    def _image_point(self, point) -> tuple[int, int]:
        x = (point.x() - self._image_rect.left()) * self._image.width() / self._image_rect.width()
        y = (point.y() - self._image_rect.top()) * self._image.height() / self._image_rect.height()
        return (
            max(0, min(self._image.width(), round(x))),
            max(0, min(self._image.height(), round(y))),
        )

    @staticmethod
    def _normalized_selection(start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
        return (min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1]))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ESI Kinetics Analysis")
        self.resize(1100, 700)
        self.setAcceptDrops(True)
        self.experiment = None
        self.results = []
        self.reference_signal_mask = None
        self.reference_background = None
        self._crop_view_origin = (0, 0)
        self._preview_stage = None
        self._segment_scan_key = None
        self._segment_scan_labels = None
        self.stage_selector = QComboBox()
        self.stage_selector.currentIndexChanged.connect(self.refresh_preview)
        self.preview = ImagePreview()
        self.preview.cropSelected.connect(self.apply_crop_selection)
        self.preview.segmentSelected.connect(self.select_segment)
        self.summary = QLabel("No experiment loaded")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1_000_000_000)
        self.threshold.setValue(0)
        self.background = QDoubleSpinBox()
        self.background.setRange(0, 1_000_000_000)
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.1, 10.0)
        self.contrast.setSingleStep(0.1)
        self.contrast.setValue(1.0)
        self.x0 = QSpinBox()
        self.y0 = QSpinBox()
        self.x1 = QSpinBox()
        self.y1 = QSpinBox()
        for widget in (
            self.threshold, self.background, self.contrast,
            self.x0, self.y0, self.x1, self.y1,
        ):
            widget.setKeyboardTracking(False)
        self.crop_enabled = QCheckBox("Crop image (drag to select)")
        self.crop_selection_complete = False
        self.zero_outside_box = QCheckBox("Set pixels outside selected box to 0")
        self.zero_outside_box.setToolTip(
            "Keep the original image dimensions but zero all pixels outside the selected rectangle."
        )
        self.zero_outside_box.setEnabled(False)
        self.background_enabled = QCheckBox("Remove background")
        self.background_enabled.setChecked(True)
        self.connected_region_enabled = QCheckBox("Use largest connected bright region")
        self.connected_region_enabled.setChecked(False)
        self.scan_segments_enabled = QCheckBox("Find bright segments above background average")
        self.segment_offset = QDoubleSpinBox()
        self.segment_offset.setRange(0, 1_000_000_000)
        self.segment_offset.setSingleStep(1)
        self.segment_offset.setValue(0)
        self.segment_offset.setKeyboardTracking(False)
        self.isolate_segment_enabled = QCheckBox("Keep selected segment only")
        self.region_mask = None
        self.selected_segment = 0
        self.use_reference_profile = QCheckBox("Subtract saved background reference")
        self.use_reference_profile.setToolTip(
            "Subtract the saved reference background pixel by pixel; values at or below the reference become zero."
        )
        self.use_reference_profile.setEnabled(False)
        self.combine_enabled = QCheckBox("Use combined stage images")
        self.combine_enabled.setChecked(True)
        self.mode = QComboBox()
        self.mode.addItems(["Threshold mask", "Subtract background"])
        self._build_ui()

    def _build_ui(self) -> None:
        import_button = QPushButton("Import image folder")
        import_button.clicked.connect(self.import_folder)
        export_button = QPushButton("Export results")
        export_button.clicked.connect(self.export)
        reset_button = QPushButton("Reset processing")
        reset_button.clicked.connect(self.reset)
        save_profile_button = QPushButton("Save background image")
        save_profile_button.clicked.connect(self.save_profile)
        load_profile_button = QPushButton("Load background image")
        load_profile_button.clicked.connect(self.load_profile)
        controls = QFormLayout()
        controls.addRow(import_button)
        controls.addRow("Stage", self.stage_selector)
        controls.addRow(self.combine_enabled)
        controls.addRow(self.crop_enabled)
        controls.addRow("Crop x0 / x1", self._pair(self.x0, self.x1))
        controls.addRow("Crop y0 / y1", self._pair(self.y0, self.y1))
        controls.addRow(self.zero_outside_box)
        controls.addRow(self.background_enabled)
        controls.addRow(self.connected_region_enabled)
        controls.addRow(self.scan_segments_enabled)
        controls.addRow("Segment brightness above average", self.segment_offset)
        controls.addRow(self.isolate_segment_enabled)
        controls.addRow(self.use_reference_profile)
        controls.addRow(self._pair(save_profile_button, load_profile_button))
        controls.addRow("Background mode", self.mode)
        controls.addRow("Threshold", self.threshold)
        controls.addRow("Background value", self.background)
        controls.addRow("Preview contrast", self.contrast)
        controls.addRow(reset_button)
        controls.addRow(export_button)
        self.crop_enabled.stateChanged.connect(self.crop_mode_changed)
        self.zero_outside_box.stateChanged.connect(self.refresh_preview)
        self.background_enabled.stateChanged.connect(self.refresh_preview)
        self.connected_region_enabled.stateChanged.connect(self.refresh_preview)
        self.scan_segments_enabled.stateChanged.connect(self.segment_scan_changed)
        self.segment_offset.valueChanged.connect(self.segment_scan_changed)
        self.isolate_segment_enabled.stateChanged.connect(self.refresh_preview)
        self.use_reference_profile.stateChanged.connect(self.reference_profile_changed)
        self.mode.currentIndexChanged.connect(self.refresh_preview)
        self.contrast.valueChanged.connect(self.preview.set_contrast)
        for widget in (self.threshold, self.background, self.x0, self.y0, self.x1, self.y1):
            widget.valueChanged.connect(self.refresh_preview)
        self.summary.setWordWrap(True)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(controls)
        left_layout.addWidget(self.summary)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(left)
        splitter = QSplitter()
        splitter.addWidget(control_scroll)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    @staticmethod
    def _pair(first: QWidget, second: QWidget) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(first)
        layout.addWidget(second)
        return widget

    def import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self._load_folder(Path(folder))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if any(url.isLocalFile() and Path(url.toLocalFile()).is_dir()
               for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        folders = [Path(url.toLocalFile()) for url in event.mimeData().urls()
                   if url.isLocalFile() and Path(url.toLocalFile()).is_dir()]
        if not folders:
            event.ignore()
            return
        self._load_folder(folders[0])
        event.acceptProposedAction()

    def _load_folder(self, folder: Path) -> None:
        try:
            self.experiment = load_experiment(folder)
            self._crop_view_origin = (0, 0)
            self._preview_stage = None
            self._clear_segment_scan_cache()
            self.stage_selector.blockSignals(True)
            self.stage_selector.clear()
            self.stage_selector.addItems([stage.name for stage in self.experiment.stages])
            height, width = next(iter(self.experiment.combined.values())).shape
            self.region_mask = None
            self.selected_segment = 0
            if self.reference_background is not None and self.reference_background.shape != (height, width):
                self.reference_background = None
                self.use_reference_profile.blockSignals(True)
                self.use_reference_profile.setChecked(False)
                self.use_reference_profile.blockSignals(False)
                self.use_reference_profile.setEnabled(False)
                QMessageBox.warning(
                    self,
                    "Reference dimensions differ",
                    "This folder's images do not match the loaded background reference dimensions. "
                    "Reference subtraction was disabled.",
                )
            for spinbox, maximum, value in ((self.x0, width, 0), (self.y0, height, 0),
                                            (self.x1, width, width), (self.y1, height, height)):
                spinbox.setRange(0, maximum)
                spinbox.setValue(value)
            self.stage_selector.blockSignals(False)
            self._update_background_controls()
            self.refresh_preview()
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Import failed", str(error))

    def settings(self) -> PipelineSettings:
        mode = "threshold" if self.mode.currentIndex() == 0 else "subtract"
        use_profile = (
            self.use_reference_profile.isChecked()
            and self.reference_background is not None
        )
        return PipelineSettings(
            crop=Crop(self.x0.value(), self.y0.value(), self.x1.value(), self.y1.value()),
            background_mode=mode,
            threshold=self.threshold.value(),
            background_value=self.background.value(),
            enable_combine=self.combine_enabled.isChecked(),
            enable_crop=(
                self.crop_enabled.isChecked()
                and self.crop_selection_complete
                and not self.zero_outside_box.isChecked()
            ),
            enable_background=self.background_enabled.isChecked() and not use_profile,
            enable_connected_region=(
                self.connected_region_enabled.isChecked() and not self.scan_segments_enabled.isChecked()
            ),
            region_mask=(
                self.region_mask
                if self.isolate_segment_enabled.isChecked() and not use_profile
                else None
            ),
            reference_background=self.reference_background if use_profile else None,
            enable_reference_background=use_profile,
            zero_outside_crop=(
                self.zero_outside_box.isChecked() and self.crop_selection_complete
            ),
        )

    def _update_background_controls(self) -> None:
        enabled = not self.use_reference_profile.isChecked()
        for widget in (self.background_enabled, self.mode, self.threshold, self.background):
            widget.setEnabled(enabled)

    def reference_profile_changed(self) -> None:
        self._update_background_controls()
        self.refresh_preview()

    def save_profile(self) -> None:
        if self.experiment is None or self.region_mask is None:
            QMessageBox.information(
                self,
                "Reference profile incomplete",
                "Select the target segment before saving its background reference.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save background image",
            "background_reference.npz",
            "Background reference (*.npz)",
        )
        if not path:
            return
        profile_path = Path(path)
        if profile_path.suffix.lower() != ".npz":
            profile_path = profile_path.with_suffix(".npz")
        try:
            stage = self.stage_selector.currentText()
            reference_background = estimate_reference_background(
                self.experiment.combined[stage], self.region_mask
            )
            save_reference_profile(self.region_mask, reference_background, profile_path)
            self.reference_signal_mask = None
            self.reference_background = None
            self.isolate_segment_enabled.setChecked(False)
            self.scan_segments_enabled.setChecked(False)
            self.region_mask = None
            self.selected_segment = 0
            self.use_reference_profile.blockSignals(True)
            self.use_reference_profile.setChecked(False)
            self.use_reference_profile.setEnabled(False)
            self.use_reference_profile.blockSignals(False)
            self._update_background_controls()
            self.preview.set_reference_masks(None)
            self.refresh_preview()
            QMessageBox.information(
                self,
                "Background image saved",
                f"Saved background image to:\n{profile_path}\n\n"
                "The target highlight was cleared. Load this file later to subtract it.",
            )
        except (OSError, ValueError, np.linalg.LinAlgError) as error:
            QMessageBox.warning(self, "Cannot save profile", str(error))

    def load_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load background image", "", "Background reference (*.npz)"
        )
        if not path:
            return
        try:
            signal_mask, background_reference = load_reference_profile(Path(path))
            if self.experiment is not None:
                image_shape = next(iter(self.experiment.combined.values())).shape
                if signal_mask.shape != image_shape:
                    raise ValueError("Profile dimensions do not match the currently loaded images")
            self.reference_signal_mask = None
            self.reference_background = background_reference
            self.region_mask = None
            self.selected_segment = 0
            self.isolate_segment_enabled.setChecked(False)
            self.scan_segments_enabled.setChecked(False)
            self.preview.set_reference_masks(None)
            self.use_reference_profile.setEnabled(True)
            self.use_reference_profile.setChecked(True)
            self.refresh_preview()
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Cannot load profile", str(error))

    def apply_crop_selection(self, x0: int, y0: int, x1: int, y1: int) -> None:
        origin_x, origin_y = self._crop_view_origin
        x0 += origin_x
        x1 += origin_x
        y0 += origin_y
        y1 += origin_y
        for spinbox, value in ((self.x0, x0), (self.y0, y0), (self.x1, x1), (self.y1, y1)):
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        self.crop_selection_complete = True
        self.zero_outside_box.setEnabled(True)
        self._crop_view_origin = (x0, y0)
        self._clear_segment_scan_cache()
        self.refresh_preview()

    def crop_mode_changed(self) -> None:
        self.crop_selection_complete = False
        self._crop_view_origin = (0, 0)
        if not self.crop_enabled.isChecked():
            self.zero_outside_box.setChecked(False)
            self.zero_outside_box.setEnabled(False)
        self.preview.set_crop_mode(self.crop_enabled.isChecked())
        self.refresh_preview()

    def segment_scan_changed(self) -> None:
        if not self.use_reference_profile.isChecked():
            self.region_mask = None
        self.selected_segment = 0
        self._clear_segment_scan_cache()
        self.refresh_preview()

    def _clear_segment_scan_cache(self) -> None:
        self._segment_scan_key = None
        self._segment_scan_labels = None

    def _segment_labels_for(self, stage: str, image: np.ndarray, crop: Crop | None) -> np.ndarray:
        crop_key = (crop.x0, crop.y0, crop.x1, crop.y1) if crop is not None else None
        key = (stage, crop_key, self.segment_offset.value())
        if key != self._segment_scan_key or self._segment_scan_labels is None:
            self._segment_scan_labels = connected_component_labels(
                image, self.segment_threshold(image)
            )
            self._segment_scan_key = key
        return self._segment_scan_labels

    def segment_threshold(self, image: np.ndarray) -> float:
        return float(image.mean()) + self.segment_offset.value()

    def select_segment(self, x: int, y: int) -> None:
        if self.use_reference_profile.isChecked():
            self.use_reference_profile.setChecked(False)
        stage = self.stage_selector.currentText()
        combined = self.experiment.combined[stage]
        crop = self.settings().crop if self.crop_enabled.isChecked() and self.crop_selection_complete else None
        scan_image = crop_image(combined, crop) if crop is not None else combined
        labels = self._segment_labels_for(stage, scan_image, crop)
        label = int(labels[y, x])
        if not label:
            return
        mask = np.zeros_like(combined, dtype=bool)
        if crop is None:
            mask = labels == label
        else:
            mask[crop.y0:crop.y1, crop.x0:crop.x1] = labels == label
        self.region_mask = mask
        self.selected_segment = label
        self.refresh_preview()

    def refresh_preview(self) -> None:
        if self.experiment is None or self.stage_selector.currentIndex() < 0:
            return
        stage = self.stage_selector.currentText()
        if stage != self._preview_stage:
            if (not self.use_reference_profile.isChecked()
                    and not self.zero_outside_box.isChecked()):
                self._crop_view_origin = (0, 0)
                self.crop_selection_complete = False
            self._preview_stage = stage
            self._clear_segment_scan_cache()
        try:
            result = process_image(stage, self.experiment.combined[stage], self.settings())
        except ValueError as error:
            self.summary.setText(f"{stage}: {error}")
            return
        self.results = [result]
        self.summary.setText(
            f"{stage}: "
            f"{'saved-reference corrected; ' if result.settings.enable_reference_background else ''}"
            f"{result.brightness:,.3f} brightness; "
            f"mean {result.mean_intensity:,.3f}; "
            f"{result.active_pixels:,} active pixels; max {result.maximum:,.3f}"
        )
        combined = self.experiment.combined[stage]
        isolate_segment = self.isolate_segment_enabled.isChecked() and self.region_mask is not None
        use_profile = self.use_reference_profile.isChecked() and self.reference_background is not None
        image = result.image if (
            use_profile or isolate_segment
            or self.settings().zero_outside_crop
            or (self.crop_enabled.isChecked() and self.crop_selection_complete)
        ) else combined
        if use_profile:
            full_corrected_image = np.maximum(combined - self.reference_background, 0.0)
            display_max = max(float(full_corrected_image.max()), 1.0)
        else:
            display_max = max(float(combined.max()), 1.0)
        self.preview.set_image(image, display_max=display_max)
        self.preview.set_contrast(self.contrast.value())
        display_signal_mask = (
            self.region_mask
            if self.region_mask is not None
            and (self.isolate_segment_enabled.isChecked() or self.scan_segments_enabled.isChecked())
            and not use_profile
            else None
        )
        if self.crop_enabled.isChecked() and self.crop_selection_complete:
            crop = self.settings().crop
            if crop is not None:
                if display_signal_mask is not None:
                    display_signal_mask = crop_image(display_signal_mask, crop)
        self.preview.set_reference_masks(display_signal_mask)
        self.preview.set_segment_labels(None)
        if self.scan_segments_enabled.isChecked() and not isolate_segment and not use_profile:
            crop = self.settings().crop if self.crop_enabled.isChecked() and self.crop_selection_complete else None
            scan_image = crop_image(combined, crop) if crop is not None else combined
            labels = self._segment_labels_for(stage, scan_image, crop)
            self.preview.set_segment_labels(labels, self.selected_segment)
        if self.crop_enabled.isChecked() and not self.crop_selection_complete:
            self.preview.set_crop_mode(True)

    def reset(self) -> None:
        self.crop_enabled.setChecked(False)
        self.zero_outside_box.setChecked(False)
        self.zero_outside_box.setEnabled(False)
        self.background_enabled.setChecked(True)
        self.connected_region_enabled.setChecked(False)
        self.scan_segments_enabled.setChecked(False)
        self.segment_offset.setValue(0)
        self.isolate_segment_enabled.setChecked(False)
        self.use_reference_profile.setChecked(False)
        self.region_mask = None
        self.selected_segment = 0
        self._clear_segment_scan_cache()
        self.mode.setCurrentIndex(0)
        self.threshold.setValue(0)
        self.background.setValue(0)
        self.contrast.setValue(1.0)
        self.x0.setValue(0)
        self.y0.setValue(0)
        self.x1.setValue(self.x1.maximum())
        self.y1.setValue(self.y1.maximum())
        self.refresh_preview()

    def export(self) -> None:
        if self.experiment is None:
            QMessageBox.information(self, "Nothing to export", "Import a folder and process a stage first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select export folder")
        if folder:
            results = [process_image(stage.name, self.experiment.combined[stage.name], self.settings())
                       for stage in self.experiment.stages]
            export_results(results, Path(folder))
            QMessageBox.information(self, "Export complete", "Processed images and brightness_summary.csv were exported.")


def run() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
