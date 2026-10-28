from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QRect, Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QDoubleSpinBox, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
    QScrollArea,
    QStackedWidget,
)

from .io import (
    export_results,
    analyze_delay_folders,
    analyze_delay_images,
    analyze_delay_images_with_crop,
    average_selected_frame_folders,
    apply_crop_to_folder,
    export_delay_analysis,
    export_delay_curve_analysis,
    extract_video_batch,
    load_experiment,
    load_reference_profile,
    read_image,
    save_reference_profile,
)
from .models import Crop, PipelineSettings
from .processing import (
    connected_component_labels,
    crop_image,
    estimate_reference_background,
    process_image,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v", ".wmv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class DelayAnalysisWorker(QThread):
    analysisComplete = Signal(str, int)
    analysisFailed = Signal(str)

    def __init__(self, parent_folder: Path, background_reference: np.ndarray, workbook_path: Path) -> None:
        super().__init__()
        self.parent_folder = parent_folder
        self.background_reference = background_reference.copy()
        self.workbook_path = workbook_path

    def run(self) -> None:
        try:
            rows = analyze_delay_folders(self.parent_folder, self.background_reference)
            export_delay_analysis(rows, self.workbook_path)
            self.analysisComplete.emit(str(self.workbook_path), len(rows))
        except Exception as error:
            self.analysisFailed.emit(str(error))


class DelayImageAnalysisWorker(QThread):
    analysisComplete = Signal(str, int)
    analysisFailed = Signal(str)

    def __init__(self, image_folder: Path, workbook_path: Path) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.workbook_path = workbook_path

    def run(self) -> None:
        try:
            rows = analyze_delay_images(self.image_folder)
            export_delay_analysis(rows, self.workbook_path)
            self.analysisComplete.emit(str(self.workbook_path), len(rows))
        except Exception as error:
            self.analysisFailed.emit(str(error))


class VideoExtractionWorker(QThread):
    progress = Signal(str, int, int)
    extractionComplete = Signal(object)

    def __init__(self, video_paths: list[Path]) -> None:
        super().__init__()
        self.video_paths = video_paths

    def run(self) -> None:
        results, failures = extract_video_batch(
            self.video_paths,
            frame_count=100,
            progress_callback=lambda path, index, total: self.progress.emit(
                path.name, index, total
            ),
        )
        self.extractionComplete.emit((results, failures))


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
        self._view_rect = QRect()
        self._selection = None
        self._selection_visible = False
        self._drag_start = None
        self._drag_current = None
        self._crop_mode = False
        self._selection_dragging = False
        self._selection_drag_origin = None
        self._selection_drag_start = None
        self._segment_labels = None
        self._selected_segment = 0
        self._reference_signal_mask = None

    def set_image(self, image: np.ndarray, display_max: float | None = None) -> None:
        self._source_image = image.astype(np.float64, copy=True)
        self._display_max = max(float(display_max if display_max is not None else image.max()), 1.0)
        self._render_image()
        self._view_rect = QRect(0, 0, self._image.width(), self._image.height())
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

    def set_selection_visible(self, visible: bool) -> None:
        self._selection_visible = visible
        self.update()

    def set_view_crop(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._image.isNull() or x1 <= x0 or y1 <= y0:
            return
        margin_x = max(1, round((x1 - x0) * 0.08))
        margin_y = max(1, round((y1 - y0) * 0.08))
        left = max(0, x0 - margin_x)
        top = max(0, y0 - margin_y)
        right = min(self._image.width(), x1 + margin_x)
        bottom = min(self._image.height(), y1 + margin_y)
        self._view_rect = QRect(left, top, right - left, bottom - top)
        self.update()

    def set_view_full(self) -> None:
        if not self._image.isNull():
            self._view_rect = QRect(0, 0, self._image.width(), self._image.height())
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
        painter.drawImage(self._image_rect, self._image, self._view_rect)
        selection = self._drag_current or (
            self._selection if self._crop_mode or self._selection_visible else None
        )
        if selection:
            x0, y0, x1, y1 = selection
            scale_x = self._image_rect.width() / self._view_rect.width()
            scale_y = self._image_rect.height() / self._view_rect.height()
            local_x0 = x0 - self._view_rect.x()
            local_y0 = y0 - self._view_rect.y()
            local_x1 = x1 - self._view_rect.x()
            local_y1 = y1 - self._view_rect.y()
            rectangle = QRect(
                self._image_rect.x() + round(local_x0 * scale_x),
                self._image_rect.y() + round(local_y0 * scale_y),
                round((local_x1 - local_x0) * scale_x),
                round((local_y1 - local_y0) * scale_y),
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
        if (event.button() == Qt.MouseButton.LeftButton and image_rect.contains(point)):
            x, y = self._image_point(point)
            if (self._selection is not None and self._selection_visible and not self._crop_mode
                    and self._selection[0] <= x <= self._selection[2] and self._selection[1] <= y <= self._selection[3]):
                self._selection_dragging = True
                self._selection_drag_origin = (x, y)
                self._selection_drag_start = self._selection
                self.grabMouse()
                self.update()
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
        if self._selection_dragging and self._selection_drag_start is not None:
            current = self._image_point(event.position().toPoint())
            dx = current[0] - self._selection_drag_origin[0]
            dy = current[1] - self._selection_drag_origin[1]
            x0, y0, x1, y1 = self._selection_drag_start
            width = x1 - x0
            height = y1 - y0
            max_x = max(0, self._source_image.shape[1] - width)
            max_y = max(0, self._source_image.shape[0] - height)
            translated_x0 = min(max(0, x0 + dx), max_x)
            translated_y0 = min(max(0, y0 + dy), max_y)
            self._selection = (
                translated_x0,
                translated_y0,
                translated_x0 + width,
                translated_y0 + height,
            )
            self.update()
            event.accept()
            return
        if self._crop_mode and self._drag_start:
            current = self._image_point(event.position().toPoint())
            self._drag_current = self._normalized_selection(self._drag_start, current)
            self.update()
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._selection_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._selection_dragging = False
            self._selection_drag_origin = None
            self._selection_drag_start = None
            self.releaseMouse()
            if self._selection is not None:
                self.cropSelected.emit(*self._selection)
            self.update()
            event.accept()
            return
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
        display_size = self._view_rect.size()
        display_size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x = (self.width() - display_size.width()) // 2
        y = (self.height() - display_size.height()) // 2
        return QRect(x, y, display_size.width(), display_size.height())

    def _image_point(self, point) -> tuple[int, int]:
        x = self._view_rect.x() + (
            (point.x() - self._image_rect.left())
            * self._view_rect.width()
            / self._image_rect.width()
        )
        y = self._view_rect.y() + (
            (point.y() - self._image_rect.top())
            * self._view_rect.height()
            / self._image_rect.height()
        )
        return (
            max(self._view_rect.x(), min(self._view_rect.x() + self._view_rect.width(), round(x))),
            max(self._view_rect.y(), min(self._view_rect.y() + self._view_rect.height(), round(y))),
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
        self._video_worker = None
        self._delay_analysis_worker = None
        self._delay_image_analysis_worker = None
        self.batch_crop: Crop | None = None
        self.uploaded_background_image: np.ndarray | None = None
        self.uploaded_background_path: Path | None = None
        self._crop_view_origin = (0, 0)
        self._crop_reference_image_loaded = False
        self._crop_reference_path: Path | None = None
        self._reference_photo_original: np.ndarray | None = None
        self._selected_signal_mask: np.ndarray | None = None
        self._background_subtraction_enabled = False
        self._averaged_selected_folder: Path | None = None
        self._preview_stage = None
        self.show_crop_overlay = QCheckBox("Use saved crop for all following photos")
        self.apply_crop_enabled = QCheckBox("Apply saved crop")
        self.apply_crop_enabled.setEnabled(False)
        self.subtract_saved_background_enabled = QCheckBox("Subtract saved background")
        self.subtract_saved_background_enabled.setEnabled(False)
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
        self.control_mode = QComboBox()
        self.control_mode.addItems(["New controls", "Old controls"])
        self.control_mode.setCurrentIndex(0)
        self._build_ui()

    def _build_ui(self) -> None:
        import_video_button = QPushButton("Import videos (select multiple)")
        import_video_button.clicked.connect(self.import_video)
        self.import_video_button = import_video_button
        import_selected_image_button = QPushButton("Upload reference photo")
        import_selected_image_button.clicked.connect(self.import_selected_image)
        self.import_selected_image_button = import_selected_image_button
        average_selected_frames_button = QPushButton("Average selected frame folders")
        average_selected_frames_button.clicked.connect(self.average_selected_frame_folders)
        self.average_selected_frames_button = average_selected_frames_button
        self.cropped_image_analysis_button = QPushButton("Analyze averaged selected frames")
        self.cropped_image_analysis_button.clicked.connect(self.analyze_cropped_delay_images)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        delete_outside_crop_button = QPushButton("Delete outside crop")
        delete_outside_crop_button.clicked.connect(self.delete_outside_crop)
        upload_background_button = QPushButton("Upload background image")
        upload_background_button.clicked.connect(self.upload_background_image)
        save_cropped_reference_button = QPushButton("Save cropped photo")
        save_cropped_reference_button.clicked.connect(self.save_cropped_reference_image)
        analyze_saved_folder_button = QPushButton("Analyze saved cropped folder")
        analyze_saved_folder_button.clicked.connect(self.analyze_saved_cropped_photo_folder)
        controls_layout = QFormLayout()
        controls_layout.addRow("1. Upload video files", import_video_button)
        controls_layout.addRow("2. Average selected frame folders", average_selected_frames_button)
        controls_layout.addRow("3. Reference photo upload and crop", import_selected_image_button)
        controls_layout.addRow(self.show_crop_overlay)
        controls_layout.addRow(delete_outside_crop_button)
        controls_layout.addRow("4. Upload background", upload_background_button)
        controls_layout.addRow("5. Save cropped photo", save_cropped_reference_button)
        controls_layout.addRow("6. Analyze saved cropped folder", analyze_saved_folder_button)

        self.crop_enabled.stateChanged.connect(self.crop_mode_changed)
        self.show_crop_overlay.stateChanged.connect(self.update_crop_overlay)
        self.apply_crop_enabled.stateChanged.connect(self.update_crop_overlay)
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
        left_layout.addLayout(controls_layout)
        left_layout.addWidget(reset_button, alignment=Qt.AlignmentFlag.AlignLeft)
        left_layout.addStretch(1)

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

    def import_video(self) -> None:
        video_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select videos to extract",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.webm *.mpeg *.mpg *.m4v *.wmv);;All files (*)",
        )
        if not video_paths:
            return

        self._start_video_import([Path(video_path) for video_path in video_paths])

    def analyze_delay_folders(self) -> None:
        if self.reference_background is None:
            QMessageBox.information(
                self,
                "Load background first",
                "Load the saved background image before starting batch delay analysis.",
            )
            return
        parent_folder = QFileDialog.getExistingDirectory(
            self, "Select the folder containing all delay image folders"
        )
        if not parent_folder:
            return
        default_path = str(Path(parent_folder) / "delay_mean_intensity.xlsx")
        workbook_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save time delay vs mean intensity workbook",
            default_path,
            "Excel workbook (*.xlsx)",
        )
        if not workbook_path:
            return
        output_path = Path(workbook_path)
        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        try:
            self.batch_analysis_button.setEnabled(False)
            self.summary.setText("Analyzing delay folders with the loaded background...")
            self._delay_analysis_worker = DelayAnalysisWorker(
                Path(parent_folder), self.reference_background, output_path
            )
            self._delay_analysis_worker.analysisComplete.connect(self._delay_analysis_complete)
            self._delay_analysis_worker.analysisFailed.connect(self._delay_analysis_failed)
            self._delay_analysis_worker.start()
        except (OSError, ValueError, RuntimeError) as error:
            self.batch_analysis_button.setEnabled(True)
            QMessageBox.critical(self, "Batch analysis failed", str(error))

    def analyze_delay_images(self) -> None:
        image_folder = QFileDialog.getExistingDirectory(
            self, "Select the Highest Intensity Time Points folder"
        )
        if not image_folder:
            return
        default_path = str(Path(image_folder) / "highest_intensity_mean_intensity.xlsx")
        workbook_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save time delay vs mean intensity workbook",
            default_path,
            "Excel workbook (*.xlsx)",
        )
        if not workbook_path:
            return
        output_path = Path(workbook_path)
        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        try:
            self.selected_image_analysis_button.setEnabled(False)
            self.summary.setText("Analyzing one highest-intensity image per time delay...")
            self._delay_image_analysis_worker = DelayImageAnalysisWorker(
                Path(image_folder), output_path
            )
            self._delay_image_analysis_worker.analysisComplete.connect(
                self._delay_analysis_complete
            )
            self._delay_image_analysis_worker.analysisFailed.connect(
                self._delay_image_analysis_failed
            )
            self._delay_image_analysis_worker.start()
        except (OSError, ValueError, RuntimeError) as error:
            self.selected_image_analysis_button.setEnabled(True)
            QMessageBox.critical(self, "Image analysis failed", str(error))

    def save_batch_crop(self) -> None:
        if not self.crop_selection_complete:
            QMessageBox.information(
                self,
                "No crop selected",
                "Select a crop on the current preview before saving the fixed batch crop.",
            )
            return
        self.batch_crop = Crop(
            self.x0.value(),
            self.y0.value(),
            self.x1.value(),
            self.y1.value(),
        )
        self.apply_crop_enabled.setEnabled(True)
        self.show_crop_overlay.setChecked(True)
        self.preview.set_crop_mode(False)
        self.update_crop_overlay()
        QMessageBox.information(
            self,
            "Batch crop saved",
            "This crop will be applied to every image in the highest-intensity folder export.",
        )

    def update_crop_overlay(self) -> None:
        edit_crop = self.crop_enabled.isChecked() or self._crop_reference_image_loaded
        if self.batch_crop is None:
            self.preview.set_view_full()
            self.preview.set_selection_visible(False)
            self.preview.set_crop_mode(edit_crop)
            return
        if self.show_crop_overlay.isChecked():
            origin_x, origin_y = (0, 0)
            if (self.crop_enabled.isChecked() and self.crop_selection_complete
                    and not self.zero_outside_box.isChecked()):
                origin_x, origin_y = self._crop_view_origin
            if self._crop_reference_image_loaded:
                self.preview.set_view_crop(
                    self.batch_crop.x0,
                    self.batch_crop.y0,
                    self.batch_crop.x1,
                    self.batch_crop.y1,
                )
            self.preview.set_crop_mode(False)
            self.preview.set_selection(
                self.batch_crop.x0 - origin_x,
                self.batch_crop.y0 - origin_y,
                self.batch_crop.x1 - origin_x,
                self.batch_crop.y1 - origin_y,
            )
            self.preview.set_selection_visible(True)
        else:
            self.preview.set_selection_visible(False)
            self.preview.set_view_full()
            self.preview.set_crop_mode(edit_crop)

    def apply_crop_to_image_folder(self) -> None:
        if self.batch_crop is None:
            QMessageBox.information(
                self,
                "Batch crop required",
                "Set and save a crop before applying it to a folder of uncropped images.",
            )
            return
        source_folder = QFileDialog.getExistingDirectory(self, "Select uncropped image folder")
        if not source_folder:
            return
        output_folder = QFileDialog.getExistingDirectory(self, "Choose where to save the cropped images")
        if not output_folder:
            return
        try:
            applied = apply_crop_to_folder(Path(source_folder), self.batch_crop, Path(output_folder))
            self.summary.setText(f"Applied crop to {len(list(applied.iterdir()))} images.")
            QMessageBox.information(
                self,
                "Crop applied",
                f"Saved the cropped images to:\n{applied}",
            )
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.critical(self, "Crop application failed", str(error))

    def import_selected_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select one selected frame image",
            "",
            "Image files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        self.load_reference_image(Path(path))

    def upload_background_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select background image",
            "",
            "Image files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        try:
            background = read_image(Path(path))
            if self.preview._source_image.size and background.shape != self.preview._source_image.shape:
                raise ValueError("Background image dimensions must match the reference photo")
            self.uploaded_background_image = background
            self.uploaded_background_path = Path(path)
            self.summary.setText(f"Background uploaded: {Path(path).name}")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Cannot upload background", str(error))

    def load_reference_image(self, image_path: Path) -> None:
        image = read_image(image_path)
        saved_crop = self.batch_crop
        self._crop_reference_image_loaded = True
        self._crop_reference_path = image_path
        self._reference_photo_original = image.copy()
        self._selected_signal_mask = None
        self._segment_scan_labels = None
        self.crop_selection_complete = saved_crop is not None
        self.reference_background = None
        self.reference_signal_mask = None
        self.subtract_saved_background_enabled.setChecked(False)
        self.subtract_saved_background_enabled.setEnabled(False)
        self.apply_crop_enabled.setChecked(False)
        self.apply_crop_enabled.setEnabled(False)
        self.crop_enabled.setChecked(True)
        if saved_crop is not None:
            self.batch_crop = Crop(saved_crop.x0, saved_crop.y0, saved_crop.x1, saved_crop.y1)
        else:
            self.batch_crop = None
        height, width = image.shape[:2]
        for spinbox, maximum, value in (
            (self.x0, width, 0),
            (self.y0, height, 0),
            (self.x1, width, width),
            (self.y1, height, height),
        ):
            spinbox.setRange(0, maximum)
            spinbox.setValue(value)
        if saved_crop is not None:
            self.x0.setValue(saved_crop.x0)
            self.y0.setValue(saved_crop.y0)
            self.x1.setValue(saved_crop.x1)
            self.y1.setValue(saved_crop.y1)
        self._crop_view_origin = (0, 0)
        self._preview_stage = None
        self.preview.set_reference_masks(None)
        self.preview.set_segment_labels(None)
        self.preview.set_crop_mode(True)
        self.preview.set_image(image, display_max=255.0)
        self.preview.set_contrast(self.contrast.value())
        if self.batch_crop is not None and self.show_crop_overlay.isChecked():
            self.preview.set_crop_mode(False)
            self.preview.set_selection(
                self.batch_crop.x0,
                self.batch_crop.y0,
                self.batch_crop.x1,
                self.batch_crop.y1,
            )
            self.preview.set_selection_visible(True)
        else:
            self.preview.set_selection_visible(False)
            self.preview.set_crop_mode(True)
        self.summary.setText(f"Selected frame ready for crop: {image_path.name}")
        self.refresh_reference_segments()

    def analyze_cropped_delay_images(self) -> None:
        if not self.apply_crop_enabled.isChecked():
            QMessageBox.information(
                self,
                "Apply crop first",
                "Select Apply saved crop before analyzing averaged selected frames.",
            )
            return
        if self.batch_crop is None:
            QMessageBox.information(self, "Batch crop required", "Draw and save a crop first.")
            return
        if not self.subtract_saved_background_enabled.isChecked():
            QMessageBox.information(
                self,
                "Background subtraction required",
                "Select Subtract saved background before analyzing averaged selected frames.",
            )
            return
        if self.reference_background is None or self._selected_signal_mask is None:
            QMessageBox.information(
                self,
                "Background profile required",
                "Upload or save a background profile before the final analysis.",
            )
            return
        image_folder = QFileDialog.getExistingDirectory(
            self, "Select the Averaged Selected Frames folder"
        )
        if not image_folder:
            return
        default_path = str(Path(image_folder) / "averaged_selected_intensity.xlsx")
        workbook_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cropped batch workbook",
            default_path,
            "Excel workbook (*.xlsx)",
        )
        if not workbook_path:
            return
        output_path = Path(workbook_path)
        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        try:
            self.cropped_image_analysis_button.setEnabled(False)
            self.summary.setText("Applying saved background and crop to averaged time points...")
            rows = analyze_delay_images_with_crop(
                Path(image_folder),
                self.batch_crop,
                background_reference=self.reference_background,
                signal_mask=self._selected_signal_mask,
            )
            export_delay_curve_analysis(rows, output_path)
            self.cropped_image_analysis_button.setEnabled(True)
            self.summary.setText(f"Batch analysis complete: {len(rows)} time delays.")
            QMessageBox.information(
                self,
                "Averaged-frame analysis complete",
                f"Analyzed {len(rows)} averaged time points and saved the workbook to:\n{output_path}",
            )
        except (OSError, ValueError, RuntimeError) as error:
            self.cropped_image_analysis_button.setEnabled(True)
            QMessageBox.critical(self, "Cropped batch analysis failed", str(error))

    def _delay_image_analysis_failed(self, message: str) -> None:
        self.summary.setText("Highest-intensity image analysis failed")
        self.selected_image_analysis_button.setEnabled(True)
        QMessageBox.critical(self, "Image analysis failed", message)

    def _delay_analysis_complete(self, workbook_path: str, delay_count: int) -> None:
        self.summary.setText(f"Batch analysis complete: {delay_count} time delays.")
        self.batch_analysis_button.setEnabled(True)
        self.selected_image_analysis_button.setEnabled(True)
        QMessageBox.information(
            self,
            "Batch analysis complete",
            f"Analyzed {delay_count} delay folders and saved the workbook to:\n{workbook_path}",
        )

    def _delay_analysis_failed(self, message: str) -> None:
        self.summary.setText("Batch analysis failed")
        self.batch_analysis_button.setEnabled(True)
        QMessageBox.critical(self, "Batch analysis failed", message)

    def _start_video_import(self, video_paths: list[Path]) -> None:
        try:
            self.import_video_button.setEnabled(False)
            self.summary.setText(f"Preparing {len(video_paths)} videos for frame extraction...")
            self._video_worker = VideoExtractionWorker(video_paths)
            self._video_worker.progress.connect(self._video_extraction_progress)
            self._video_worker.extractionComplete.connect(self._video_extraction_complete)
            self._video_worker.start()
        except (OSError, ValueError, RuntimeError) as error:
            self.import_video_button.setEnabled(True)
            QMessageBox.critical(self, "Video import failed", str(error))

    def _video_extraction_progress(self, video_name: str, index: int, total: int) -> None:
        self.summary.setText(f"Processing video {index} of {total}: {video_name}")

    def _video_extraction_complete(
        self, outcome: tuple[list[tuple[Path, int]], list[str]]
    ) -> None:
        results, failures = outcome
        self.import_video_button.setEnabled(True)
        if len(results) == 1 and not failures:
            self._load_folder(results[0][0])
        completed_text = "\n".join(
            f"{folder}: kept {count} of 100 frames" for folder, count in results
        )
        failure_text = "\n".join(failures)
        self.summary.setText(
            f"Video extraction complete: {len(results)} succeeded, {len(failures)} failed."
        )
        title = "Video extraction complete" if not failures else "Video extraction finished with errors"
        message = completed_text or "No videos were processed successfully."
        if failure_text:
            message += f"\n\nFailed videos:\n{failure_text}"
        QMessageBox.information(self, title, message)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if any(
            url.isLocalFile()
            and (
                Path(url.toLocalFile()).is_dir()
                or Path(url.toLocalFile()).suffix.lower() in VIDEO_SUFFIXES
                or Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES
            )
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        folders = [path for path in paths if path.is_dir()]
        videos = [
            path for path in paths
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        videos.extend(
            video_path
            for folder in folders
            for video_path in sorted(folder.rglob("*"))
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_SUFFIXES
        )
        videos = list(dict.fromkeys(videos))
        if videos:
            self._start_video_import(videos)
        elif folders:
            self._load_folder(folders[0])
        elif any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in paths):
            image_path = next(
                path for path in paths
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            self.load_reference_image(image_path)
        else:
            event.ignore()
            return
        event.acceptProposedAction()

    def _load_folder(self, folder: Path) -> None:
        try:
            self.experiment = load_experiment(folder)
            self._crop_reference_image_loaded = False
            self._selected_signal_mask = None
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

    def average_selected_frame_folders(self) -> None:
        selected_root = QFileDialog.getExistingDirectory(
            self, "Select the folder containing selected time-point folders"
        )
        if not selected_root:
            return
        try:
            averaged_folder = average_selected_frame_folders(Path(selected_root))
            self._averaged_selected_folder = averaged_folder
            self.summary.setText(f"Averaged selected frames saved to: {averaged_folder}")
            QMessageBox.information(
                self,
                "Selected frames averaged",
                f"Created one averaged image per time point in:\n{averaged_folder}",
            )
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.critical(self, "Could not average selected frames", str(error))

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
        if self._crop_reference_image_loaded:
            if self._selected_signal_mask is None:
                QMessageBox.information(
                    self,
                    "Signal segment required",
                    "Enable bright-segment detection and click the signal segment before saving.",
                )
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save background profile",
                "background_reference.npz",
                "Background reference (*.npz)",
            )
            if not path:
                return
            profile_path = Path(path)
            if profile_path.suffix.lower() != ".npz":
                profile_path = profile_path.with_suffix(".npz")
            try:
                reference_background = estimate_reference_background(
                    self.preview._source_image, self._selected_signal_mask
                )
                save_reference_profile(
                    self._selected_signal_mask, reference_background, profile_path
                )
                self.reference_background = reference_background
                self.reference_signal_mask = self._selected_signal_mask.copy()
                self.subtract_saved_background_enabled.setEnabled(True)
                self.apply_crop_enabled.setEnabled(self.batch_crop is not None)
                self.summary.setText(f"Saved background profile: {profile_path.name}")
                QMessageBox.information(
                    self,
                    "Background profile saved",
                    f"Saved the selected signal and fitted background to:\n{profile_path}",
                )
            except (OSError, ValueError, np.linalg.LinAlgError) as error:
                QMessageBox.warning(self, "Cannot save profile", str(error))
            return
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
            if self._crop_reference_image_loaded:
                if signal_mask.shape != self.preview._source_image.shape:
                    raise ValueError("Profile dimensions do not match the reference photo")
            elif self.experiment is not None:
                image_shape = next(iter(self.experiment.combined.values())).shape
                if signal_mask.shape != image_shape:
                    raise ValueError("Profile dimensions do not match the currently loaded images")
            self.reference_signal_mask = signal_mask.copy()
            self.reference_background = background_reference
            self._selected_signal_mask = signal_mask.copy()
            self.region_mask = None
            self.selected_segment = 0
            self.isolate_segment_enabled.setChecked(False)
            self.scan_segments_enabled.setChecked(False)
            self.preview.set_reference_masks(None)
            self.use_reference_profile.setEnabled(True)
            self.use_reference_profile.setChecked(True)
            self.subtract_saved_background_enabled.setEnabled(True)
            self.preview.set_reference_masks(signal_mask)
            self.refresh_preview()
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Cannot load profile", str(error))

    def apply_crop_selection(self, x0: int, y0: int, x1: int, y1: int) -> None:
        origin_x, origin_y = self._crop_view_origin
        x0 += origin_x
        x1 += origin_x
        y0 += origin_y
        y1 += origin_y
        if x1 <= x0:
            x0, x1 = x1, x0
        if y1 <= y0:
            y0, y1 = y1, y0
        self.batch_crop = Crop(x0, y0, x1, y1)
        for spinbox, value in ((self.x0, x0), (self.y0, y0), (self.x1, x1), (self.y1, y1)):
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)
        self.crop_selection_complete = True
        self.zero_outside_box.setEnabled(True)
        self.crop_enabled.setChecked(False)
        self.preview.set_crop_mode(False)
        self.preview.set_selection(x0, y0, x1, y1)
        self.preview.set_selection_visible(self.show_crop_overlay.isChecked())
        self._crop_view_origin = (0, 0) if self._crop_reference_image_loaded else (x0, y0)
        self._clear_segment_scan_cache()
        self.refresh_preview()

    def delete_outside_crop(self) -> None:
        if self.batch_crop is None:
            QMessageBox.information(
                self,
                "No crop selected",
                "Draw or save a crop before deleting pixels outside it.",
            )
            return
        image = self.preview._source_image.copy()
        crop = self.batch_crop
        image[:crop.y0, :] = 0.0
        image[crop.y1:, :] = 0.0
        image[:, :crop.x0] = 0.0
        image[:, crop.x1:] = 0.0
        self.preview._source_image = image
        self.preview._render_image()
        self.preview.update()
        self.summary.setText(f"Pixels outside the saved crop were cleared at {crop.x0}, {crop.y0} to {crop.x1}, {crop.y1}.")

    def save_cropped_reference_image(self) -> None:
        if self.preview._source_image.size == 0:
            QMessageBox.information(
                self,
                "No image loaded",
                "Load a reference photo before saving the cropped version.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cropped reference image",
            "cropped_reference.png",
            "Image files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in IMAGE_SUFFIXES:
            output_path = output_path.with_suffix(".png")
        image = self.preview._source_image.astype(np.float64, copy=True)
        if self.uploaded_background_image is not None:
            if self.uploaded_background_image.shape != image.shape:
                QMessageBox.warning(
                    self,
                    "Background dimensions do not match",
                    "Upload a background image with the same dimensions as the reference photo.",
                )
                return
            image = np.maximum(image - self.uploaded_background_image, 0.0)
        if self.show_crop_overlay.isChecked():
            if self.batch_crop is None:
                QMessageBox.information(
                    self,
                    "No saved crop",
                    "Draw a crop and check 'Use saved crop for all following photos' before saving.",
                )
                return
            crop = self.batch_crop
            try:
                crop.validate(image.shape[1], image.shape[0])
            except ValueError as error:
                QMessageBox.warning(self, "Crop does not fit photo", str(error))
                return
            outside_crop = np.ones(image.shape, dtype=bool)
            outside_crop[crop.y0:crop.y1, crop.x0:crop.x1] = False
            image[outside_crop] = 0.0
        image = np.clip(image, 0.0, 255.0)
        Image.fromarray(image.astype(np.uint8)).save(output_path)
        self.summary.setText(f"Saved cropped reference image: {output_path.name}")
        QMessageBox.information(
            self,
            "Cropped photo saved",
            f"Saved the current cropped reference image to:\n{output_path}",
        )

    def analyze_saved_cropped_photo_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select the folder of saved cropped photos")
        if not folder:
            return
        default_path = str(Path(folder).parent / "cropped_photo_analysis.xlsx")
        workbook_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cropped photo folder workbook",
            default_path,
            "Excel workbook (*.xlsx)",
        )
        if not workbook_path:
            return
        output_path = Path(workbook_path)
        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        try:
            crop = self.batch_crop
            candidate_shape = None
            if self.preview._source_image.size:
                candidate_shape = self.preview._source_image.shape
            if crop is not None and candidate_shape is not None:
                if crop.x1 > candidate_shape[1] or crop.y1 > candidate_shape[0]:
                    crop = None
            rows = analyze_delay_images_with_crop(
                Path(folder),
                crop=crop,
                background_reference=self.reference_background,
                signal_mask=self._selected_signal_mask,
            ) if (crop is not None or self.reference_background is not None or self._selected_signal_mask is not None) else analyze_delay_images(Path(folder))
            export_delay_curve_analysis(rows, output_path)
            self.summary.setText(f"Saved cropped photo workbook: {output_path.name}")
            QMessageBox.information(
                self,
                "Cropped photo analysis complete",
                f"Analyzed {len(rows)} cropped photos and saved the workbook to:\n{output_path}",
            )
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.critical(self, "Cropped photo analysis failed", str(error))

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
        if self._crop_reference_image_loaded:
            self.refresh_reference_segments()
            return
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
        if self._crop_reference_image_loaded:
            if self._segment_scan_labels is None:
                return
            label = int(self._segment_scan_labels[y, x])
            if not label:
                return
            self._selected_signal_mask = self._segment_scan_labels == label
            self.selected_segment = label
            self.preview.set_segment_labels(self._segment_scan_labels, label)
            self.summary.setText("Signal segment selected. Save or upload a background profile.")
            return
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

    def refresh_reference_segments(self) -> None:
        if not self._crop_reference_image_loaded:
            return
        image = self.preview._source_image
        if self.scan_segments_enabled.isChecked():
            self._segment_scan_labels = connected_component_labels(
                image, float(image.mean()) + self.segment_offset.value()
            )
            self.preview.set_segment_labels(self._segment_scan_labels, self.selected_segment)
            self.summary.setText("Click the bright segment to retain during intensity analysis.")
        else:
            self._segment_scan_labels = None
            self.selected_segment = 0
            self._selected_signal_mask = None
            self.preview.set_segment_labels(None)

    def refresh_preview(self) -> None:
        if self._crop_reference_image_loaded:
            return
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
        if self._reference_photo_original is not None:
            self.preview.set_image(self._reference_photo_original, display_max=255.0)
        self.experiment = None
        self.results = []
        self.reference_background = None
        self.reference_signal_mask = None
        self._selected_signal_mask = None
        self._crop_reference_image_loaded = False
        self._crop_reference_path = None
        self._reference_photo_original = None
        self.batch_crop = None
        self.uploaded_background_image = None
        self.uploaded_background_path = None
        self._crop_view_origin = (0, 0)
        self.crop_selection_complete = False
        self._preview_stage = None
        self.region_mask = None
        self.selected_segment = 0
        self._clear_segment_scan_cache()
        self.stage_selector.blockSignals(True)
        self.stage_selector.clear()
        self.stage_selector.blockSignals(False)

        self.crop_enabled.setChecked(False)
        self.show_crop_overlay.setChecked(False)
        self.zero_outside_box.setChecked(False)
        self.zero_outside_box.setEnabled(False)
        self.background_enabled.setChecked(True)
        self.connected_region_enabled.setChecked(False)
        self.scan_segments_enabled.setChecked(False)
        self.segment_offset.setValue(0)
        self.isolate_segment_enabled.setChecked(False)
        self.use_reference_profile.blockSignals(True)
        self.use_reference_profile.setChecked(False)
        self.use_reference_profile.setEnabled(False)
        self.use_reference_profile.blockSignals(False)
        self.apply_crop_enabled.setChecked(False)
        self.apply_crop_enabled.setEnabled(False)
        self.subtract_saved_background_enabled.setChecked(False)
        self.subtract_saved_background_enabled.setEnabled(False)
        self.mode.setCurrentIndex(0)
        self.threshold.setValue(0)
        self.background.setValue(0)
        self.contrast.setValue(1.0)
        self.combine_enabled.setChecked(True)
        self.x0.setValue(0)
        self.y0.setValue(0)
        self.x1.setValue(self.x1.maximum())
        self.y1.setValue(self.y1.maximum())

        self.preview.set_selection_visible(False)
        self.preview.set_crop_mode(False)
        self.preview.set_reference_masks(None)
        self.preview.set_segment_labels(None)
        self.preview.set_view_full()
        self.summary.setText("No experiment loaded")
        self._update_background_controls()
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
            QMessageBox.information(
                self,
                "Export complete",
                "Processed images, brightness_summary.csv, and mean_intensity_average.csv were exported.",
            )


def run() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
