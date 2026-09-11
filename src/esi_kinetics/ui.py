from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QDoubleSpinBox, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from .io import export_results, load_experiment
from .models import Crop, PipelineSettings
from .processing import process_image


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ESI Kinetics Analysis")
        self.resize(1100, 700)
        self.experiment = None
        self.results = []
        self.stage_selector = QComboBox()
        self.stage_selector.currentIndexChanged.connect(self.refresh_preview)
        self.preview = QLabel("Import an image folder to begin")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(500, 400)
        self.summary = QLabel("No experiment loaded")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1_000_000_000)
        self.threshold.setValue(0)
        self.background = QDoubleSpinBox()
        self.background.setRange(0, 1_000_000_000)
        self.x0 = QSpinBox()
        self.y0 = QSpinBox()
        self.x1 = QSpinBox()
        self.y1 = QSpinBox()
        self.crop_enabled = QCheckBox("Apply crop")
        self.background_enabled = QCheckBox("Remove background")
        self.background_enabled.setChecked(True)
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
        controls = QFormLayout()
        controls.addRow(import_button)
        controls.addRow("Stage", self.stage_selector)
        controls.addRow(self.combine_enabled)
        controls.addRow(self.crop_enabled)
        controls.addRow("Crop x0 / x1", self._pair(self.x0, self.x1))
        controls.addRow("Crop y0 / y1", self._pair(self.y0, self.y1))
        controls.addRow(self.background_enabled)
        controls.addRow("Background mode", self.mode)
        controls.addRow("Threshold", self.threshold)
        controls.addRow("Background value", self.background)
        controls.addRow(reset_button)
        controls.addRow(export_button)
        for widget in (self.crop_enabled, self.background_enabled):
            widget.stateChanged.connect(self.refresh_preview)
        self.mode.currentIndexChanged.connect(self.refresh_preview)
        for widget in (self.threshold, self.background, self.x0, self.y0, self.x1, self.y1):
            widget.valueChanged.connect(self.refresh_preview)
        self.summary.setWordWrap(True)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(controls)
        left_layout.addWidget(self.summary)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    @staticmethod
    def _pair(first: QSpinBox, second: QSpinBox) -> QWidget:
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
        try:
            self.experiment = load_experiment(Path(folder))
            self.stage_selector.clear()
            self.stage_selector.addItems([stage.name for stage in self.experiment.stages])
            height, width = next(iter(self.experiment.combined.values())).shape
            for spinbox, maximum, value in ((self.x0, width, 0), (self.y0, height, 0),
                                            (self.x1, width, width), (self.y1, height, height)):
                spinbox.setRange(0, maximum)
                spinbox.setValue(value)
            self.refresh_preview()
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Import failed", str(error))

    def settings(self) -> PipelineSettings:
        mode = "threshold" if self.mode.currentIndex() == 0 else "subtract"
        return PipelineSettings(
            crop=Crop(self.x0.value(), self.y0.value(), self.x1.value(), self.y1.value()),
            background_mode=mode,
            threshold=self.threshold.value(),
            background_value=self.background.value(),
            enable_combine=self.combine_enabled.isChecked(),
            enable_crop=self.crop_enabled.isChecked(),
            enable_background=self.background_enabled.isChecked(),
        )

    def refresh_preview(self) -> None:
        if self.experiment is None or self.stage_selector.currentIndex() < 0:
            return
        stage = self.stage_selector.currentText()
        result = process_image(stage, self.experiment.combined[stage], self.settings())
        self.results = [result]
        self.summary.setText(
            f"{stage}: {result.brightness:,.3f} brightness; "
            f"{result.active_pixels:,} active pixels; max {result.maximum:,.3f}"
        )
        image = result.image
        if image.size:
            scaled = np.clip(image / max(float(image.max()), 1.0) * 255, 0, 255).astype(np.uint8)
            qimage = QImage(scaled.data, scaled.shape[1], scaled.shape[0], scaled.strides[0], QImage.Format.Format_Grayscale8)
            self.preview.setPixmap(QPixmap.fromImage(qimage.copy()).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))

    def reset(self) -> None:
        self.crop_enabled.setChecked(False)
        self.background_enabled.setChecked(True)
        self.mode.setCurrentIndex(0)
        self.threshold.setValue(0)
        self.background.setValue(0)
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
