# Agent Guide: ESI Kinetics Analysis

## Purpose

This is a local Qt desktop application for analyzing experiment photographs. Users import a folder of images, group them into stages, sum all frames in each stage, select a relevant region of interest, remove background light, calculate brightness, and export images and data.

## Current structure

- `src/esi_kinetics/models.py`: serializable domain objects and pipeline settings.
- `src/esi_kinetics/processing.py`: pure NumPy processing functions. Keep this module free of Qt and filesystem concerns.
- `src/esi_kinetics/io.py`: folder loading, stage grouping, and CSV/image export helpers.
- `src/esi_kinetics/ui.py`: PySide6 workflow shell and user controls.
- `main.py`: application entry point.
- `tests/`: headless tests for scientific behavior.

## Scientific invariants

1. Input frames in one stage must have matching height and width.
2. Frame accumulation uses `float64`; never sum directly into an integer dtype because 100+ frames can overflow.
3. A processed image is derived from the immutable combined image and current settings. Do not destructively overwrite source or combined data.
4. Cropping uses half-open NumPy coordinates: `y0:y1, x0:x1`.
5. Threshold mode retains pixels strictly greater than the threshold.
6. Background subtraction clips negative values to zero before brightness calculation.
7. Brightness is the sum of processed pixel values in raw detector-count units unless a future calibration factor is explicitly applied.
8. Stage order is deterministic and must be preserved in exports.

## Pipeline

`source frames -> float64 sum -> ROI crop -> background operation -> brightness metrics`

Every step must be independently bypassable in the UI. The UI should reconstruct the result from the last valid immutable state rather than applying repeated operations to an already processed array.

## Extension guidance

- Add new scientific behavior as a pure function with tests before connecting it to Qt.
- Keep file-format parsing behind `io.py`; future HDF5 or multi-page TIFF support should not leak into UI code.
- Store user choices in `PipelineSettings` and include them in project/session exports.
- Preserve original stage membership and source paths in any saved project.
- Avoid changing the meaning of `brightness` without a versioned data-model change.

## Validation commands

```powershell
python -m pytest -q
python main.py
```

Use synthetic fixtures for automated tests. A real experiment image can be used for manual smoke testing, but must not be required for the test suite.
