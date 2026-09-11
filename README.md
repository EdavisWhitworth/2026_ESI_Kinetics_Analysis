# ESI Kinetics Analysis

A local desktop application for combining experimental photographs by stage, selecting regions of interest, removing background light, and summarizing brightness across an experiment.

## Status

Initial MVP foundation. The numerical processing core is implemented and tested; the Qt workflow currently provides import, stage combination, crop, background processing, results, and export entry points.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Run

```powershell
python main.py
```

## Test

```powershell
python -m pytest -q
```

See [Agents.MD](Agents.MD) for architecture, invariants, and guidance for future iterations.
