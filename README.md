# Low-Power ECG Arrhythmia Detection

This project explores real-time ECG arrhythmia detection using quantized
convolutional neural networks for low-power edge devices.

The goal is to move arrhythmia detection from offline logger review toward
on-device inference that runs closer to the patient.

## Quick Start

Install dependencies with `uv`. The dependency groups are split so Raspberry Pi
installs do not pull TensorFlow unless training dependencies are requested.

### Raspberry Pi / Edge Inference

```bash
uv sync --only-group edge
uv run --group edge scripts/evaluate.py --mode trace
```

### Training Workstation

```bash
uv sync --group train --group analysis
uv run --group train scripts/prepare_mitbih.py
uv run --group train scripts/prepare_incart.py
uv run --group train scripts/train.py
uv run --group train scripts/evaluate.py --mode experiments
```

### Analysis Notebooks

```bash
uv run --group analysis jupyter notebook
```

## Data

Raw ECG records are expected under:

```text
data/mitbih/
data/incart/
```

Generated dataset windows are ignored by Git:

```text
mitbih_windows.npz
incart_windows.npz
```

See `data/README.md` for the expected local data layout.

## Runtime Evaluation

Run the evaluation script with an explicit mode:

```bash
uv run --group edge scripts/evaluate.py --mode trace
uv run --group train scripts/evaluate.py --mode sweep
uv run --group train scripts/evaluate.py --mode experiments
uv run --group train scripts/evaluate.py --mode all
```

Modes:

- `trace`: generate representative threshold trace plots.
- `sweep`: run adaptive-threshold parameter sweeps and write CSV results.
- `experiments`: run static/adaptive evaluations across MIT-BIH and INCART.
- `all`: run every runtime workflow.

## Current Layout

```text
scripts/                     Runnable workflows
src/ecg_arrhythmia/          Reusable ECG runtime, model, and metric helpers
models/ecg_int8.tflite       Tracked deployable INT8 model
data/README.md               Local data layout notes
outputs/README.md            Generated output notes
docs/experiments/            Experiment notes
docs/analysis/               Sweep analysis artifacts and notebooks
```

## Contributor Entry Points

Start with the file that matches the kind of change you want to make:

```text
src/ecg_arrhythmia/model.py       Model architectures and TFLite export helpers
src/ecg_arrhythmia/threshold.py   Adaptive threshold algorithm
src/ecg_arrhythmia/inference.py   TFLite runtime wrapper
src/ecg_arrhythmia/metrics.py     Threshold selection and binary metrics
scripts/prepare_mitbih.py         MIT-BIH preprocessing and split metadata
scripts/prepare_incart.py         INCART preprocessing and resampling
scripts/train.py                  Main training/export workflow
scripts/evaluate.py               Runtime evaluation, sweeps, and trace plots
docs/experiments/                 Written experiment notes
docs/analysis/                    Notebooks and curated sweep outputs
```


## Legacy Requirements

`requirements.txt` and `requirements_edge.txt` are kept temporarily during the
`uv` migration. Prefer `pyproject.toml` and `uv.lock` once the lockfile has been
generated and verified.
