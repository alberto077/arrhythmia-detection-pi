# Low-Power ECG Arrhythmia Detection

This project explores real-time ECG arrhythmia detection using quantized
convolutional neural networks for low-power edge devices.

The research goal is to show that arrhythmia detection can move from offline
logger review to on-device inference that runs closer to the patient.

## Target Device

- Raspberry Pi 5 Model B Rev

## Dependencies

- Training and analysis: `requirements.txt`
- Edge/runtime inference: `requirements_edge.txt`

## Setup

Create a local Python environment and install the training dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Raspberry Pi or edge-only inference, install the smaller runtime
dependency set instead:

```bash
pip install -r requirements_edge.txt
```

## Data

Raw ECG records are expected under:

```text
data/mitbih/
data/incart/
```

Generated window files are ignored by Git:

```text
mitbih_windows.npz
incart_windows.npz
```

## Prepare Datasets

Build the MIT-BIH training, validation, and test windows:

```bash
python3 prepare_mitbih.py
```

Build the INCART external evaluation windows:

```bash
python3 prepare_incart.py
```

## Train And Export

Train the CNN and export Float32 and INT8 TFLite models:

```bash
python3 train_and_export.py
```

Expected model outputs:

```text
models/ecg_float32.tflite
models/ecg_int8.tflite
```

Only `models/ecg_int8.tflite` is currently tracked as the deployable edge
model.

## Runtime Evaluation

Run the runtime script with an explicit mode:

```bash
python3 runtime_ecg_infer.py --mode trace
python3 runtime_ecg_infer.py --mode sweep
python3 runtime_ecg_infer.py --mode experiments
python3 runtime_ecg_infer.py --mode all
```

Modes:

- `trace`: generate representative threshold trace plots.
- `sweep`: run adaptive-threshold parameter sweeps and write CSV results.
- `experiments`: run static/adaptive evaluations across MIT-BIH and INCART.
- `all`: run every runtime workflow.

Generated plots, CSVs, NumPy files, and local model variants are ignored unless
they are intentionally promoted into the repo.

## Current Layout

```text
prepare_mitbih.py              Prepare MIT-BIH windows
prepare_incart.py              Prepare INCART windows
train_and_export.py            Train CNN and export TFLite models
runtime_ecg_infer.py           Runtime evaluation and threshold experiments
dynamic_threshold.py           Adaptive threshold logic
experiments/                   Experiment notes
CSV_analysis/                  Sweep analysis artifacts
models/ecg_int8.tflite         Tracked deployable INT8 model
```
