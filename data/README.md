# Local Data Layout

This directory is for raw ECG database files and generated local data. Raw data
and generated `.npz` files are intentionally ignored by Git.

Expected raw record folders:

```text
data/mitbih/
data/incart/
```

Dataset preparation commands:

```bash
uv run --group train scripts/prepare_mitbih.py
uv run --group train scripts/prepare_incart.py
```

Generated files:

```text
mitbih_windows.npz
incart_windows.npz
```
