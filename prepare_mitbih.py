import os
import numpy as np
import wfdb
from collections import Counter
from scipy.signal import butter, sosfiltfilt

# Configuration / constants

data_dir = 'data/mitbih'

# Specific records to use (subset of MIT-BIH arrhythmia DB)

records = ['100', '102', '103', '105', '108', '109', '112', '113']

# Patient splits
train_recs = ['100', '102', '103', '108', '113', '112']
val_recs   = ['105']
test_recs  = ['109']

# MIT-BIH Sampling rate (Hz)
fs = 360  # samples per second

# windows length in seconds to capture QRS complex
win_sec = 2.0
# window length in samples (expected 720)
win = int(win_sec * fs)
# half window length (expected 360) to get context before and after the beat
half = win // 2

# Label mapping for Normal vs Ventricular beats
NORMAL_SYMS = {'N', 'L', 'R', 'e', 'j'}
VENT_SYMS   = {'V', 'E'}

def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=fs, order=2):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)


# Data download
os.makedirs(data_dir, exist_ok=True)
print("Downloading records (first run only)—this may re-download if files already exist.")
wfdb.dl_database('mitdb', dl_dir=data_dir, records=records)


# Window extraction
X_list, y_list, rid_list = [], [], []

# Loop through each record and build windows centered on annotated beats
for record in records:
    rec_path = os.path.join(data_dir, record)

    # Load the raw ECG waveform
    rec = wfdb.rdrecord(rec_path)
    raw_sig = rec.p_signal[:, 0].astype(np.float32)

    # Apply Bandpass Filter to clean the signal
    sig = bandpass_filter(raw_sig)

    # Load annotations for beat symbols and sample indices
    ann = wfdb.rdann(rec_path, 'atr')

    # Per-record z-score signal normalization to make signals across different patients comparable in scale.
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    sig = sig.astype(np.float32)


    # Map symbol to binary label if it belongs to our sets; else skip
    for samp, sym in zip(ann.sample, ann.symbol):
        if sym in NORMAL_SYMS:
            label = 0
        elif sym in VENT_SYMS:
            label = 1
        else:
            continue

        # Compute window bounds centered on this beat
        start, end = samp - half, samp + half

        # Skip beats where a full 2-second window would run off the edge of the signal.
        if start < 0 or end > len(sig):
            continue

        # Slice the 2-second window (shape: (win,))
        window = sig[start:end]

        # Append data, label, and the record id (for patient-wise split later)
        X_list.append(window)
        y_list.append(label)
        rid_list.append(record)

# Convert lists to structured numpy arrays
X = np.stack(X_list).astype(np.float32)
y = np.array(y_list, dtype=np.int64)
rids = np.array(rid_list)

# Quick dataset overview
print("Collected windows:", X.shape)
print(f"Positive rate (V) = mean(y): {y.mean():.4f}")
print("Per-record window counts:", Counter(rids))

# Patient-wise split
# Helper: select windows belonging to specific patients.
def in_recs(reclist):
    rec_filter = np.isin(rids, reclist)
    return X[rec_filter], y[rec_filter]


# Create the splits using subject lists defined above
X_train, y_train = in_recs(train_recs)
X_val,   y_val   = in_recs(val_recs)
X_test,  y_test  = in_recs(test_recs)

# Add a channel dimension because Conv1D needs (batch, time, channels).
X_train = X_train[..., None]
X_val   = X_val[..., None]
X_test  = X_test[..., None]


# class weights so keras scales the loss per sample for the unbalanced data:
cnt = Counter(y_train.tolist())
w0 = 1.0
w1 = cnt[0] / max(1, cnt[1]) if cnt[1] > 0 else 1.0
class_weight = {0: w0, 1: w1}

np.savez_compressed(
    'mitbih_windows.npz',
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
    X_all=X[..., None],
    y_all=y,
    rids_all=rids.astype(str),
    train_recs=np.array(train_recs),
    val_recs=np.array(val_recs),
    test_recs=np.array(test_recs),
    class_weight_0=np.array(w0, dtype=np.float32),
    class_weight_1=np.array(w1, dtype=np.float32),
    fs=np.array(fs, dtype=np.int32),
    win=np.array(win, dtype=np.int32),


)

print("Saved mitbih_windows.npz.")
