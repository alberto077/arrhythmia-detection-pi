# prepare_mitbih.py  (patient-wise split, heavily commented)
#
# Goal:
#   1) Download selected MIT-BIH records (signals + annotations)
#   2) Build 2-second ECG windows centered on annotated beats
#   3) Map labels to a binary task: Normal (0) vs Ventricular (1)
#   4) Normalize per record (z-score) so amplitudes are comparable
#   5) Perform a PATIENT-WISE split: train on some records, validate on one, test on another
#   6) Save arrays + useful metadata for training/evaluation later
#
# Why patient-wise?
#   Randomly splitting by windows can leak patient-specific patterns into both train and test,
#   inflating metrics. Patient-wise ensures the model is evaluated on unseen subjects.

import os
import numpy as np
import wfdb
from collections import Counter  # for quick label and record counts

# Configuration / constants

data_dir = 'data/mitbih'

# Specific records to use (subset of MIT-BIH arrhythmia DB)
# You originally used this exact set; we’ll also define which records go to train/val/test.

records = ['100', '102', '103', '105', '108', '109', '112', '113']
# A “record” is typically ~30 minutes of ECG sampled at 360 Hz, with a header + signal + annotation trio of files:
#
# 100.hea → header (metadata: sampling rate, leads, gain, start time, etc.)
#
# 100.dat → the raw ECG signal data
#
# 100.atr → the cardiologist beat annotations (symbols like N, V, etc.)

# --- Patient-wise split lists ---
# Train on 6 patients, validate on 1, test on 1 (you can reshuffle later if you like)
train_recs = ['100', '102', '103', '108', '113', '112']  # training subjects
val_recs   = ['105']                                     # validation subject
test_recs  = ['109']                                     # test subject (held out)

# Sampling rate for MIT-BIH ECG signals (Hz)
fs = 360  # samples per second

# Window config:
#   2-second windows centered on the beat annotation → 720 samples total.
#   We use "half" to take equal samples before and after the beat.
win_sec = 2.0
win = int(win_sec * fs)  # window length in samples (expected 720)
half = win // 2          # half window length (expected 360)

# Label mapping for a first-pass binary task:
#   Normal vs Ventricular beats.
#   These symbol sets follow common AAMI simplifications for MIT-BIH.
NORMAL_SYMS = {'N', 'L', 'R', 'e', 'j'}  # map to label 0
VENT_SYMS   = {'V', 'E'}                  # map to label 1

# Ensure output directory exists (where WFDB will place files)
os.makedirs(data_dir, exist_ok=True)

print("Downloading records (first run only)—this may re-download if files already exist.")
# NOTE: Per your request, we do NOT guard this download; it will run each time.
# WFDB will place: <record>.dat, <record>.hea, <record>.atr under data_dir
wfdb.dl_database('mitdb', dl_dir=data_dir, records=records)

# -----------------------------
# Window extraction
# -----------------------------
# We’ll accumulate:
#   X_list: the windows (each is shape (win,))
#   y_list: the labels (0 normal, 1 ventricular)
#   rid_list: the record id for each window (e.g., '100', ...), to enable patient-wise splitting
X_list, y_list, rid_list = [], [], []

# Loop through each record and build windows centered on annotated beats
for record in records:
    rec_path = os.path.join(data_dir, record)

    # Load the raw ECG waveform (multi-channel possible; we take the first channel)
    # rec.p_signal -> shape (num_samples, num_channels)
    rec = wfdb.rdrecord(rec_path)
    sig = rec.p_signal[:, 0].astype(np.float32)

    # Load annotations: 'atr' file provides beat symbols and sample indices
    ann = wfdb.rdann(rec_path, 'atr')

    # Per-record z-score normalization:
    #   This makes signals across different patients comparable in scale.
    #   (At runtime on Pi, we'll z-score each window with its own mean/std.)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    # Iterate all annotated beats and collect windows around beats
    for samp, sym in zip(ann.sample, ann.symbol):
        # Map symbol to binary label if it belongs to our sets; else skip
        if sym in NORMAL_SYMS:
            label = 0
        elif sym in VENT_SYMS:
            label = 1
        else:
            continue  # skip other beat types for this binary task

        # Compute window bounds centered on this beat
        start, end = samp - half, samp + half

        # Guard against windows that would run off the signal edges
        if start < 0 or end > len(sig):
            continue

        # Slice the 2-second window (shape: (win,))
        window = sig[start:end]

        # Append data, label, and the record id (for patient-wise split later)
        X_list.append(window)
        y_list.append(label)
        rid_list.append(record)

# Convert lists to structured numpy arrays
# X: (num_windows, win) all windows      y: (num_windows,) all labels    rids: (num_windows,) id to match
X = np.stack(X_list).astype(np.float32)
y = np.array(y_list, dtype=np.int64)
rids = np.array(rid_list)

# Quick dataset overview
print("Collected windows:", X.shape, " | Positive rate (V) = mean(y):", f"{y.mean():.4f}")
print("Per-record window counts:", Counter(rids))

# -----------------------------
# Patient-wise split
# -----------------------------
# Helper to select windows belonging to a list of record IDs
def in_recs(reclist):
    rec_filter = np.isin(rids, reclist)   # boolean array: True for windows whose record id is in reclist
    return X[rec_filter], y[rec_filter]   # pick only those windows (rows) for X and y


# Create the splits using subject lists defined above
X_train, y_train = in_recs(train_recs)  # windows from train subjects only
X_val,   y_val   = in_recs(val_recs)    # windows from validation subject
X_test,  y_test  = in_recs(test_recs)   # windows from held-out test subject

# Add a channel dimension so shapes become:
#(num_windows, time, channels) -> (N, 720, 1)
# This matches what 1D CNNs in Keras expect.
# Without that extra axis, Keras would see (batch, time) and complain
# because Conv1D needs (batch, time, channels).
X_train = X_train[..., None]
X_val   = X_val[..., None]
X_test  = X_test[..., None]

# -----------------------------
# Oversample ventricular beats in training set only
# -----------------------------
pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]

num_pos = len(pos_idx)
num_neg = len(neg_idx)

print(f"Before oversampling: pos={num_pos}, neg={num_neg}")

# Target ratio: 1 positive for every 5 negatives
target_pos = num_neg // 5

if num_pos > 0:
    reps = target_pos // num_pos
    remainder = target_pos % num_pos

    oversampled_pos_idx = np.concatenate([
        np.repeat(pos_idx, reps),
        np.random.choice(pos_idx, remainder, replace=True)
    ])

    X_train = np.concatenate([X_train[neg_idx], X_train[oversampled_pos_idx]])
    y_train = np.concatenate([y_train[neg_idx], y_train[oversampled_pos_idx]])

print(f"After oversampling: pos={np.sum(y_train==1)}, neg={np.sum(y_train==0)}")


print("Split shapes:")
print("  Train:", X_train.shape, y_train.shape)
print("  Val:  ", X_val.shape,   y_val.shape)
print("  Test: ", X_test.shape,  y_test.shape)

# -----------------------------
# Class weights (for imbalanced data)
# -----------------------------
# Ventricular beats are usually rarer; class_weight helps the loss function
# pay more attention to the positive class.

# Why we need class weights
#
# Imbalance problem: Suppose your train set has 95% normal, 5% ventricular.
# A dumb model predicting all 0s gets 95% accuracy but 0% recall for arrhythmias.
#
# Goal: Make errors on the positive class (ventricular) “cost more”
# so the model learns features for them instead of ignoring them.

# What class_weight does in Keras
#
# When you pass class_weight={0:w0, 1:w1} to model.fit(...), Keras scales the loss per sample:
#
# For a ventricular sample, its loss is multiplied by w1.
#
# For a normal sample, its loss is multiplied by w0 (we keep this at 1.0).


#Check how many normal vs ventricular beats we have in our sample and adjust the weights accordingly

cnt = Counter(y_train.tolist()) # Creates the mapping {0: #, 1: #}
w0 = 1.0
raw_w1 = (cnt[0] / max(1, cnt[1])) if cnt[1] > 0 else 1.0  # simple heuristic
w1 = min(raw_w1, 20.0)
class_weight = {0: w0, 1: w1}
print("Class counts (train):", cnt, " -> class_weight:", class_weight)

# -----------------------------
# Save dataset artifact + metadata
# -----------------------------
# We include:
#   - arrays: X_train/X_val/X_test and y_train/y_val/y_test
#   - fs, win: sampling metadata
#   - the actual record lists used for each split (reproducibility)
#   - class weights (so the training script can import them)
#   - symbol sets (so you remember exactly which mapping you used)

# --- Save full dataset too (so runtime can analyze specific patients later) ---
X_all = X[..., None]          # (N, 720, 1) channel dim added
y_all = y                     # (N,)
rids_all = rids.astype(str)   # (N,) record id for each window

np.savez_compressed(
    'mitbih_windows.npz',
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,

    X_all=X_all, y_all=y_all, rids_all=rids_all,

    fs=fs, win=win,
    train_recs=np.array(train_recs),
    val_recs=np.array(val_recs),
    test_recs=np.array(test_recs),
    class_weight_0=np.array(w0, dtype=np.float32),
    class_weight_1=np.array(w1, dtype=np.float32),
    NORMAL_SYMS=np.array(sorted(list(NORMAL_SYMS))),
    VENT_SYMS=np.array(sorted(list(VENT_SYMS))),

)

print("Saved mitbih_windows.npz (patient-wise splits + metadata).")
