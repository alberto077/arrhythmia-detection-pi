import os
import numpy as np
import wfdb
from collections import Counter
from scipy.signal import butter, sosfiltfilt


data_dir = 'data/mitbih'

records = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230, 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


train_recs = [r for r in DS1 if r >= 200]
val_recs   = [r for r in DS1 if r < 200]
test_recs  = DS2

# MIT-BIH Sampling rate (Hz)
fs = 360  
win_sec = 2.0
win = int(win_sec * fs)
half = win // 2

NORMAL = {'N','L', 'R', 'e', 'j'}
SVEB = {'A', 'a', 'J', 'S'}
VEB = {'V', 'E'}


def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=fs, order=2):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

data_empty = False
if data_empty:
    os.makedirs(data_dir, exist_ok=True)
    wfdb.dl_database('mitdb', dl_dir=data_dir, records=[str(record) for record in records])


X_list, y_list, rid_list = [], [], []

for record in records:
    rec_path = os.path.join(data_dir, str(record))

    rec = wfdb.rdrecord(rec_path)
    raw_sig = rec.p_signal[:, 0].astype(np.float32)
    sig = bandpass_filter(raw_sig)

    ann = wfdb.rdann(rec_path, 'atr')

    # Z-score
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    sig = sig.astype(np.float32)


    for samp, sym in zip(ann.sample, ann.symbol):
        if sym in NORMAL:
            label = 0
        elif sym in SVEB:
            label = 1
        elif sym in VEB:
            label = 1
        else:
            continue

        start, end = samp - half, samp + half

        if start < 0 or end > len(sig):
            continue

        window = sig[start:end]

        X_list.append(window)
        y_list.append(label)
        rid_list.append(record)

X = np.stack(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int64)
rids = np.array(rid_list)


train_mask = np.isin(rids, train_recs)
val_mask = np.isin(rids, val_recs)
test_mask = np.isin(rids, test_recs)

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask], y[val_mask]
X_test,  y_test  = X[test_mask], y[test_mask]

X_train = X_train[..., None]
X_val   = X_val[..., None]
X_test  = X_test[..., None]


cnt = Counter(y_train.tolist())
w0 = len(y_train) / (len(cnt) * max(1, cnt[0]))
w1 = len(y_train) / (len(cnt) * max(1, cnt[1]))
class_weight = {0: w0, 1: w1}

np.savez_compressed(
    'mitbih_windows.npz',
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    rids_train=rids[train_mask],
    rids_eval=rids[val_mask],
    rids_test=rids[test_mask],
    w0 =np.float32(w0),
    w1 =np.float32(w1),
)

print("Saved mitbih_windows.npz.")
