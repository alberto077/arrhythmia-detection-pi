import os
import numpy as np
import wfdb
from collections import Counter
from scipy.signal import butter, sosfiltfilt, resample_poly


data_dir = 'data/incart'

records = [f'I{i:02d}' for i in range(1, 76)]

fs_target = 360
fs_source = 257

win_sec = 2.0
win = int(win_sec * fs_target) 
half = win // 2                   

NORMAL = {'N','L', 'R', 'e', 'j'}
SVEB = {'A', 'a', 'J', 'S'}
VEB = {'V', 'E'}


def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=fs_target, order=2):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

X_list, y_list, rid_list = [], [], []

for record in records:
    rec_path = os.path.join(data_dir, record)

    rec = wfdb.rdrecord(rec_path)
    mlii_idx = rec.sig_name.index('MLII') if 'MLII' in rec.sig_name else 0
    raw_sig = rec.p_signal[:, mlii_idx].astype(np.float32)

    sig_resampled = resample_poly(raw_sig, up=fs_target, down=fs_source).astype(np.float32)

    sig = bandpass_filter(sig_resampled, fs=fs_target)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    sig = sig.astype(np.float32)

    ann = wfdb.rdann(rec_path, 'atr')
    resampled_samples = np.round(ann.sample * (fs_target / fs_source)).astype(int)

    n_before = len(X_list)

    for samp, sym in zip(resampled_samples, ann.symbol):
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
        
        X_list.append(sig[start:end])
        y_list.append(label)
        rid_list.append(record)


X = np.stack(X_list).astype(np.float32)
y = np.array(y_list, dtype=np.int64)
rids = np.array(rid_list, dtype=str)


np.savez_compressed(
    'incart_windows.npz',
    X_all=X[..., None],
    y_all=y,
    rids_all=rids.astype(str),
)

print(f"\nSaved incart_windows.npz")
