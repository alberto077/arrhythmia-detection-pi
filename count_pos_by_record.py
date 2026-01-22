import numpy as np
from collections import defaultdict

d = np.load("mitbih_windows.npz", allow_pickle=True)
y = d["y_all"]
r = d["rids_all"].astype(str)

counts = defaultdict(lambda: [0, 0])  # [total, pos]
for rid, label in zip(r, y):
    counts[rid][0] += 1
    counts[rid][1] += int(label == 1)

print("Record | total_windows | ventricular_windows | pos_rate")
for rid in sorted(counts.keys()):
    total, pos = counts[rid]
    print(f"{rid:>5} | {total:>12} | {pos:>18} | {pos/total if total else 0:.4f}")
