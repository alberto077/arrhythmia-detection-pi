import numpy as np
from collections import deque


class DynamicThreshold:
    def __init__(
            self,
            global_threshold: float,
            buffer_size: int = 150,
            margin: float = 0.15,
            floor: float = 0.40,
            ceil: float = 0.95,
            warmup_length: int = 50,
            update_freq: int = 10,
            admit_ceil: float = 0.85,
    ):
        self.global_threshold = global_threshold
        self.buffer_size = buffer_size
        self.margin = margin
        self.floor = floor
        self.ceil = ceil
        self.warmup_length = warmup_length
        self.update_freq = update_freq
        self.admit_ceil = admit_ceil

        # Mutable state — all reset via .reset()
        self.buffer = deque(maxlen=buffer_size)
        self.window_count = 0
        self.threshold = global_threshold
        self.is_warmup = True
        self.last_threshold_update = 0
        self.alert_count = 0

    def update(self, prob: float) -> dict:
        """Feed one probability and get back the current decision."""
        self.window_count += 1

        is_alert = prob >= self.threshold
        if is_alert:
            self.alert_count += 1

        if prob < self.admit_ceil:
            self.buffer.append(prob)

        threshold_updated = False
        if (self.window_count % self.update_freq == 0
                and not self.is_warmup
                and len(self.buffer) >= 30):
            self._recompute_threshold()
            threshold_updated = True

        if (self.is_warmup
                and len(self.buffer) >= 30
                and self.window_count >= self.warmup_length):
            self._exit_warmup()

        return {
            'is_alert': is_alert,
            'threshold_updated': threshold_updated,
            'current_threshold': self.threshold,
            'buffer_size': len(self.buffer),
        }

    def get_threshold(self) -> float:
        return self.threshold

    def get_diagnostics(self) -> dict:
        buf = np.array(self.buffer) if self.buffer else np.array([0.0])
        p95 = float(np.percentile(buf, 95)) if len(buf) > 0 else 0.0

        return {
            'window_count': self.window_count,
            'is_warmup': self.is_warmup,
            'current_threshold': self.threshold,
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer_size,
            'buffer_p95': p95,
            'alert_count': self.alert_count,
        }

    def reset(self):
        self.buffer.clear()
        self.window_count = 0
        self.threshold = self.global_threshold
        self.is_warmup = True
        self.last_threshold_update = 0
        self.alert_count = 0

    def _recompute_threshold(self):
        if len(self.buffer) < 10:
            return

        buf = np.array(self.buffer)
        p95 = np.percentile(buf, 95)

        new_threshold = p95 + self.margin
        new_threshold = float(np.clip(new_threshold, self.floor, self.ceil))

        self.threshold = new_threshold
        self.last_threshold_update = self.window_count

    def _exit_warmup(self):
        self.is_warmup = False
        self._recompute_threshold()

        buf = np.array(self.buffer)
        p95 = float(np.percentile(buf, 95))

        print(f"[DynamicThreshold] Warmup complete after {self.window_count} windows")
        print(f"  Baseline P95: {p95:.4f}")
        print(f"  Initial threshold: {self.threshold:.4f}")
        print(f"  Buffer size: {len(self.buffer)}")