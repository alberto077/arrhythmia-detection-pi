import numpy as np


try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError as exc:
        raise ImportError("No lightweight TFLite runtime found") from exc


class ECGDetector:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_scale, self.input_zero_point = self.input_details[0]["quantization"]
        self.output_scale, self.output_zero_point = self.output_details[0]["quantization"]

    def predict(self, window_1d):
        w = np.asarray(window_1d, dtype=np.float32)
        w = (w - w.mean()) / (w.std() + 1e-8)
        x = w[None, :, None].astype(np.float32)

        if self.input_details[0]["dtype"] == np.int8:
            x = x / self.input_scale + self.input_zero_point
            x = np.clip(np.round(x), -128, 127).astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()

        y = self.interpreter.get_tensor(self.output_details[0]["index"])
        if self.output_details[0]["dtype"] == np.int8:
            y = (y.astype(np.float32) - self.output_zero_point) * self.output_scale

        return float(y.reshape(-1)[0])

