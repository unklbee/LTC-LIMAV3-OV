# src/object_detector/inference.py
import cv2
import openvino as ov
import numpy as np

from .preprocessing import letterbox
from .config import Config

class Detector:
    def __init__(self,
                 weights: str,
                 device: str = "AUTO",
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45):
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres

        # Initialize OpenVINO Core
        self.core = ov.Core()

        # Load model - OpenVINO expects .xml file
        if weights.endswith('.onnx'):
            # Convert ONNX to OpenVINO format if needed
            model = self.core.read_model(weights)
        elif weights.endswith('.xml'):
            model = self.core.read_model(weights)
        else:
            raise ValueError(f"Unsupported model format: {weights}")

        # Compile model for specified device
        self.compiled_model = self.core.compile_model(model, device)

        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Create inference request
        self.infer_request = self.compiled_model.create_infer_request()

    def predict(self, im: np.ndarray) -> np.ndarray:
        # Run inference
        self.infer_request.infer({self.input_layer: im})

        # Get output
        raw = self.infer_request.get_output_tensor(0).data
        preds = raw.reshape(-1, 7)

        # Filter by confidence
        mask = preds[:, 6] > self.conf_thres
        preds = preds[mask]
        if preds.shape[0] == 0:
            return np.zeros((0,7), dtype=preds.dtype)

        # NMS
        boxes = preds[:, :4].copy()
        xywh = boxes
        xywh[:, 2:] -= xywh[:, :2]
        indices = cv2.dnn.NMSBoxes(
            bboxes=xywh.tolist(),
            scores=preds[:, 6].tolist(),
            score_threshold=self.conf_thres,
            nms_threshold=self.iou_thres
        )
        if len(indices) == 0:
            return np.zeros((0,7), dtype=preds.dtype)

        inds = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]
        return preds[inds]

# -------------------------------------------------
# After class Detector is complete:
# -------------------------------------------------

def detect(detector: Detector, img: np.ndarray):
    """
    Preprocess image and run prediction.
    """
    tensor, ratio, dwdh = letterbox(
        img,
        new_shape=Config.INPUT_SIZE,
        color=Config.PAD_COLOR,
        auto=False,
        scaleup=True,
        stride=Config.STRIDE,
    )
    preds = detector.predict(tensor)
    return preds, ratio, dwdh