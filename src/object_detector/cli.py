# src/object_detector/cli.py
import argparse
import random
import cv2
import numpy as np

from .config      import Config
from .loader      import load_source
from .preprocessing import letterbox
from .inference   import Detector
from .visualization import draw_detections

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--source', default='0',
        help="Webcam jika '0', atau path ke file video/image"
    )
    p.add_argument(
        '--names', default=str(Config.NAMES_PATH),
        help="File .names untuk kelas"
    )
    p.add_argument(
        '--weights', default=str(Config.WEIGHTS_PATH),
        help="Path ke model OpenVINO (.xml) atau ONNX (.onnx)"
    )
    p.add_argument(
        '--device', default='AUTO',
        choices=['CPU', 'GPU', 'AUTO', 'NPU'],
        help="OpenVINO device (CPU, GPU, AUTO, NPU)"
    )
    p.add_argument(
        '--conf-thres', type=float,
        default=Config.CONFIDENCE_THRESHOLD,
        help="Confidence threshold (0.0–1.0)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Initialize detector with OpenVINO device
    det = Detector(args.weights, args.device)
    names = open(args.names).read().splitlines()
    colors = [tuple(random.randint(0,255) for _ in range(3)) for _ in names]

    # Load source (image vs video/webcam)
    is_img, wait, frame, cap = load_source(args.source, Config.IMAGE_EXTENSIONS)

    print(f"Using OpenVINO device: {args.device}")
    print(f"Model: {args.weights}")

    while True:
        if not is_img:
            ret, frame = cap.read()
            if not ret:
                break

        # Preprocess + infer
        tensor, ratio, dwdh = letterbox(
            frame,
            new_shape=Config.INPUT_SIZE,
            color=Config.PAD_COLOR,
            auto=False,
            scaleup=True,
            stride=Config.STRIDE,
        )
        dets = det.predict(tensor)

        # Filter hasil deteksi berdasarkan threshold & kelas
        if dets.size:
            # threshold
            dets = dets[dets[:,4] >= args.conf_thres]
            # hanya kendaraan
            ids = dets[:,5].astype(int)
            dets = dets[np.isin(ids, [names.index(c) for c in Config.VEHICLE_CLASSES if c in names])]

        # Gambar kotak + label
        out = draw_detections(frame, dets, ratio, dwdh, names, colors, args.conf_thres)

        cv2.imshow('Detections (OpenVINO)', out)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()