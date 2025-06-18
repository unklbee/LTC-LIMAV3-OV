# Directory structure
#cd src
#pip install -e .
#python -m object_detector.gui.main_window
#pyinstaller --name lima_traffic_counting --onefile --windowed --paths src --add-data "data;data" --add-data "models;models" run.py

# vehicle_counting_yolov7/
# ├── .venv
# ├── build
# ├── data
# │   └── class.names
# ├── models
# │   ├── yolov7.onnx
# │   └── yolov7-tiny.onnx
# ├── README.md
# ├── requirements.txt
# ├── setup.py
# ├── src/
# │   └── object_detector/
# │       ├── __init__.py
# │       ├── __main__.py
# │       ├── cli.py
# │       ├── config.py
# │       ├── inference.py
# │       ├── loader.py
# │       ├── preprocessing.py
# │       ├── visualization.py
# │       └── gui/
# │           ├── __init__.py
# │           ├── line_drawer.py
# │           ├── main_window.py
# │           ├── roi_drawer.py
# │           └── video_thread.py
# │
# └── tests/
#     ├── test_loader.py
#     └── test_inference.py

# [GPU Decode Thread]
#     └─> queue_frames_gpu   (frame GPU memory)
#
# [Preproc+Infer Thread]
#     └─ pop frame_gpu
#     └─ cv2.cuda.resize & letterbox → prepped_gpu
#     └─ ort_session.run_with_iobinding(prepped_gpu) → raw_dets (GPU)
#     └─ queue_dets (tuple of raw_frame_gpu, dets, ratio, dwdh)
#
# [Postproc+Draw Thread]
#     └─ pop (raw_frame_gpu, dets, ratio, dwdh)
#     └─ GPU NMS (built‐in ONNXRuntime NMS) or CPU NMS
#     └─ cv2.cuda.boxFilter / cv2.cuda.addWeighted → overlay_gpu
#     └─ queue_render (out_frame_gpu)
#
# [UI Thread]
#     └─ pop out_frame_gpu
#     └─ one copy GPU→CPU to QImage or direct GL texture upload
#     └─ display

# Upgrade tracker ke ByteTrack atau DeepSORT.
#
# Tweak hyperparameter (max_age, min_hits, IoU, threshold per-kelas).
#
# Gunakan region band & multiple-line logic.
#
# Kalibrasi perspektif untuk akurasi spasial.


# setup.py
from setuptools import setup, find_packages

setup(
    name="object_detector",
    version="0.1.0",
    description="Lightweight object detection toolkit with CLI and GUI using OpenVINO",
    author="Hasbi",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "opencv-python>=4.11.0.86",
        "numpy>=2.2.5",
        "PySide6>=6.9.0",
        "openvino>=2024.0.0"  # Replace onnxruntime with openvino
    ],
    extras_require={
        "dev-tools": [
            "openvino-dev>=2024.0.0"  # For model conversion tools
        ],
    },
    entry_points={
        "console_scripts": [
            "odetect=object_detector.cli:main",
            "odetect-gui=object_detector.gui.main_window:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)