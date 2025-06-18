# src/object_detector/config.py

from pathlib import Path
from typing import Tuple, List, Dict
from zoneinfo import ZoneInfo
import openvino as ov


class Config:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    MODEL_DIR:    Path = PROJECT_ROOT / "models"
    DATA_DIR:     Path = PROJECT_ROOT / "data"

    # Changed from .onnx to .xml for OpenVINO
    WEIGHTS_PATH: Path = MODEL_DIR / "yolov7-tiny.xml"
    NAMES_PATH:   Path = DATA_DIR / "class.names"

    # tempat file SQLite untuk menyimpan counts
    DB_PATH:      Path = DATA_DIR / "counts.db"
    # interval simpan (detik)
    COUNT_SAVE_INTERVAL_SEC: int = 1 * 60

    # URL endpoint untuk push via API (kosong = tidak push)
    API_URL: str = ""

    # host identifier default
    HOST_ID: str = "default-host"
    TIMEZONE: ZoneInfo = ZoneInfo("Asia/Jakarta")

    # OpenVINO devices - replaced ONNX providers
    OPENVINO_DEVICES: List[str] = ["GPU", "CPU", "AUTO"]

    # OpenVINO device mapping for GUI
    DEVICE_MAP: Dict[str, str] = {
        "Intel GPU": "GPU",
        "CPU": "CPU",
        "Auto": "AUTO",
        "Intel NPU": "NPU"  # For newer Intel processors with NPU
    }

    VEHICLE_CLASSES: List[str] = [
        "bicycle", "car", "truck", "motorbike", "bus"
    ]

    # preprocessing
    INPUT_SIZE: Tuple[int, int]      = (640, 640)
    PAD_COLOR: Tuple[int, int, int]  = (114, 114, 114)
    STRIDE:     int                  = 32

    # inference
    CONFIDENCE_THRESHOLD: float      = 0.25
    DEFAULT_FPS                      = 30
    IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}

    @classmethod
    def ensure_dirs(cls) -> None:
        """
        Panggil di startup untuk memastikan semua folder output/data ada.
        """
        # direktori yang wajib ada
        for d in (cls.MODEL_DIR, cls.DATA_DIR, cls.DB_PATH.parent):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_available_devices(cls) -> Dict[str, str]:
        """
        Get available OpenVINO devices on the current system.
        """
        try:
            core = ov.Core()
            available_devices = core.available_devices

            device_mapping = {}
            for device in available_devices:
                if "GPU" in device:
                    device_mapping["Intel GPU"] = "GPU"
                elif "CPU" in device:
                    device_mapping["CPU"] = "CPU"
                elif "NPU" in device:
                    device_mapping["Intel NPU"] = "NPU"

            # Always add AUTO as fallback
            device_mapping["Auto"] = "AUTO"

            return device_mapping
        except Exception:
            # Fallback if OpenVINO is not properly installed
            return {"CPU": "CPU", "Auto": "AUTO"}