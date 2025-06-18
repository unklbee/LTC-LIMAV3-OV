import json
from typing import Any, Dict, List, Tuple
from pathlib import Path


def pts_to_serializable(pts: List[Any]) -> List[Tuple[float, float]]:
    """
    Convert list of QPointF or raw (x, y) tuples/lists to a serializable
    list of (x, y) tuples.
    """
    serial: List[Tuple[float, float]] = []
    for p in pts:
        try:
            # Qt QPoint or QPointF
            x, y = float(p.x()), float(p.y())
        except Exception:
            # raw tuple or list
            x, y = float(p[0]), float(p[1])
        serial.append((x, y))
    return serial


def read_preset_file(path: str) -> Dict[str, Any]:
    """
    Read and validate a preset JSON file.
    """
    data = json.loads(Path(path).read_text())
    required = {"source_kind", "source_value", "model", "device",
                "confidence_threshold", "roi", "line"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Preset missing keys: {missing}")
    return data


def write_preset_file(path: str, preset: Dict[str, Any]) -> None:
    """
    Write preset dictionary to JSON file.
    """
    with open(path, "w") as f:
        json.dump(preset, f, indent=2)
