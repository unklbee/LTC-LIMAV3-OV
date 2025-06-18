from PySide6.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QPushButton,
    QFileDialog, QInputDialog
)
from PySide6.QtCore import Signal


class SourceModelPanel(QWidget):
    source_changed = Signal(object)  # emits 0, filepath, or rtsp URL
    model_changed  = Signal(str)
    device_changed = Signal(str)

    def __init__(self, model_names, device_keys, parent=None):
        super().__init__(parent)
        self.model_names = model_names
        self.device_keys = device_keys
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QFormLayout(self)
        # Source selector
        self.source_cb = QComboBox()
        self.source_cb.addItems(["Webcam", "File", "RTSP"])
        self.source_cb.setToolTip("Select video source type")

        # Button to set source (no shortcut)
        self.open_btn = QPushButton("Set Source")
        self.open_btn.setToolTip("Set the selected video source")

        layout.addRow("Source:", self.source_cb)
        layout.addRow("", self.open_btn)

        # Model selector
        self.model_cb = QComboBox()
        self.model_cb.addItems(self.model_names)
        self.model_cb.setToolTip("Select ONNX model for detection")
        layout.addRow("Model:", self.model_cb)

        # Device selector
        self.device_cb = QComboBox()
        self.device_cb.addItems(self.device_keys)
        self.device_cb.setToolTip("Select execution device (CPU or GPU)")
        layout.addRow("Device:", self.device_cb)

    def _connect_signals(self):
        self.open_btn.clicked.connect(self._on_open)
        self.model_cb.currentTextChanged.connect(self.model_changed)
        self.device_cb.currentTextChanged.connect(self.device_changed)

    def _on_open(self):
        kind = self.source_cb.currentText()
        if kind == "Webcam":
            self.source_changed.emit(0)
        elif kind == "File":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video/Image", "",
                "Video (*.mp4 *.avi);;Image (*.jpg *.png)"
            )
            if path:
                self.source_changed.emit(path)
        else:  # RTSP
            url, ok = QInputDialog.getText(
                self, "RTSP URL", "Enter RTSP URL:"
            )
            if ok and url.strip():
                self.source_changed.emit(url.strip())
