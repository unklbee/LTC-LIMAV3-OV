from PySide6.QtWidgets import QWidget, QFormLayout, QSlider, QLabel
from PySide6.QtCore import Qt, Signal

class DetectionSettingsPanel(QWidget):
    conf_changed = Signal(int)  # emits slider value 0–100

    def __init__(self, initial=50, parent=None):
        super().__init__(parent)
        self._build_ui(initial)
        self._connect_signals()

    def _build_ui(self, initial):
        layout = QFormLayout(self)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(initial)
        self.label = QLabel(f"{initial}%")
        layout.addRow("Confidence:", self.slider)
        layout.addRow("", self.label)

    def _connect_signals(self):
        self.slider.valueChanged.connect(self._on_value)

    def _on_value(self, val):
        self.label.setText(f"{val}%")
        self.conf_changed.emit(val)
