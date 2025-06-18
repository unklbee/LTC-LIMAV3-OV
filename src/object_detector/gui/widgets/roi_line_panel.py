from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence

class RoiLinePanel(QWidget):
    roi_requested  = Signal()
    line_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.btn_roi  = QPushButton("Define ROI")
        self.btn_roi .setShortcut(QKeySequence("Ctrl+R"))
        self.btn_line = QPushButton("Define Line")
        self.btn_line.setShortcut(QKeySequence("Ctrl+L"))
        layout.addWidget(self.btn_roi)
        layout.addWidget(self.btn_line)

    def _connect_signals(self):
        self.btn_roi.clicked.connect(self.roi_requested)
        self.btn_line.clicked.connect(self.line_requested)

    def clear(self):
        """
        Reset any internal ROI/Line state and restore button texts.
        Called when MainWindow.reset() is invoked.
        """

        # Jika nanti ada state internal for highlight / label, reset di sini.
        # Untuk sekarang, cukup pastikan tombol kembali ke teks aslinya:
        self.btn_roi.setText("Define ROI")
        self.btn_line.setText("Define Line")