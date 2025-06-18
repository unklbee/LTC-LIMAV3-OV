# src/object_detector/gui/widgets/status_bar.py

from PySide6.QtWidgets import QStatusBar, QLabel, QMenu
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

class StatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.fps_text_label   = QLabel("FPS:")
        self.fps_value_label  = QLabel("0.0")
        self.count_text_label = QLabel("Count:")
        self.count_value_label= QLabel("—")

        # warna teks tetap hitam
        style_black = "color: black;"
        for lbl in (self.fps_text_label, self.count_text_label, self.count_value_label):
            lbl.setStyleSheet(style_black)

        self.addPermanentWidget(self.fps_text_label)
        self.addPermanentWidget(self.fps_value_label)
        self.addPermanentWidget(self.count_text_label)
        self.addPermanentWidget(self.count_value_label)

        # context-menu untuk reset counting
        self.count_value_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.count_value_label.customContextMenuRequested.connect(self._show_context_menu)

    def update(self, fps: float, count_text: str):
        # update FPS
        self.fps_value_label.setText(f"{fps:.1f}")
        if fps < 10:
            self.fps_value_label.setStyleSheet("color: red; font-weight: bold;")
        elif fps < 20:
            self.fps_value_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.fps_value_label.setStyleSheet("color: green; font-weight: bold;")

        # update count
        self.count_value_label.setText(count_text)

    def _show_context_menu(self, pos):
        menu = QMenu()
        reset_action = QAction("Reset Count", self)
        reset_action.triggered.connect(self._reset_count)
        menu.addAction(reset_action)
        menu.exec(self.count_value_label.mapToGlobal(pos))

    def _reset_count(self):
        # 1) reset counter di back-end
        main_win = self.parent()
        if hasattr(main_win, "async_vid") and main_win.async_vid:
            main_win.async_vid.reset_counts()

        # 2) reset UI label
        self.count_value_label.setText("—")

    def setState(self, state_name: str):
        # contoh: pakai tooltip untuk debug
        self.setToolTip(f"State: {state_name}")
