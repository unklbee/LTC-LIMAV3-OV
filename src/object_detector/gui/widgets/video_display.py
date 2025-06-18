from pathlib import Path
from PySide6.QtWidgets import QLabel, QMessageBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter

class VideoDisplay(QLabel):
    file_dropped = Signal(object)  # emits filepath

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background:#222;color:#888;")
        self.setAcceptDrops(True)

        self._pixmap = QPixmap()
        self.setText(
            "No Video Loaded\n\n"
            "Please select a video source to start."
        )

    def setFramePixmap(self, pixmap: QPixmap) -> None:
        """Set the pixmap and repaint."""
        self._pixmap = pixmap
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        url = event.mimeData().urls()[0].toLocalFile()
        ext = Path(url).suffix.lower()
        if ext in {".mp4", ".avi", ".jpg", ".png"}:
            self.file_dropped.emit(url)
            # Setelah drop file, kembalikan ke teks instructive:
            self.setText(
                "No Video Loaded\n\n"
                "Please select a video source to start."
            )
        else:
            QMessageBox.warning(
                self, "Unsupported Format",
                "Only video (.mp4/.avi) or images (.jpg/.png) are supported."
            )

    def resizeEvent(self, event):
        """Handle resize event to trigger repaint."""
        super().resizeEvent(event)
        self.update()  # ini supaya widget repaint setelah resize

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self._pixmap.isNull():
            pixmap_aspect = self._pixmap.width() / self._pixmap.height()
            widget_aspect = self.width() / self.height()

            if pixmap_aspect > widget_aspect:
                new_width = self.width()
                new_height = int(new_width / pixmap_aspect)
                x_offset = 0
                y_offset = (self.height() - new_height) // 2
            else:
                new_height = self.height()
                new_width = int(new_height * pixmap_aspect)
                x_offset = (self.width() - new_width) // 2
                y_offset = 0

            scaled_pixmap = self._pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        else:
            super().paintEvent(event)
