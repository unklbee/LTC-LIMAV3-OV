# src/object_detector/gui/line_drawer.py

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
from PySide6.QtGui    import QPixmap, QPainter, QPen, QMouseEvent
from PySide6.QtCore   import Qt, QPoint

class LineDrawer(QDialog):
    """
    Dialog untuk menggambar satu garis (dua titik).
    Klik sekali untuk titik awal, klik lagi untuk titik akhir.
    """
    def __init__(self, pixmap: QPixmap):
        super().__init__()
        self.setWindowTitle("Define Counting Line")
        self.orig   = pixmap
        self.label  = QLabel()
        self.label.setPixmap(self.orig)
        self.label.setFixedSize(self.orig.size())
        self.points = []
        self.label.installEventFilter(self)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)

    def eventFilter(self, obj, ev):
        if obj is self.label and isinstance(ev, QMouseEvent):
            if ev.type() == QMouseEvent.Type.MouseButtonPress and ev.button() == Qt.LeftButton:
                pt = ev.position().toPoint()
                self.points.append(pt)
                self._draw()
                if len(self.points) == 2:
                    self.accept()
            return False
        return super().eventFilter(obj, ev)

    def _draw(self):
        pix = self.orig.copy()
        painter = QPainter(pix)
        pen = QPen(Qt.green, 2)
        painter.setPen(pen)
        for p in self.points:
            painter.drawEllipse(p, 4, 4)
        if len(self.points) == 2:
            painter.drawLine(self.points[0], self.points[1])
        painter.end()
        self.label.setPixmap(pix)

    def get_line(self):
        if self.exec() == QDialog.Accepted and len(self.points) == 2:
            return [(p.x(), p.y()) for p in self.points]
        return None
