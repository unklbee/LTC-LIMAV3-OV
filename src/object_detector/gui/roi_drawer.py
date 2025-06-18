# src/object_detector/gui/roi_drawer.py

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QPushButton
from PySide6.QtGui    import QPixmap, QPainter, QPen
from PySide6.QtCore   import Qt, QPoint, QEvent


class PolyRoiDrawer(QDialog):
    """
    Dialog untuk menggambar polygonal ROI.
    Klik kiri sekali untuk tambah titik, double‐klik untuk selesai.
    """
    def __init__(self, pixmap: QPixmap):
        super().__init__()
        self.setWindowTitle("Define Polygon ROI")
        self.orig = pixmap
        self.points = []  # list[QPoint]

        # Setup label untuk gambar
        self.label = QLabel(self)
        self.label.setPixmap(self.orig)
        self.label.setFixedSize(self.orig.size())
        # Pastikan kita dapat menerima mouse press/double‐click
        self.label.setMouseTracking(True)
        self.label.installEventFilter(self)

        # OK / Cancel buttons
        btn_ok     = QPushButton("OK", self)
        btn_cancel = QPushButton("Cancel", self)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(btn_ok)
        lay.addWidget(btn_cancel)
        self.setLayout(lay)

    def eventFilter(self, obj, ev):
        # hanya tangani event pada self.label
        if obj is self.label and ev.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            # klik kiri sekali → tambah titik
            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                pt: QPoint = ev.pos()
                self.points.append(pt)
                self._draw_polygon()
                return True

            # double‐click mana saja → selesai
            if ev.type() == QEvent.MouseButtonDblClick:
                # pastikan minimal 3 titik
                if len(self.points) >= 3:
                    self.accept()
                return True

        # fallback
        return super().eventFilter(obj, ev)

    def _draw_polygon(self):
        """Gambar titik dan garis per segmen di label."""
        pix = self.orig.copy()
        p = QPainter(pix)
        pen = QPen(Qt.red, 2)
        p.setPen(pen)

        # gambar titik dan garis
        for i, pt in enumerate(self.points):
            p.drawEllipse(pt, 3, 3)
            if i > 0:
                p.drawLine(self.points[i - 1], pt)
        p.end()

        self.label.setPixmap(pix)

    def get_polygon(self):
        """
        Tampilkan dialog, kembalikan daftar (x,y) atau None.
        """
        if self.exec() == QDialog.Accepted:
            return [(pt.x(), pt.y()) for pt in self.points]
        return None
