# src/object_detector/gui/video_thread.py

import cv2
import numpy as np
import time
import logging
from threading import Lock
from PySide6.QtCore import QThread, Signal, QPointF
from PySide6.QtGui  import QImage

from ..config import Config
from ..preprocessing import letterbox
from ..visualization import draw_detections

logger = logging.getLogger(__name__)

class VideoThread(QThread):
    # emitted setiap kali frame baru siap ditampilkan
    change_pixmap_signal = Signal(QImage)
    # emitted jika terjadi error baca frame
    error_signal         = Signal(str)

    def __init__(
        self,
        source,
        detector,
        names,
        colors,
        conf_thres: float,
        enable_detect: bool = False,
        enable_count:  bool = False
    ):
        super().__init__()
        self.source       = source
        self.detector     = detector
        self.names        = names
        self.colors       = colors
        self.conf_thres   = conf_thres
        self.enable_detect = enable_detect
        self.enable_count  = enable_count

        # lock untuk mencegah race condition saat toggle flags
        self._flag_lock = Lock()

        # kelas kendaraan yang ingin dihitung
        self.vehicle_ids = [
            names.index(c) for c in Config.VEHICLE_CLASSES if c in names
        ]

        self.roi_poly   = None
        self.line_pts   = None
        self.count      = 0
        self.current_fps = 0.0
        self.counted    = set()
        self.last_frame = None

    @property
    def total_count(self):
        """Total kendaraan yang sudah terhitung."""
        return self.count

    def set_roi(self, polygon):
        """Tentukan ROI sebagai array Nx1x2 untuk cv2.fillPoly()."""
        coords = []
        for p in polygon:
            if isinstance(p, QPointF):
                coords.append((int(p.x()), int(p.y())))
            else:
                x, y = p
                coords.append((int(x), int(y)))
        arr = np.array(coords, dtype=np.int32)
        self.roi_poly = arr.reshape(-1, 1, 2)

    def set_line(self, line_pts):
        """Tentukan garis hitung, reset counter."""
        coords = []
        for p in line_pts:
            if isinstance(p, QPointF):
                coords.append((int(p.x()), int(p.y())))
            else:
                x, y = p
                coords.append((int(x), int(y)))
        self.line_pts = tuple(coords)
        self.count = 0
        self.counted.clear()
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        self._den  = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
        self._line = (x1, y1, x2, y2)

    def set_enable_detect(self, flag: bool):
        """Toggle enable_detect dengan thread-safety."""
        with self._flag_lock:
            self.enable_detect = flag

    def set_enable_count(self, flag: bool):
        """Toggle enable_count dengan thread-safety."""
        with self._flag_lock:
            self.enable_count = flag

    def run(self):
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS or 30
        frame_time = 1.0 / fps
        prev_time = time.time()

        while not self.isInterruptionRequested() and cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                self.error_signal.emit("Cannot read frame (stream ended or error).")
                break

            self.last_frame = frame

            # hitung FPS aktual
            dt = t0 - prev_time
            self.current_fps = (1.0 / dt) if dt > 0 else 0.0
            prev_time = t0

            # gambar ROI & garis pada frame
            frame = self._draw_roi_and_line(frame)

            # baca flags dengan lock
            with self._flag_lock:
                do_detect = self.enable_detect
                do_count  = self.enable_count

            # proses frame
            try:
                if do_detect:
                    qimg = self._process_frame(frame)
                else:
                    qimg = self._to_qimage(frame)
            except Exception:
                logger.exception("Error processing frame")
                qimg = self._to_qimage(frame)

            # kirim ke UI
            self.change_pixmap_signal.emit(qimg)

            # sesuaikan kecepatan loop agar sesuai FPS video
            elapsed = time.time() - t0
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        cap.release()

    def stop(self):
        """Hentikan thread dengan cepat."""
        self.requestInterruption()
        self.wait(1000)

    def _process_frame(self, frame: np.ndarray) -> QImage:
        """Lakukan preprocessing → inferensi → filtering → update count → visualisasi."""
        tensor, ratio, dwdh = letterbox(
            frame,
            new_shape=Config.INPUT_SIZE,
            color=Config.PAD_COLOR,
            auto=False,
            scaleup=True,
            stride=Config.STRIDE,
        )

        dets = self.detector.predict(tensor)
        if dets.size:
            ids  = dets[:, 5].astype(int)
            dets = dets[np.isin(ids, self.vehicle_ids)]

        dets = self._filter_by_roi(dets, ratio, dwdh)

        if self.enable_count:
            self._update_count(dets, ratio, dwdh)

        # gambar kotak deteksi
        img = draw_detections(
            frame, dets, ratio, dwdh,
            self.names, self.colors, self.conf_thres
        )

        # tulis teks hitungan
        if self.enable_count:
            cv2.putText(
                img, f"Count: {self.count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2
            )

        return self._to_qimage(img)

    def _draw_roi_and_line(self, frame: np.ndarray) -> np.ndarray:
        """Overlay ROI (fillPoly) dan garis hitung (line)."""
        if self.roi_poly is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.roi_poly], (128, 0, 128))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        if self.line_pts is not None:
            (x1, y1), (x2, y2) = self.line_pts
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def _filter_by_roi(self, dets, ratio, dwdh):
        """Buang deteksi yang pusat bounding box-nya di luar ROI."""
        if self.roi_poly is None or dets.size == 0:
            return dets
        contour = self.roi_poly.reshape(-1, 2)
        keep = []
        for *_, x0, y0, x1b, y1b, cls, conf in dets:
            cx = ((x0 + x1b) / 2 - dwdh[0]) / ratio
            cy = ((y0 + y1b) / 2 - dwdh[1]) / ratio
            keep.append(
                cv2.pointPolygonTest(contour, (int(cx), int(cy)), False) >= 0
            )
        return dets[np.array(keep)]

    def _update_count(self, dets, ratio, dwdh):
        """Hitung jumlah kendaraan yang melewati garis."""
        if self.line_pts is None or dets.size == 0:
            return
        x1, y1, x2, y2 = self._line
        for *_, x0, y0, x1b, y1b, cls, conf in dets:
            cx = ((x0 + x1b) / 2 - dwdh[0]) / ratio
            cy = ((y0 + y1b) / 2 - dwdh[1]) / ratio
            num  = abs((y2 - y1) * cx - (x2 - x1) * cy + x2*y1 - y2*x1)
            dist = num / self._den if self._den else float("inf")
            key  = (int(cx), int(cy))
            if dist < 5 and key not in self.counted:
                self.counted.add(key)
                self.count  += 1

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        """Konversi BGR image (NumPy) ke QImage RGB."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
