import threading
from datetime import datetime
import cv2
import numpy as np
import time
import logging
from queue import Queue, Empty
from threading import Thread, Event
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Callable

from .config import Config
from .db import CountDatabase
from .preprocessing import letterbox
from sort_tracker import Sort  # pip install sort-tracker

logger = logging.getLogger(__name__)

def _box_iou(box, boxes):
    """
    box: [x0,y0,x1,y1], boxes: Nx4 array
    return: array IoU length N
    """
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


class AsyncVideo:
    """
    Pipeline asinkron: capture → inferensi/tracking/counting → output.
    """

    def __init__(
        self,
        source: Any,
        detector: Any,
        names: List[str],
        colors: List[Tuple[int, int, int]],
        conf_thres: float,
        enable_count: bool,
        vehicle_ids: List[int],
        save_interval_sec: Optional[int] = None,
        on_frame: Optional[Callable[[np.ndarray], None]]    = None,
        on_count: Optional[Callable[[Dict[int,int]], None]] = None,
        on_error: Optional[Callable[[Exception], None]]     = None,
    ):
        # — core config —
        self.source       = source
        self.detector     = detector
        self.names        = names
        self.colors       = colors
        self.conf_thres   = conf_thres
        self.enable_detect = False
        self.enable_count  = enable_count
        self.vehicle_ids   = set(vehicle_ids)
        self._last_counts_snapshot = {}

        # — callbacks —
        self.on_frame = on_frame
        self.on_count = on_count
        self.on_error = on_error

        # — thread‐safety lock —
        self._lock = threading.Lock()

        # — tracker & counting state —
        self.tracker        = Sort(max_age=5, min_hits=1, iou_threshold=0.3)
        self.track_sides    : Dict[int, float] = {}
        self.counted_tracks : set[int]         = set()
        self.track_history  : Dict[int, deque] = {}
        self.track_classes  : Dict[int, int]   = {}
        self.max_trace_len  = 20
        self.counts         : Dict[int,int]    = {cid: 0 for cid in self.vehicle_ids}

        # — ROI & line placeholders —
        self.roi_poly = None
        self.line_pts = None
        self._line    = (0, 0, 0, 0)

        # — FPS timing setup —
        self.current_fps = 0.0
        self._last_ts    = time.time()

        # — Video I/O: paksa MSMF, fallback DSHOW —
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source!r}")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS or 30
        self.frame_time = 1.0 / fps

        # — input buffer —
        self._buf_in = Queue(maxsize=2)

        # — start worker threads —
        self._running = Event()
        self._running.set()
        Thread(target=self._capture_loop,   daemon=True).start()
        Thread(target=self._inference_loop, daemon=True).start()

        # — optional DB periodic save —
        self.db = None
        try:
            self.db = CountDatabase(str(Config.DB_PATH))
        except Exception as e:
            logger.warning(f"CountDatabase init failed: {e}")
        self._save_interval = save_interval_sec or Config.COUNT_SAVE_INTERVAL_SEC


    def set_roi(self, polygon: List[Any]):
        coords = []
        for p in polygon:
            if hasattr(p, "x") and hasattr(p, "y"):
                coords.append((int(p.x()), int(p.y())))
            else:
                x,y = p
                coords.append((int(x), int(y)))
        arr = np.array(coords, dtype=np.int32)
        self.roi_poly = arr.reshape(-1,1,2)

    def set_line(self, pts: List[Any]):
        coords = []
        for p in pts:
            if hasattr(p, "x") and hasattr(p, "y"):
                coords.append((int(p.x()), int(p.y())))
            else:
                x,y = p
                coords.append((int(x), int(y)))
        if len(coords)!=2:
            raise ValueError("Line requires exactly 2 points")
        (x1,y1),(x2,y2) = coords
        self.line_pts = ((x1,y1),(x2,y2))
        self._line    = (x1,y1,x2,y2)
        # reset counts/history
        with self._lock:
            for k in self.counts: self.counts[k]=0
        self.track_sides.clear()
        self.counted_tracks.clear()
        self.track_history.clear()

    def set_enable_detect(self, flag: bool):
        self.enable_detect = flag

    def set_enable_count(self, flag: bool):
        self.enable_count = flag

    def stop(self):
        self._running.clear()
        self.cap.release()
        if self.db:
            self.db.close()
        logger.info("AsyncVideo stopped and resources released.")

    def _capture_loop(self):
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._running.clear()
                break
            if self._buf_in.full():
                try: self._buf_in.get_nowait()
                except Empty: pass
            self._buf_in.put(frame)
            time.sleep(self.frame_time)

    def _inference_loop(self):
        while self._running.is_set():
            try:
                frame = self._buf_in.get(timeout=1)
            except Empty:
                continue

            img = self._draw_roi_and_line(frame.copy())
            if self.enable_detect:
                try:
                    dets, ratio, dwdh = self._detect(img)
                    dets = self._filter_vehicles(dets)
                    dets = self._filter_by_roi(dets, ratio, dwdh)
                    if dets.size and self.enable_count:
                        boxes = self._denormalize_boxes(dets, ratio, dwdh)
                        classes = dets[:, 5].astype(int)
                        tracks = self.tracker.update(boxes)
                        # map track→class
                        self.track_classes.clear()
                        if tracks.ndim == 2 and tracks.shape[1] >= 5:
                            tid_idx = tracks.shape[1] - 1
                            det_boxes = boxes[:, :4]
                            for tr in tracks:
                                tid = int(tr[tid_idx])
                                ious = _box_iou(tr[:4], det_boxes)
                                best = np.argmax(ious)
                                self.track_classes[tid] = classes[best] if ious[best] > 0.3 else None
                        self._update_history(tracks)
                        self._update_count_tracks(tracks)
                        self._draw_tracks(img, tracks)

                except Exception as e:
                    logger.exception("Error in pipeline")
                    if self.on_error:
                        self.on_error(e)

            # frame callback
            if self.on_frame:
                self.on_frame(img)

            # update FPS
            now = time.time()
            dt = now - self._last_ts
            if dt > 0:
                self.current_fps = 1 / dt
            self._last_ts = now

    def _detect(self, img: np.ndarray):
        """Run letterbox + ONNX inference."""
        tensor, ratio, dwdh = letterbox(
            img, new_shape=Config.INPUT_SIZE,
            color=Config.PAD_COLOR, auto=False,
            scaleup=True, stride=Config.STRIDE
        )
        dets = self.detector.predict(tensor)
        return dets, ratio, dwdh

    def _filter_vehicles(self, dets: np.ndarray) -> np.ndarray:
        """Keep only configured vehicle classes."""
        if dets.size == 0: return dets
        cls_ids = dets[:, 5].astype(int)
        mask = np.isin(cls_ids, list(self.vehicle_ids))
        return dets[mask]

    def _denormalize_boxes(
            self, dets: np.ndarray,
            ratio: float, dwdh: Tuple[float, float]
    ) -> np.ndarray:
        """Convert padded coords → original pixel boxes [x1,y1,x2,y2,score]."""
        dw, dh = dwdh
        out = []
        for *_, x0p, y0p, x1p, y1p, cls, conf in dets:
            x0 = (x0p - dw) / ratio
            y0 = (y0p - dh) / ratio
            x1 = (x1p - dw) / ratio
            y1 = (y1p - dh) / ratio
            out.append([x0, y0, x1, y1, float(conf)])
        return np.array(out, dtype=float)

    def _update_history(self, tracks: np.ndarray):
        """Append centroid to history deque per track_id."""
        tid_idx = tracks.shape[1] - 1
        for tr in tracks:
            tid = int(tr[tid_idx])
            cx = int((tr[0] + tr[2]) / 2)
            cy = int((tr[1] + tr[3]) / 2)
            hist = self.track_history.setdefault(tid, deque(maxlen=self.max_trace_len))
            hist.append((cx, cy))

    def _draw_tracks(self, img: np.ndarray, tracks: np.ndarray):
        """Draw tails + boxes + class name & ID for each track."""
        tid_idx = tracks.shape[1] - 1
        for tr in tracks:
            tid = int(tr[tid_idx])
            color = self.colors[tid % len(self.colors)]

            # 1) Draw tail
            pts = list(self.track_history.get(tid, []))
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2)

            # 2) Draw box
            x0, y0, x1, y1 = map(int, tr[:4])
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            # 3) Determine label = "class_name:ID" or fallback "ID<ID>"
            cls_id = self.track_classes.get(tid)
            if cls_id is not None and 0 <= cls_id < len(self.names):
                cls_name = self.names[cls_id]
                label = f"{cls_name}:{tid}"
            else:
                label = f"ID{tid}"

            # 4) Draw label
            cv2.putText(
                img, label,
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

    def _filter_by_roi(
            self, dets: np.ndarray,
            ratio: float, dwdh: Tuple[float, float]
    ) -> np.ndarray:
        """Keep only detections whose centroids fall inside ROI polygon."""
        if self.roi_poly is None or dets.size == 0:
            return dets
        contour = self.roi_poly.reshape(-1, 2)
        dw, dh = dwdh
        mask = []
        for *_, x0p, y0p, x1p, y1p, cls, conf in dets:
            cx = ((x0p + x1p) / 2 - dw) / ratio
            cy = ((y0p + y1p) / 2 - dh) / ratio
            mask.append(cv2.pointPolygonTest(contour, (int(cx), int(cy)), False) >= 0)
        return dets[np.array(mask)]

    def _update_count_tracks(self, tracks: np.ndarray):
        if self.line_pts is None or self.roi_poly is None or tracks.size == 0:
            print("ROI atau Line belum diatur! Pastikan kedua pengaturan ini telah diatur.")
            return
        x1, y1, x2, y2 = self._line
        A, B, C = (y2 - y1), (x1 - x2), (x2 * y1 - x1 * y2)
        tid_idx = tracks.shape[1] - 1

        for tr in tracks:
            tid = int(tr[tid_idx])
            cx, cy = (tr[0] + tr[2]) / 2, (tr[1] + tr[3]) / 2
            side = A * cx + B * cy + C
            prev = self.track_sides.get(tid)
            if prev is None:
                self.track_sides[tid] = side
            else:
                if prev * side < 0 and tid not in self.counted_tracks:
                    self.counted_tracks.add(tid)
                    cls_id = self.track_classes.get(tid)
                    if cls_id is not None and cls_id in self.counts:
                        self.counts[cls_id] += 1
                        print(f"[DEBUG] Counted {self.names[cls_id]}! Total {self.counts[cls_id]}")
                        # 🚀 Kirim count ke GUI langsung di sini:
                        if self.on_count:
                            snapshot = {i: c for i, c in self.counts.items() if c > 0}
                            self.on_count(snapshot)
                self.track_sides[tid] = side

    def _periodic_save_loop(self):
        # mulai loop selama AsyncVideo masih berjalan
        while self._running.is_set():
            # catat waktu mulai interval
            start_ts = datetime.now(Config.TIMEZONE)
            # tunggu interval yang sudah Anda set (misal 5 menit)
            time.sleep(self._save_interval)
            # catat waktu akhir interval
            end_ts = datetime.now(Config.TIMEZONE)

            # ambil snapshot counts dan reset atomik
            with self._lock:
                snapshot = dict(self.counts)
                # ———————————————————————————————
                #  PANGGIL KE DATABASE DI SINI  ←←←
                # ———————————————————————————————
                try:
                    # method ini sesuai dengan definisi di CountDatabase
                    self.db.insert_interval_counts(start_ts, end_ts, snapshot)
                except Exception as e:
                    logger.warning(f"DB save failed: {e}")

                # reset counters setelah disimpan
                for k in self.counts:
                    self.counts[k] = 0

            # (opsional) kirim update kosong ke UI supaya status bar flush
            if self.on_count:
                self.on_count({})

    def _draw_roi_and_line(self, frame: np.ndarray) -> np.ndarray:
        """Overlay ROI polygon and counting line."""
        if self.roi_poly is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.roi_poly], (128, 0, 128))
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        if self.line_pts is not None:
            (x1, y1), (x2, y2) = self.line_pts
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_ts
        if dt > 0:
            self.current_fps = 1.0 / dt
        self._last_ts = now
