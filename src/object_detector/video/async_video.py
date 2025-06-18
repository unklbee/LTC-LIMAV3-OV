# src/object_detector/video/async_video.py
import os
from collections import deque
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue, Empty
import time, logging
from typing import Optional, Callable, Any, List, Tuple, Dict

import cv2, numpy as np

from ..db                import CountDatabase
from ..tracking.tracker  import create_tracker
from ..tracking.counting import update_count_tracks          # ← satu-satunya
from ..utils.geometry    import (
    filter_by_roi, denormalize_boxes, filter_classes, box_iou
)
from ..utils.drawing     import draw_roi_and_line, draw_tracks, draw_detections
from .capture            import create_capture
from ..inference         import detect
from ..config            import Config

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
class AsyncVideo:
    def __init__(
        self,
        source: Any,
        detector: Any,
        names:   List[str],
        colors:  List[Tuple[int,int,int]],
        conf_thres: float,
        enable_count: bool,
        vehicle_ids: List[int],
        save_interval_sec: Optional[int] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
        on_count: Optional[Callable[[Dict[int,int]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        db_path: str = None,
        host_id: str = None,
        api_url: str = None
    ):
        # ── config & callback
        self.source, self.detector = source, detector
        self._loop_source = isinstance(source, str) and os.path.isfile(source)
        self.names,  self.colors   = names, colors
        self.conf_thres            = conf_thres
        self.enable_detect         = False
        self.enable_count          = enable_count
        self.vehicle_ids           = set(vehicle_ids)
        self.on_frame, self.on_count, self.on_error = on_frame, on_count, on_error

        # ── sync primitives
        self._cnt_lock     = Lock()
        self._running      = Event(); self._running.set()
        self._save_enabled = Event()   # ← ditrigger GUI saat RUNNING
        self._stop_requested = Event()

        # ── tracking / counting state
        self.tracker        = create_tracker()
        self.track_sides    : Dict[int,int] = {}
        self.counted_tracks : set          = set()
        self.track_history  : Dict[int,deque] = {}
        self.track_classes  : Dict[int,int] = {}
        self.max_trace_len  = 20
        self.counts         = {cid: 0 for cid in self.vehicle_ids}

        # ── ROI & line
        self.roi_poly = None          # OpenCV format Nx1x2
        self.line_pts = None
        self._line    = (0,0,0,0)

        # ── video capture
        self.cap       = create_capture(source)
        fps            = self.cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS or 30
        self.frame_time= 1./fps

        # ── buffers
        self._buf_in = Queue(maxsize=2)

        # ── DB setup: gunakan parameter, fallback ke Config
        self.db_path = db_path or str(Config.DB_PATH)
        self.host_id = host_id or Config.HOST_ID
        # ambil URL dari param atau fallback ke Config.API_URL
        self.api_url = api_url or Config.API_URL
        self._save_interval = save_interval_sec or Config.COUNT_SAVE_INTERVAL_SEC
        self.camera_id = id(self)

        # Buat CountDatabase dengan host_id & api_url
        try:
            self.db = CountDatabase(
                self.db_path,
                host_id = self.host_id,
                api_url = self.api_url
            )
        except Exception as e:
            self.db = None
            logger.warning("CountDatabase init failed: %s", e)

        # ── worker threads
        self._capture_thread   = Thread(target=self._capture_loop,   daemon=True)
        self._inference_thread = Thread(target=self._inference_loop, daemon=True)
        self._save_thread      = Thread(target=self._periodic_save_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()
        self._save_thread.start()

        # ── perf
        self.current_fps = 0.0
        self._last_ts    = time.time()
        self._started = False

    # ───────────────────── public helpers ────────────────────────────
    def set_roi(self, polygon):
        self.roi_poly = np.array(
            [(int(p.x()), int(p.y())) if hasattr(p,"x") else (int(p[0]),int(p[1]))
             for p in polygon], dtype=np.int32
        ).reshape(-1,1,2)

    def set_line(self, pts):
        (x1,y1), (x2,y2) = [
            (int(p.x()), int(p.y())) if hasattr(p,"x") else (int(p[0]),int(p[1]))
            for p in pts
        ]
        self.line_pts = ((x1,y1),(x2,y2)); self._line = (x1,y1,x2,y2)
        self.reset_counts()

    def set_enable_detect(self, flag: bool): self.enable_detect = flag
    def set_enable_count (self, flag: bool): self.enable_count  = flag

    def set_save_enabled(self, flag: bool):
        if flag:
            self._save_enabled.set()
        else:
            if self._save_enabled.is_set():
                self._flush_counts_once()
            self._save_enabled.clear()

    # ───────────────────── thread: capture ───────────────────────────
    def _capture_loop(self):
        while self._running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self._loop_source:
                    # ulangi dari frame 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            if self._buf_in.full():
                try: self._buf_in.get_nowait()
                except Empty: pass
            self._buf_in.put(frame)
            time.sleep(self.frame_time)
        self._running.clear()

    # ───────────────────── thread: inference ─────────────────────────
    def _inference_loop(self):
        while self._running.is_set():
            try:
                frame = self._buf_in.get(timeout=1)
            except Empty:
                continue

            # — Jika bukan RUNNING → preview-only: hitung FPS + kirim frame →
            if not self.enable_detect:
                # 📈 update FPS
                now = time.time()
                dt = now - self._last_ts
                if dt > 0:
                    self.current_fps = 1.0 / dt
                self._last_ts = now

                #kirim frame preview
                if self.on_frame:
                    self.on_frame(frame)
                # skip seluruh pipeline deteksi/drawing
                continue

            # gambar ROI/line, tapi aman bila None atau error
            frame_copy = frame.copy()
            try:
                img = draw_roi_and_line(frame_copy, self.roi_poly, self.line_pts)
            except Exception:
                img = frame_copy

            if self.enable_detect:
                # seluruh block deteksi/drawing tracks dalam satu try,
                # tapi kesalahan di sini TIDAK mematikan loop
                try:
                    dets, ratio, dwdh = self._detect(img)
                    dets = filter_classes(dets, self.vehicle_ids)
                    dets = filter_by_roi(dets, self.roi_poly, ratio, dwdh)

                    if dets.size and self.enable_count:
                        # denorm & tracking
                        boxes   = denormalize_boxes(dets, ratio, dwdh, frame.shape[:2])
                        classes = dets[:,5].astype(int)
                        tracks  = self.tracker.update(boxes)

                        # map track → class
                        self.track_classes.clear()
                        if tracks.ndim==2 and tracks.shape[1]>=5:
                            tid_idx   = tracks.shape[1]-1
                            det_boxes = boxes[:,:4]
                            for tr in tracks:
                                tid  = int(tr[tid_idx])
                                ious = box_iou(tr[:4], det_boxes)
                                best = int(np.argmax(ious))
                                if ious[best] > .30:
                                    self.track_classes[tid] = classes[best]

                        # history
                        self._update_history(tracks)

                        # counting
                        with self._cnt_lock:
                            before = dict(self.counts)
                            update_count_tracks(
                                tracks, self._line, self.roi_poly,
                                self.counts, self.track_sides,
                                self.counted_tracks, self.track_classes,
                                self.names, None
                            )
                            changed = any(self.counts[c] > before.get(c,0)
                                          for c in self.counts)

                        if changed and self.on_count:
                            self.on_count(dict(self.counts))

                        draw_tracks(img, tracks, self.track_history,
                                    self.track_classes, self.names, self.colors)
                    elif dets.size:
                        draw_detections(img, dets, ratio, dwdh, self.names, self.colors)

                except Exception as e:
                    logger.exception("Error in pipeline")
                    # laporkan tapi jangan berhenti preview
                    if self.on_error: self.on_error(e)

            # selalu kirim frame ke UI, walau detect=False atau error
            if self.on_frame:
                self.on_frame(img)

            # FPS
            now=time.time()
            dt=now-self._last_ts
            if dt > 0:
                self.current_fps=1.0 / dt
            self._last_ts=now

    # ───────────────────── thread: DB flush ──────────────────────────
    def _periodic_save_loop(self):
        start_ts = datetime.now(Config.TIMEZONE)
        waited = 0.0
        step = 0.25
        last_mon = time.monotonic()
        while self._running.is_set():
            time.sleep(step)
            # hitung delta monotonic
            now_mon = time.monotonic()
            if not self._save_enabled.is_set():
                # jika tidak RUNNING, reset timer
                start_ts = datetime.now(Config.TIMEZONE)
                waited = 0.0
                last_mon = now_mon
                continue
            waited += (now_mon - last_mon)
            last_mon = now_mon

            if waited < self._save_interval:
                continue
            # cap timer untuk menghindari drift
            waited = 0.0
            end_ts = datetime.now(Config.TIMEZONE)

            self._flush_counts_once(start_ts, end_ts)
            start_ts = end_ts

    # ───────────────────── helpers ───────────────────────────────────
    def _detect(self, img): return detect(self.detector, img)

    def _update_history(self, tracks):
        tid_idx = tracks.shape[1]-1
        for tr in tracks:
            tid = int(tr[tid_idx])
            cx  = int((tr[0]+tr[2])/2); cy=int((tr[1]+tr[3])/2)
            self.track_history.setdefault(tid, deque(maxlen=self.max_trace_len)).append((cx,cy))

    def _flush_counts_once(self, start_ts=None, end_ts=None):
        if start_ts is None: start_ts=datetime.now(Config.TIMEZONE)
        if end_ts   is None: end_ts  =datetime.now(Config.TIMEZONE)

        with self._cnt_lock:
            snapshot = {k:v for k,v in self.counts.items() if v}
            if not snapshot: return
            try:
                if self.db:
                    self.db.insert_interval_counts(start_ts, end_ts, snapshot)
                ok=True
            except Exception as e:
                logger.warning("DB flush failed: %s", e); ok=False
            if ok: self.reset_counts(lock_already_held=True)

        if self.on_count: self.on_count(dict(self.counts))

    def reset_counts(self, *, lock_already_held=False):
        if not lock_already_held: self._cnt_lock.acquire()
        try:
            for k in self.counts: self.counts[k]=0
            self.counted_tracks.clear(); self.track_sides.clear(); self.track_history.clear()
        finally:
            if not lock_already_held: self._cnt_lock.release()

    def restart_preview(self):
        """
        Clear buffer & rewind source, matikan deteksi —
        hanya preview frame mentah.
        """
        # 1) Clear queue
        try:
            while True:
                self._buf_in.get_nowait()
        except Empty:
            pass

        # 2) Rewind kalau file
        if self._loop_source:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 3) Pastikan detect mati (hanya preview)
        self.enable_detect = False

    def set_save_interval(self, sec: int):
        """Update interval simpan (dalam detik) on the fly."""
        self._save_interval = sec

    def start(self):
        """
        Mulai pipeline deteksi / counting.
        Dipanggil sekali oleh MainWindow ketika user tekan Start pertama kali.
        """
        if not self._started:
            # baru pertama kali: reset counts, aktifkan save, enable detect+count
            self.reset_counts()
            self.set_save_enabled(True)
            self.enable_detect = True
            self.enable_count = True
            self._started = True
        else:
            # kalau sudah pernah start & kemudian pause, nyalakan kembali
            self.enable_detect = True
            self.enable_count = True
            self.set_save_enabled(True)

    def pause(self):
        """
        Pause pipeline—deteksi berhenti, tapi thread tetap jalan untuk preview.
        """
        self.enable_detect = False
        self.enable_count = False
        # flush ke DB sisa count sekarang
        self._flush_counts_once()
        self.set_save_enabled(False)

    def resume(self):
        """
        Lanjutkan dari pause—deteksi counting dilanjut tanpa reset.
        """
        # counts sudah di‐flush di pause, tapi counts internal setelah itu sudah nol
        self.enable_detect = True
        self.enable_count = True
        self.set_save_enabled(True)

    # ───────────────────── teardown ──────────────────────────────────
    def stop(self):
        self._stop_requested.set()
        self._running.clear()

        # Flush any remaining counts before shutting down workers
        if self._save_enabled.is_set():
            try:
                self._flush_counts_once()
            except Exception as e:  # pragma: no cover - log only
                logger.warning("Final count flush failed: %s", e)

        self.cap.release()
        self._capture_thread.join()
        self._inference_thread.join()
        self._save_thread.join()
