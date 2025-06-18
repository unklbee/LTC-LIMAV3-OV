# Directory: src/object_detector/gui/main_window.py
import sys, random, logging
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import openvino as ov  # Changed from onnxruntime
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QSplitter,
    QDialog, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox,
    QLabel
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage

from ..config   import Config
from ..inference import Detector
from ..video.async_video import AsyncVideo
from .ui_state import AppState, update_ui_by_state, notify_state_change
from .presets  import pts_to_serializable, read_preset_file, write_preset_file
from .coords   import serializable_to_pts
from .video_thread import VideoThread
from .roi_drawer   import PolyRoiDrawer
from .line_drawer  import LineDrawer
from .widgets.source_model_panel       import SourceModelPanel
from .widgets.detection_settings_panel import DetectionSettingsPanel
from .widgets.roi_line_panel           import RoiLinePanel
from .widgets.video_display            import VideoDisplay
from .widgets.status_bar               import StatusBar
from .widgets.database_settings_panel  import DatabaseSettingsPanel
from .widgets.menu_panel               import setup_menubar

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

@contextmanager
def block_signals(widgets: List[QWidget]):
    for w in widgets: w.blockSignals(True)
    yield
    for w in widgets: w.blockSignals(False)

# ──────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.last_count_text = "—"
        Config.ensure_dirs()
        self.setWindowTitle("LIMA - Traffic Counter (OV1)")
        self.step_labels = []
        step_bar = QWidget();
        hb = QHBoxLayout(step_bar)
        for title in ("Source", "ROI", "Line", "Start"):
            lbl = QLabel(title)
            hb.addWidget(lbl)
            self.step_labels.append(lbl)
        self.statusBar().addPermanentWidget(step_bar)
        self.resize(1200, 700)

        # ── load models & classes - Look for .xml files first, then .onnx as fallback
        self.model_paths = sorted(Path(Config.MODEL_DIR).glob("*.xml"))
        if not self.model_paths:
            # Fallback to ONNX files if no OpenVINO models found
            self.model_paths = sorted(Path(Config.MODEL_DIR).glob("*.onnx"))
            if not self.model_paths:
                QMessageBox.critical(self, "Error",
                                     f"No OpenVINO (.xml) or ONNX (.onnx) models found in {Config.MODEL_DIR}\n\n"
                                     f"Please convert your ONNX models to OpenVINO format using:\n"
                                     f"python convert_models.py")
                sys.exit(1)
            else:
                QMessageBox.information(self, "Info",
                                        f"Using ONNX models as fallback. For better performance, "
                                        f"convert to OpenVINO format using: python convert_models.py")

        self.names  = Path(Config.NAMES_PATH).read_text().splitlines()
        self.colors = [tuple(random.randint(0,255) for _ in range(3)) for _ in self.names]

        # ── OpenVINO device map instead of providers
        self.device_map: Dict[str, str] = {}
        self._refresh_devices()

        # ── UI widgets
        setup_menubar(self)
        # ── Add Database Settings menu item
        db_act = self.menuBar().addAction("Database Settings")
        db_act.triggered.connect(self._open_db_settings)

        self.src_panel      = SourceModelPanel([p.name for p in self.model_paths],
                                               list(self.device_map.keys()))
        self.det_panel      = DetectionSettingsPanel(initial=int((1.0-Config.CONFIDENCE_THRESHOLD)*100))
        self.roi_line_panel = RoiLinePanel()
        self.video_disp     = VideoDisplay()
        self.start_btn      = QPushButton("Start"); self.start_btn.setFixedHeight(40)
        self.status_bar     = StatusBar(); self.setStatusBar(self.status_bar)

        ctrl = QWidget(); lay = QVBoxLayout(ctrl)
        for w in (self.src_panel, self.det_panel, self.roi_line_panel): lay.addWidget(w)
        lay.addStretch(); lay.addWidget(self.start_btn)

        splitter = QSplitter(Qt.Horizontal); splitter.addWidget(ctrl); splitter.addWidget(self.video_disp)
        splitter.setStretchFactor(1,1); self.setCentralWidget(splitter)

        # signals
        self.src_panel.source_changed.connect(self._on_source)
        self.src_panel.model_changed .connect(self._on_model)
        self.src_panel.device_changed.connect(self._on_device)
        self.det_panel.conf_changed   .connect(self._on_conf)
        self.roi_line_panel.roi_requested .connect(self._on_define_roi)
        self.roi_line_panel.line_requested.connect(self._on_define_line)
        self.start_btn.clicked.connect(self._on_toggle)

        # internal state
        self.state = AppState.INIT
        self.current_source = None
        self._saved_roi = None
        self._saved_line = None
        self.async_vid    = None
        self.video_thread = None

        # detector pertama
        self._reload_detector()
        self._set_state(self.state)

    # ───────────────────────── helper state & UI ─────────────────────
    def _set_state(self, new_state: AppState):
        self.state = new_state
        update_ui_by_state(self, self.state)
        notify_state_change(self, self.state)

    # ───────────────────────── OpenVINO devices/model/conf ──────────────────
    def _refresh_devices(self):
        """Get available OpenVINO devices instead of ONNX providers"""
        try:
            core = ov.Core()
            available_devices = core.available_devices

            self.device_map.clear()

            # Map available devices to user-friendly names
            for device in available_devices:
                if "GPU" in device:
                    self.device_map["Intel GPU"] = "GPU"
                elif "CPU" in device:
                    self.device_map["CPU"] = "CPU"
                elif "NPU" in device:
                    self.device_map["Intel NPU"] = "NPU"

            # Always add AUTO as it works on all systems
            self.device_map["Auto"] = "AUTO"

            # Ensure we have at least CPU
            if "CPU" not in self.device_map:
                self.device_map["CPU"] = "CPU"

        except Exception as e:
            logger.warning(f"Failed to get OpenVINO devices: {e}")
            # Fallback device mapping
            self.device_map = {"CPU": "CPU", "Auto": "AUTO"}

        # Update combobox if it exists
        if hasattr(self, "src_panel"):
            self.src_panel.device_cb.clear()
            self.src_panel.device_cb.addItems(self.device_map.keys())

    def _reload_detector(self):
        self.status_bar.showMessage("Loading model...")
        QApplication.processEvents()

        idx = self.src_panel.model_cb.currentIndex()
        model_pth = str(self.model_paths[idx])

        # Get OpenVINO device instead of providers
        dev_key = self.src_panel.device_cb.currentText()
        if dev_key not in self.device_map:
            dev_key = next(iter(self.device_map))
            self.src_panel.device_cb.setCurrentText(dev_key)

        device = self.device_map[dev_key]

        try:
            # Use OpenVINO device instead of providers list
            self.detector = Detector(model_pth, device)
            self.status_bar.showMessage(f"Model ready on {dev_key}")
        except Exception as e:
            logger.exception("Init Detector failed")
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}\n\nTry a different device or convert model to OpenVINO format.")
            return

        th = 1.0 - (self.det_panel.slider.value()/100.0)
        self.detector.conf_thres = th

    # ─────────────────────────── source/model/dev/conf ───────────────
    def _on_source(self, src): self.current_source = src; self._start_preview()
    def _on_model (self, _):   self._reload_detector();   self._update_async_detector()
    def _on_device(self, _):   self._reload_detector();   self._update_async_detector()
    def _on_conf  (self, val): th=1.0 - (val/100.0); self.detector.conf_thres=th

    def _update_async_detector(self):
        if self.async_vid: self.async_vid.detector = self.detector

    # ─────────────────────────── preview & async video ───────────────
    def _start_preview(self):
        # Stop old preview if exists
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        if self.async_vid:
            self.async_vid.stop()
            self.async_vid = None

        # Create new AsyncVideo based on current_source & detector
        self.async_vid = AsyncVideo(
            source=self.current_source,
            detector=self.detector,
            names=self.names,
            colors=self.colors,
            conf_thres=self.detector.conf_thres,
            enable_count=False,
            vehicle_ids=[self.names.index(c) for c in Config.VEHICLE_CLASSES if c in self.names],
            save_interval_sec=Config.COUNT_SAVE_INTERVAL_SEC,
            db_path=str(Config.DB_PATH),
            host_id=Config.HOST_ID,
            api_url=Config.API_URL,
            on_frame=None,
            on_count=None,
            on_error=None,
        )

        # Ensure clean buffer & rewind (preview only)
        self.async_vid.restart_preview()

        # Create VideoThread and connect signals → GUI slots
        self.video_thread = VideoThread(self.async_vid)
        self.async_vid.on_frame = self.video_thread.frame_ready.emit
        self.async_vid.on_count = self.video_thread.count_ready.emit
        self.async_vid.on_error = lambda e: self.video_thread.error.emit(str(e))

        self.video_thread.frame_ready.connect(self._on_qt_frame)
        self.video_thread.count_ready.connect(self._on_qt_count)
        self.video_thread.error      .connect(self._on_qt_error)

        # Start capture + inference thread
        self.video_thread.start()

        # Set UI state: if both ROI+Line → READY, if only one → respective state
        if self._saved_roi and self._saved_line:
            self._set_state(AppState.READY)
        elif self._saved_roi:
            self._set_state(AppState.ROI_DEFINED)
        elif self._saved_line:
            self._set_state(AppState.LINE_DEFINED)
        else:
            self._set_state(AppState.SOURCE_SET)

    # ─────────────────────────── Start/Stop button ───────────────────
    @Slot()
    def _on_toggle(self):
        if self.state == AppState.INIT or self.state in (AppState.SOURCE_SET, AppState.ROI_DEFINED,
                                                         AppState.LINE_DEFINED):
            # Not ready to start
            QMessageBox.information(self, "Info", "Please define both ROI & Line first.")
            return

        # READY → START for the first time
        if self.state == AppState.READY:
            self.async_vid.start()
            self._set_state(AppState.RUNNING)
            return

        # RUNNING → PAUSE
        if self.state == AppState.RUNNING:
            if QMessageBox.question(self, "Confirm", "Stop vehicle counting?",
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self.async_vid.pause()
                self._set_state(AppState.PAUSED)
            return

        # PAUSED → RESUME
        if self.state == AppState.PAUSED:
            self.async_vid.resume()
            self._set_state(AppState.RUNNING)
            return

    # ─────────────────────────── Qt slots (frame/count/error) ────────
    @Slot(object)
    def _on_qt_frame(self, frame: np.ndarray):
        h,w = frame.shape[:2]
        qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
        self.video_disp.setFramePixmap(QPixmap.fromImage(qimg))
        # Update FPS every frame, but use last count text
        fps = getattr(self.async_vid, "current_fps", 0.0)
        self.status_bar.update(fps, self.last_count_text)

    @Slot(dict)
    def _on_qt_count(self, counts: Dict[int,int]):
        # Build count string, save, and update UI (including FPS)
        self.last_count_text = (
            "—" if not counts
            else "   ".join(f"{self.names[i]}:{c}" for i, c in counts.items())
        )
        fps = getattr(self.async_vid, "current_fps", 0.0)
        self.status_bar.update(fps, self.last_count_text)

    @Slot(str)
    def _on_qt_error(self, msg):
        QMessageBox.critical(self,"Error", msg)
        if self.video_thread: self.video_thread.stop(); self.video_thread=None
        self._set_state(AppState.READY)

    # ─────────────────────────── ROI & Line drawer ───────────────────
    @Slot()
    def _on_define_roi(self):
        if not self.video_disp._pixmap or self.video_disp._pixmap.isNull():
            QMessageBox.warning(self,"Warning","No frame available for ROI."); return
        poly = PolyRoiDrawer(self.video_disp._pixmap).get_polygon()
        if not poly: return
        self._saved_roi=poly; self.async_vid.set_roi(poly)
        self._set_state(AppState.READY if self._saved_line else AppState.ROI_DEFINED)

    @Slot()
    def _on_define_line(self):
        if not self.video_disp._pixmap or self.video_disp._pixmap.isNull():
            QMessageBox.warning(self,"Warning","No frame available for Line."); return
        line = LineDrawer(self.video_disp._pixmap).get_line()
        if not line: return
        self._saved_line=line; self.async_vid.set_line(line)
        new_state = AppState.READY if self._saved_roi else AppState.LINE_DEFINED
        self._set_state(new_state)

    @Slot()
    def _open_db_settings(self):
        """Open Database Settings dialog from menu bar"""
        dlg = QDialog(self)
        dlg.setWindowTitle("Database Settings")
        layout = QVBoxLayout(dlg)
        panel = DatabaseSettingsPanel()
        layout.addWidget(panel)
        dlg.exec()

        # After dialog closes, apply new settings
        if self.async_vid:
            # Update host_id in DB writer
            if self.async_vid.db:
                # update host_id
                self.async_vid.db.set_host_id(Config.HOST_ID)
                # update API URL
                self.async_vid.db.set_api_url(Config.API_URL)
            # Update save interval (seconds) in AsyncVideo
            self.async_vid.set_save_interval(Config.COUNT_SAVE_INTERVAL_SEC)

    def save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON Files (*.json)")
        if not path:
            return

        kind = self.src_panel.source_cb.currentText()
        val = int(self.current_source) if kind.lower().startswith("webcam") else self.current_source
        threshold = getattr(self.detector, 'conf_thres', Config.CONFIDENCE_THRESHOLD)

        preset = {
            "source_kind": kind,
            "source_value": val,
            "model": self.src_panel.model_cb.currentText(),
            "device": self.src_panel.device_cb.currentText(),
            "confidence_threshold": threshold,
            "roi": pts_to_serializable(self._saved_roi or []),
            "line": pts_to_serializable(self._saved_line or []),

            # Database Settings
            "db_path": str(Config.DB_PATH),
            "save_interval_sec": Config.COUNT_SAVE_INTERVAL_SEC,
            "host_id": Config.HOST_ID,
            "api_url": Config.API_URL,
        }

        try:
            write_preset_file(path, preset)
            self.status_bar.showMessage(f"Preset saved to: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preset:\n{e}")

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON Files (*.json)")
        if not path:
            return

        try:
            preset = read_preset_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset:\n{e}")
            return

        # Reset UI + threads
        self.reset()

        # Apply source/model/device without signals
        with block_signals([self.src_panel.source_cb,
                            self.src_panel.model_cb,
                            self.src_panel.device_cb]):
            self.src_panel.source_cb.setCurrentText(preset["source_kind"])
            self.current_source = (
                0 if preset["source_kind"].lower().startswith("webcam")
                else preset["source_value"]
            )
            if preset["model"] in [p.name for p in self.model_paths]:
                self.src_panel.model_cb.setCurrentText(preset["model"])
            if preset["device"] in self.device_map:
                self.src_panel.device_cb.setCurrentText(preset["device"])

        # Apply confidence
        self.det_panel.slider.setValue(int((1.0 - preset["confidence_threshold"]) * 100))

        # Apply DB settings
        from pathlib import Path
        Config.DB_PATH = Path(preset["db_path"])
        Config.COUNT_SAVE_INTERVAL_SEC = preset["save_interval_sec"]
        Config.HOST_ID = preset["host_id"]
        Config.API_URL = preset["api_url"]

        # Update running AsyncVideo if any
        if self.async_vid and self.async_vid.db:
            self.async_vid.db.set_host_id(Config.HOST_ID)
            self.async_vid.db.set_api_url(Config.API_URL)
            self.async_vid.set_save_interval(Config.COUNT_SAVE_INTERVAL_SEC)

        QApplication.processEvents()

        # Restart preview & apply ROI/Line
        self._start_preview()
        self.async_vid.restart_preview()

        roi_pts = serializable_to_pts(preset["roi"])
        line_pts = serializable_to_pts(preset["line"])
        if len(roi_pts) >= 3:
            self._saved_roi = roi_pts
            self.async_vid.set_roi(roi_pts)
            self._set_state(AppState.ROI_DEFINED)
        if len(line_pts) >= 2:
            self._saved_line = line_pts
            self.async_vid.set_line(line_pts)
            new_state = (
                AppState.READY if self.state == AppState.ROI_DEFINED
                else AppState.LINE_DEFINED
            )
            self._set_state(new_state)

        self.status_bar.showMessage(f"Preset loaded from: {path}")

    def reset(self):
        if self.video_thread: self.video_thread.stop(); self.video_thread=None
        if self.async_vid: self.async_vid.stop(); self.async_vid=None
        self.current_source=None; self._saved_roi=None; self._saved_line=None
        self.src_panel.source_cb.setCurrentIndex(0); self.src_panel.model_cb.setCurrentIndex(0)
        self.src_panel.device_cb.clear(); self._refresh_devices(); self.src_panel.device_cb.addItems(self.device_map.keys()); self.src_panel.device_cb.setCurrentIndex(0)
        self.det_panel.slider.setValue(int((1.0-Config.CONFIDENCE_THRESHOLD)*100))
        self.video_disp._pixmap=QPixmap(); self.video_disp.clear(); self.video_disp.setText("No Video Loaded\n\nPlease select a video source to start.")
        self.video_disp.update()
        try: self.roi_line_panel.clear()
        except: pass
        self.state=AppState.INIT; update_ui_by_state(self,self.state)
        self.status_bar.update(fps=0.0,count_text="—")
        self._refresh_devices()

    def closeEvent(self, event):
        # 1) Stop VideoThread (Qt thread for preview & count)
        if hasattr(self, "video_thread") and self.video_thread:
            try:
                self.video_thread.stop()
            except AttributeError:
                self.video_thread.quit()
            self.video_thread.wait()

        # 2) Stop AsyncVideo internal worker threads
        if hasattr(self, "async_vid") and self.async_vid:
            self.async_vid.stop()

        # 3) Continue close to parent
        super().closeEvent(event)

    @Slot()
    def _show_about(self) -> None:
        QMessageBox.information(self, "About",
                                "LIMA - Traffic Counter\n"
                                "Version 1.0\n"
                                "Powered by Lintas Mediatama\n\n"
                                f"Available devices: {', '.join(self.device_map.keys())}")

    def _show_help(self) -> None:
        help_text = """
        LIMA Traffic Counter - OV Edition
        
        Quick Start Guide:
        1. Select video source (webcam or file)
        2. Choose device (Intel GPU recommended for better performance)
        3. Select detection model (.xml or .onnx)
        4. Define ROI (Region of Interest) on the video
        5. Define counting line across the traffic path
        6. Press Start to begin vehicle detection and counting
        
        Performance Tips:
        • Use Intel GPU for 2-3x better performance
        • Convert ONNX models to OpenVINO (.xml) format for optimization
        • Adjust confidence threshold for better detection accuracy
        
        Keyboard Shortcuts:
        • Ctrl+S: Save preset
        • Ctrl+O: Load preset  
        • Ctrl+R: Reset application
        • Ctrl+Q: Exit application
        """
        QMessageBox.information(self, "User Guide", help_text)

def main():
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__=='__main__':
    main()