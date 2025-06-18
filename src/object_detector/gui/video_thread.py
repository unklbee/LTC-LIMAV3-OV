# src/object_detector/gui/video_thread.py

from PySide6.QtCore import QThread, Signal

class VideoThread(QThread):
    frame_ready = Signal(object)   # tetap
    count_ready = Signal(object)   # _ubah_ dari Signal(dict)
    error       = Signal(str)

    def __init__(self, async_vid):
        super().__init__()
        self.async_vid = async_vid

    def run(self):
        # cukup buka event loop agar thread hidup sampai quit() dipanggil
        self.exec()

    def stop(self):
        # berhentikan AsyncVideo dan thread Qt-nya
        self.async_vid.stop()
        self.quit()
        self.wait()
