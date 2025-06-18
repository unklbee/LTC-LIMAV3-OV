from enum import Enum, auto
from typing import Any

class AppState(Enum):
    INIT = auto()
    SOURCE_SET = auto()
    ROI_DEFINED = auto()
    LINE_DEFINED = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()

# mapping utama: widget enable/disable + teks tombol
_STATE_UI = {
    AppState.INIT:        {"src": True,  "det": True,  "roi": False, "btn_text": "Start", "btn_en": False},
    AppState.SOURCE_SET:  {"src": True,  "det": True,  "roi": True,  "btn_text": "Start", "btn_en": False},
    AppState.ROI_DEFINED: {"src": True,  "det": True,  "roi": True,  "btn_text": "Start", "btn_en": False},
    AppState.LINE_DEFINED:{"src": True,  "det": True,  "roi": True,  "btn_text": "Start", "btn_en": False},
    AppState.READY:       {"src": True,  "det": True,  "roi": True,  "btn_text": "Start", "btn_en": True},
    AppState.RUNNING:     {"src": False, "det": False, "roi": False, "btn_text": "Stop",  "btn_en": True},
    AppState.PAUSED:      {"src": True, "det": True, "roi": True, "btn_text": "Resume", "btn_en": True},
}

# reminder / hint untuk status bar
_STATE_HINTS = {
    AppState.INIT:        "Pilih sumber dan model terlebih dahulu.",
    AppState.SOURCE_SET:  "Sekarang: gambar ROI pada frame.",
    AppState.ROI_DEFINED: "Sekarang: gambar Line untuk hitung objek.",
    AppState.LINE_DEFINED:"Klik Start setelah ROI & Line siap.",
    AppState.READY:       "Klik Start untuk memulai counting.",
    AppState.RUNNING:     "Counting… klik Stop untuk menjeda.",
    AppState.PAUSED:      "Paused. Klik Resume untuk melanjutkan.",
}

def update_ui_by_state(window: Any, state: AppState) -> None:
    cfg = _STATE_UI[state]
    # enable/disable panels
    window.src_panel     .setEnabled(cfg["src"])
    window.det_panel     .setEnabled(cfg["det"])
    window.roi_line_panel.setEnabled(cfg["roi"])
    # tombol
    window.start_btn.setText  (cfg["btn_text"])
    window.start_btn.setEnabled(cfg["btn_en"])
    # status hint
    window.status_bar.showMessage(_STATE_HINTS[state])
    window.status_bar.setState(state.name)

def notify_state_change(window: Any, state: AppState):
    """
    Aktifkan save interval hanya saat RUNNING,
    dan matikan hanya saat kita kembali ke INIT (atau RESET).
    PAUSED tidak akan mematikan save sehingga counts TIDAK di-flush.
    """
    if not hasattr(window, "async_vid") or not window.async_vid:
        return
    if state is AppState.RUNNING:
        window.async_vid.set_save_enabled(True)
    elif state is AppState.INIT:
        window.async_vid.set_save_enabled(False)
    elif state is AppState.PAUSED:
        window.async_vid._flush_counts_once()  # flush sisa counts segera
        window.async_vid.set_save_enabled(False)
