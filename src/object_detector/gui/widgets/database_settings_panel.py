# src/object_detector/gui/widgets/database_settings_panel.py

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QLineEdit, QPushButton, QFileDialog, QSpinBox
)
from ...config import Config

class DatabaseSettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)

        # — Path ke file DB
        self.db_path_edit = QLineEdit(str(Config.DB_PATH))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_db)
        layout.addRow("Database File:", self.db_path_edit)
        layout.addRow("", browse_btn)

        # — Interval simpan (menit)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 24 * 60)                            # 1…1440 menit
        self.interval_spin.setValue(Config.COUNT_SAVE_INTERVAL_SEC // 60)   # tampilkan menit
        layout.addRow("Interval (menit):", self.interval_spin)

        # — Host ID
        self.hostid_edit = QLineEdit(getattr(Config, "HOST_ID", ""))
        layout.addRow("Host ID:", self.hostid_edit)

        # — API Endpoint URL
        self.api_url_edit = QLineEdit(getattr(Config, "API_URL", ""))
        layout.addRow("API URL:", self.api_url_edit)

        # — Tombol Apply
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self._apply_settings)
        layout.addRow("", self.apply_btn)

    def _browse_db(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Database File",
            str(Config.DB_PATH),
            "SQLite DB (*.db)"
        )
        if path:
            self.db_path_edit.setText(path)

    def _apply_settings(self):
        # simpan path
        Config.DB_PATH = Path(self.db_path_edit.text())
        # konversi menit → detik
        Config.COUNT_SAVE_INTERVAL_SEC = self.interval_spin.value() * 60
        # simpan host_id
        Config.HOST_ID = self.hostid_edit.text().strip()
        # simpan API URL
        Config.API_URL = self.api_url_edit.text().strip()

        # Umpan balik
        self.apply_btn.setText("Applied ✓")
