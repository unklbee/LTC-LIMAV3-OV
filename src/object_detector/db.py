import sqlite3
import threading
import time
import queue
from typing import Dict
from datetime import datetime
import logging

import requests

logger = logging.getLogger(__name__)


class CountDatabase:
    """Thread‐safe SQLite helper untuk tabel `counts`, menggunakan worker thread dan opsi push via API dengan debug payload."""

    def __init__(self, db_path: str, host_id: str = None, api_url: str = None):
        # ―― buka koneksi dengan busy_timeout 3 detik
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=3.0  # tunggu sampai 3 detik bila DB locked
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout = 3000")
        self._create_table_if_not_exists()

        # ―― antrian dan worker thread untuk write
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        logger.info("DB worker thread started")

        # simpan konfigurasi
        self.host_id = host_id or "default-host"
        self.api_url = api_url or ""

    def set_host_id(self, host_id: str):
        """Update host_id runtime."""
        self.host_id = host_id

    def set_api_url(self, url: str):
        """Update API endpoint runtime."""
        self.api_url = url or ""

    def _create_table_if_not_exists(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS counts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                host_id         TEXT,
                interval_start  DATETIME,
                interval_end    DATETIME,
                bicycle         INTEGER DEFAULT 0,
                car             INTEGER DEFAULT 0,
                truck           INTEGER DEFAULT 0,
                motorbike       INTEGER DEFAULT 0,
                bus             INTEGER DEFAULT 0
            )
            """
        )
        self.conn.commit()
        logger.debug("Table 'counts' ensured")

    def insert_interval_counts(
        self,
        interval_start: datetime,
        interval_end:   datetime,
        counts:         Dict[int, int]
    ):
        """Enqueue data untuk disimpan di DB dan push API."""
        row = (
            self.host_id,
            interval_start.strftime("%Y-%m-%d %H:%M:%S"),
            interval_end.strftime("%Y-%m-%d %H:%M:%S"),
            counts.get(1, 0),  # bicycle
            counts.get(2, 0),  # car
            counts.get(7, 0),  # truck
            counts.get(3, 0),  # motorbike
            counts.get(5, 0),  # bus
        )
        self._queue.put(row)
        logger.debug("Enqueued DB row: %s", row)

    def _worker_loop(self):
        """Loop di thread terpisah: commit ke SQLite, lalu optional push via API atau debug payload."""
        insert_sql = (
            """
            INSERT INTO counts
              (host_id, interval_start, interval_end,
               bicycle, car, truck, motorbike, bus)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
        )

        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                row = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # tulis ke SQLite dengan retry
            for attempt in range(5):
                try:
                    self.conn.execute(insert_sql, row)
                    self.conn.commit()
                    logger.debug("DB insert committed: %s", row)
                    break
                except sqlite3.OperationalError as e:
                    logger.warning("DB locked, retrying (%d): %s", attempt, e)
                    time.sleep(0.1)
            else:
                logger.error("Failed to write to DB after retries: %s", row)

            # Siapkan payload untuk debug dan API
            payload = {
                "host_id":        self.host_id,
                "interval_start": row[1],
                "interval_end":   row[2],
                "counts": {
                    "bicycle":   row[3],
                    "car":       row[4],
                    "truck":     row[5],
                    "motorbike": row[6],
                    "bus":       row[7],
                }
            }

            # Tampilkan payload di console (INFO level)
            logger.info("Prepared API payload: %s", payload)

            if self.api_url:
                try:
                    resp = requests.post(self.api_url, json=payload, timeout=5)
                    logger.info("API request to %s returned [%d]: %s", self.api_url, resp.status_code, resp.text)
                except Exception as e:
                    logger.warning("API push failed: %s", e)
            else:
                logger.info("[DEBUG MODE] API URL not set, skipping POST.")

            self._queue.task_done()

    def close(self):
        """Shutdown worker thread dan tutup koneksi."""
        logger.info("Shutting down DB worker…")
        self._stop_event.set()
        self._worker.join()
        self._queue.join()
        self.conn.close()
        logger.info("Database closed")