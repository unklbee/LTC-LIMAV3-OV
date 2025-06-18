from threading import Event
from object_detector.video.async_video import AsyncVideo

class DummyThread:
    def __init__(self):
        self.join_called = False
    def join(self):
        self.join_called = True

class DummyCap:
    def __init__(self):
        self.released = False
    def release(self):
        self.released = True


def test_stop_flushes_counts_and_joins_threads():
    av = object.__new__(AsyncVideo)
    av._stop_requested = Event()
    av._running = Event(); av._running.set()
    av._save_enabled = Event(); av._save_enabled.set()
    av.cap = DummyCap()
    av._capture_thread = DummyThread()
    av._inference_thread = DummyThread()
    av._save_thread = DummyThread()

    called = {}
    def fake_flush(self, *a, **kw):
        called['ok'] = True
    av._flush_counts_once = fake_flush.__get__(av, AsyncVideo)

    AsyncVideo.stop(av)

    assert called.get('ok') is True
    assert av.cap.released
    assert av._capture_thread.join_called
    assert av._inference_thread.join_called
    assert av._save_thread.join_called
