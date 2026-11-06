"""Tiny cross-process GPU guard for queue_gen.py / queue_eval.py (2026-10-23: IndexTTS-2 generation + Whisper
evaluation together exceed 16 GB and Windows pages GPU memory -> ~500x slowdown).
A lock is a file logs/<name>.lock holding the owner PID; stale locks (dead PID) are ignored."""
import ctypes
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _alive(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def held(name):
    f = ROOT / "logs" / f"{name}.lock"
    try:
        return _alive(int(f.read_text().strip()))
    except (FileNotFoundError, ValueError):
        return False


def wait_free(name, poll=20):
    while held(name):
        time.sleep(poll)


class Lock:
    """Exclusive: waits until no live process holds the lock, then takes it atomically (O_EXCL)."""

    def __init__(self, name, poll=20):
        self.f, self.poll = ROOT / "logs" / f"{name}.lock", poll

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = int(self.f.read_text().strip())
                except (FileNotFoundError, ValueError):
                    continue
                if not _alive(holder):
                    self.f.unlink(missing_ok=True)  # stale lock
                    continue
                time.sleep(self.poll)

    def __exit__(self, *a):
        self.f.unlink(missing_ok=True)
