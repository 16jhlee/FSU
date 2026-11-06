"""Shared helpers: project paths, deterministic seeds, test-list I/O, conda environments and a cross-process GPU lock."""
import ctypes
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root: fsu/, configs/, repos/, ckpt/, data/, cache/, out/
REPOS = ROOT / "repos"
CKPT = ROOT / "ckpt"
DATA = ROOT / "data"
CACHE = ROOT / "cache"
OUT = ROOT / "out"
LOGS = ROOT / "logs"


def seed_of(key, mod=2 ** 31 - 1):
    """Deterministic 31-bit seed from a string key, e.g. seed_of(f"{utt}|noise|{k}") for the k-th noise draw of `utt`."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16) % mod


def noise_index(noise):
    """'utt:K' -> 'K' (K-th per-utterance noise draw); 'utt' -> '0'."""
    return noise.split(":")[1] if ":" in noise else "0"


def read_list(path):
    """Test list in the Seed-TTS-eval format: one `utt|prompt_text|prompt_wav|text` line per utterance."""
    return [tuple(line.split("|")[:4]) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_list(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("|".join(r) + "\n" for r in rows), encoding="utf-8")


def resolve(p):
    """Project-relative path -> absolute path (absolute paths are returned unchanged)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def env_python(name):
    """Python interpreter of conda environment `name`. The environments directory is taken from FSU_CONDA_ENVS, or
    derived from the interpreter running this code (expected to live in <envs>/<some env>/)."""
    envs = os.environ.get("FSU_CONDA_ENVS")
    if envs is None:
        exe = Path(sys.executable).resolve()
        envs = exe.parents[1] if os.name == "nt" else exe.parents[2]
    d = Path(envs) / name
    if os.name != "nt":
        return str(d / "bin" / "python")
    return str(d / "python.exe" if (d / "python.exe").exists() else d / "Scripts" / "python.exe")  # conda env or venv


def subprocess_env(**extra):
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1", **extra)


def _alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


class GpuLock:
    """File lock logs/<name>.lock holding the owner PID; stale locks (dead PID) are ignored.

    A 16 GB GPU cannot hold an IndexTTS-2 / GLM-TTS generation job and an ASR evaluation job at the same time, so
    the generation queue takes 'heavy_gen' for those decoders and the evaluation queue takes 'eval_job'; each waits for
    the other with wait_free()."""

    def __init__(self, name, poll=20):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.path, self.poll = LOGS / f"{name}.lock", poll

    @staticmethod
    def held(name):
        try:
            return _alive(int((LOGS / f"{name}.lock").read_text().strip()))
        except (FileNotFoundError, ValueError):
            return False

    @classmethod
    def wait_free(cls, name, poll=20):
        while cls.held(name):
            time.sleep(poll)

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = int(self.path.read_text().strip())
                except (FileNotFoundError, ValueError):
                    continue
                if not _alive(holder):
                    self.path.unlink(missing_ok=True)
                    continue
                time.sleep(self.poll)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
