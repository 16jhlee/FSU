"""Run one command while holding the evaluation GPU lock (never overlaps queue_eval jobs or IndexTTS-2 generation).
python scripts/with_eval_lock.py <python> <args...>   (tools env)"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402

gpu_lock.wait_free("heavy_gen")
with gpu_lock.Lock("eval_job"):
    sys.exit(subprocess.call(sys.argv[1:], cwd=Path(__file__).resolve().parents[1]))
