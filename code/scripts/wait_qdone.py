"""Block until every folder of a jobs module is generated (_done) and evaluated (_qdone).
python scripts/wait_qdone.py <jobs_module>
Used to keep the generation line off the GPU while speech evaluation runs (2026-10-28: TangoFlux generation + speech
evaluation exceeded 16 GB -> CUDA OOM in evaluation and a stalled generator, as with IndexTTS-2 on 10-23)."""
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
J = importlib.import_module(sys.argv[1])


def left():
    n = 0
    for name, c in J.CONFIGS.items():
        if c.get("prep"):
            continue
        for k in c["K"]:
            for s, _, _ in J.sets_for(name, k):
                n += not (J.wav_dir(name, k, s) / "_qdone").exists()
    return n


while (n := left()) > 0:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] waiting: {n} folders not evaluated", flush=True)
    time.sleep(300)
print("all evaluated", flush=True)
