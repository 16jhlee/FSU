"""E1b step 1: add ERes2Net-en embeddings/SIM to every evaluated E1 (and E0) output dir.
eval_metrics.py resumes and only computes the missing `sim_eres2net` columns."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS, ROOT, jobs  # noqa: E402

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
targets = []
for j in jobs():
    if j["out"]:
        for s, meta, lang in j["sets"]:
            targets.append((f"{j['name']}_{s}", ROOT / j["out"] / s, meta, lang))
log = ROOT / "logs" / "E1b"
log.mkdir(parents=True, exist_ok=True)
for name, d, meta, lang in targets:
    if not (d / "_evaldone").exists():
        continue
    csv = d / "_metrics" / "metrics.csv"
    if "sim_eres2net" in csv.read_text(encoding="utf-8").splitlines()[0]:
        continue
    with open(log / f"emb_{name}.log", "a", encoding="utf-8") as f:
        rc = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", meta + ".meta.lst",
                              "--wav_dir", str(d), "--lang", lang, "--metrics", "eres2net"],
                             cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(name, rc, flush=True)
