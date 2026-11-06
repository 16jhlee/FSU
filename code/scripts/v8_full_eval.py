"""v8 §1.2: full metrics (WER/CER + UTMOS; SIM already present) for seeds 1 and 2 of the Table-1 main rows, so that
Table 1 can report seed mean ± seed SD. Resumable (eval_metrics.py adds only missing columns). Uses gpu_lock like
queue_eval.py.  python scripts/v8_full_eval.py   (tools env)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8v8_jobs as J  # noqa: E402
import gpu_lock  # noqa: E402
from e1_jobs import ENVS  # noqa: E402

ROOT = J.ROOT
ROWS = {  # model -> configs (seed 0 lives in the same name via aliases)
    "CosyVoice2": ["e82_cv2_all_n2_def", "e82_cv2_all_n2_R", "A_cv2_midR", "e82_cv2_all_n10_def", "e82_cv2_all_n3_def",
                   "e82_cv2_all_n3_R", "s_cv2_midR1"],
    "CosyVoice3": ["e81_cv3_n2_def", "e81_cv3_n2_R", "A_cv3_midR", "e81_cv3_n10_def", "s_cv3_n3_def", "s_cv3_n3_R",
                   "s_cv3_midR1"],
    "IndexTTS-2": ["e81_idx_n2_def", "e81_idx_n2_R", "A_idx_midR", "e81_idx_n25_def", "s_idx_n3_def", "s_idx_n3_R"],
}
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
LOG = ROOT / "logs" / "v8_full_eval"
LOG.mkdir(parents=True, exist_ok=True)


def main():
    for model, cfgs in ROWS.items():
        for cfg in cfgs:
            for k in (1, 2):
                for s, lang in [("en", "en"), ("zh", "zh")]:
                    d = J.wav_dir(cfg, k, s)
                    f = d / "_metrics" / "metrics.csv"
                    while not (d / "_done").exists() or not f.exists():
                        print(f"waiting for {d}", flush=True)
                        time.sleep(120)
                    head = f.read_text(encoding="utf-8").splitlines()[0]
                    if "wer" in head.split(",") and "utmos" in head.split(","):
                        continue
                    meta = "data/lists/en_all" if s == "en" else "data/lists/zh_all"
                    gpu_lock.wait_free("heavy_gen")
                    with open(LOG / f"{cfg}_k{k}_{s}.log", "a", encoding="utf-8") as lf, gpu_lock.Lock("eval_job"):
                        rc = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", meta + ".meta.lst",
                                              "--wav_dir", str(d), "--lang", lang, "--metrics", "wer,utmos"],
                                             cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
                    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {cfg} k{k} {s} rc={rc}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
