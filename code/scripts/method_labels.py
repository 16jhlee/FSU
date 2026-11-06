"""v6 §0: map every existing config / tag to the paper labels -> results/method_labels.csv.
Labels (v6.1 §A): baseline, teacher, R (ours), mid+R (ours), R′ (Sel-CFG input_text) [= Selective-CFG 2509.19668
input_text rule, not ours], plus analysis-only conditions.
python scripts/method_labels.py  (any env with pandas)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as J  # noqa: E402

ROOT = J.ROOT
MODEL = {"cv2": "CosyVoice2", "cv3": "CosyVoice3", "idx": "IndexTTS-2", "f5": "F5-TTS", "zip": "ZipVoice"}
TEACHER_NFE = {"cv2": 10, "cv3": 10, "idx": 25, "f5": 32, "zip": 16}
rows = []


def add(cfg, model, method, nfe, calls, note=""):
    rows.append(dict(config=cfg, model=MODEL[model], method=method, nfe=nfe, calls=calls, note=note))


# E8' configs
for name, c in list(J.CONFIGS.items()) + [(a, {}) for a in J.ALIAS]:
    if name.startswith("prep"):
        continue
    parts = name.split("_")
    m = parts[1]
    r = name.endswith("_R")
    note = "MiniMax ko/ja/de" if name.startswith("e85") else ""
    if name.startswith("e82"):
        br, n = parts[2], int(parts[3][1:])
        if br == "all":
            meth = "R (ours)" if r else ("teacher" if n == TEACHER_NFE[m] else "baseline")
        elif br == "speaker":
            meth = "R′+R" if r else "R′ (Sel-CFG input_text)"
        else:
            meth = "token-null+R (analysis)" if r else "token-null (analysis)"
        add(name, m, meth, n, None, f"branch={br}")
    elif name.startswith(("e81", "e85")):
        n = int(parts[2][1:])
        add(name, m, "R (ours)" if r else ("teacher" if n == TEACHER_NFE[m] else "baseline"), n, None, note)
    elif name.startswith("e83"):
        add(name, m, "R (ours)" if r else "baseline", 2, 4 if not r else 3, f"schedule [0,{parts[2][1:]},1]")
    elif name.startswith("e84") and parts[2] in ("midpoint", "heun"):
        st = int(parts[3][1:])
        if parts[2] == "midpoint" and st == 1:
            meth = "mid+R (ours)" if r else "midpoint baseline"
        else:
            meth = f"{parts[2]}+R" if r else f"{parts[2]} baseline"
        add(name, m, meth, 2 * st, None, f"solver={parts[2]}, {st} step(s)")
    elif name.startswith("e84"):
        add(name, m, "R (ours)" if r else "baseline", int(parts[3][1:]), None, f"w={parts[2][1:]}")

# E7 configs (mask notation: 0 = CFG off for that step); R (ours) = interval [0.1,1]
E7_R = {("cv2", 2): "cv2_n2_m01", ("cv2", 3): "cv2_n3_m011", ("cv2", 4): "cv2_n4_m0011", ("cv2", 10): "cv2_n10_m0001111111",
        ("zip", 2): "zip_n2_m01", ("zip", 4): "zip_n4_m0111", ("zip", 8): "zip_n8_m00111111", ("zip", 16): "zip_n16_m0001111111111111"}
for (m, n), cfg in E7_R.items():
    add(f"e7:{cfg}", m, "R (ours)", n, None, "E7 tag cfg_interval_0.1_1")
    add(f"e7:{m}_default_n{n}", m, "teacher" if n == TEACHER_NFE[m] else "baseline", n, None, "E7")
add("e5:cfg_interval_0.1_1", "cv2", "R (ours)", None, None, "E5 tag (A2-c first-step w=0 at NFE 2)")

df = pd.DataFrame(rows).drop_duplicates("config")
df.to_csv(ROOT / "results" / "method_labels.csv", index=False, encoding="utf-8")
print(df.method.value_counts().to_string())
