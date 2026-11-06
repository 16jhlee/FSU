"""v9.1 §1.1 data (Stage-wise CFG reproduction, 2509.22007): COCO 2017 val, 5,000 images, first caption per image (sorted by
image id) -> data/raw/dom/coco2017val/captions.tsv (ids c17_0000..) and ref/<id>.jpg (FID reference = the same 5k images).
env_dom.  python scripts/v9_coco2017_data.py
"""
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/raw/dom/coco2017val"
DL = Path(r"C:\speechFM_store\dl\coco2017")
D.mkdir(parents=True, exist_ok=True)
(D / "ref").mkdir(exist_ok=True)
DL.mkdir(parents=True, exist_ok=True)


def fetch(url, dst):
    if dst.exists() and zipfile.is_zipfile(dst):
        return
    for k in range(40):
        # urllib stalled at 0 bytes on this machine (2026-10-27); curl resumes (-C -) and retries.
        # A curl transfer also stalled silently at 308 MB -> abort when < 1 kB/s for 60 s and resume
        subprocess.call(["curl.exe", "-L", "-s", "-C", "-", "--retry", "5", "--speed-limit", "1000", "--speed-time", "60",
                         "-o", str(dst), url])
        if zipfile.is_zipfile(dst):
            return
        print("retry", url, k, flush=True)
    raise RuntimeError(url)


fetch("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", DL / "ann.zip")
with zipfile.ZipFile(DL / "ann.zip") as z:
    ann = json.loads(z.read("annotations/captions_val2017.json"))
first = {}
for a in sorted(ann["annotations"], key=lambda a: a["id"]):
    first.setdefault(a["image_id"], a["caption"].strip())
imgs = sorted(ann["images"], key=lambda i: i["id"])
rows = [dict(id=f"c17_{j:04d}", caption=first[i["id"]], file=i["file_name"], image_id=i["id"]) for j, i in enumerate(imgs)]
pd.DataFrame(rows).to_csv(D / "captions.tsv", sep="\t", index=False)
print(len(rows), "captions", flush=True)
fetch("http://images.cocodataset.org/zips/val2017.zip", DL / "val2017.zip")
with zipfile.ZipFile(DL / "val2017.zip") as z:
    for r in rows:
        p = D / "ref" / f"{r['id']}.jpg"
        if not p.exists():
            with z.open(f"val2017/{r['file']}") as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)
print(len(list((D / "ref").glob("*.jpg"))), "reference images", flush=True)
