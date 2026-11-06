"""v8 §3.1 data: AudioCaps test (OpenSound/AudioCaps parquet shards; first 100 distinct clips: caption + reference
audio) and COCO Karpathy test (yerevann/coco-karpathy; first 100 images: first caption + image downloaded from the
COCO URL). -> data/raw/dom/audiocaps/{ref/*.wav, captions.tsv}, data/raw/dom/coco/{ref/*.jpg, captions.tsv}.
env_dom.  python scripts/v8_dom_data.py
"""
import io
from pathlib import Path

import pandas as pd
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/dom"


def audiocaps():
    d = OUT / "audiocaps"
    (d / "ref").mkdir(parents=True, exist_ok=True)
    rows, shard = [], 0
    while len(rows) < 100:
        f = hf_hub_download("OpenSound/AudioCaps", f"data/test-{shard:05d}-of-00041.parquet", repo_type="dataset")
        D = pd.read_parquet(f)
        for _, r in D.iterrows():
            if len(rows) >= 100:
                break
            a = r["audio"]
            y, sr = sf.read(io.BytesIO(a["bytes"]))
            cid = f"ac_{len(rows):03d}"
            sf.write(str(d / "ref" / f"{cid}.wav"), y, sr)
            cap = r.get("caption") if "caption" in r else r.get("text")
            rows.append(dict(id=cid, caption=str(cap).strip(), src=str(r.get("audiocap_id", r.get("youtube_id", ""))), sr=sr))
        shard += 1
    pd.DataFrame(rows).to_csv(d / "captions.tsv", sep="\t", index=False)
    print("audiocaps", len(rows), "columns", list(D.columns))


def coco():
    d = OUT / "coco"
    (d / "ref").mkdir(parents=True, exist_ok=True)
    f = hf_hub_download("yerevann/coco-karpathy", "data/test-00000-of-00001.parquet", repo_type="dataset")
    D = pd.read_parquet(f)
    rows = []
    for _, r in D.iterrows():
        if len(rows) >= 100:
            break
        url = r["url"] if "url" in r else None
        if not url:
            continue
        img = requests.get(url, timeout=60)
        if img.status_code != 200:
            continue
        cid = f"coco_{len(rows):03d}"
        (d / "ref" / f"{cid}.jpg").write_bytes(img.content)
        caps = r["sentences"]
        rows.append(dict(id=cid, caption=str(caps[0]).strip(), url=url))
    pd.DataFrame(rows).to_csv(d / "captions.tsv", sep="\t", index=False)
    print("coco", len(rows), "columns", list(D.columns))


if __name__ == "__main__":
    audiocaps()
    coco()
