"""v9 §1.1 data: COCO Karpathy test (yerevann/coco-karpathy) -> data/raw/dom/coco1k/captions.tsv (all 5,000 test images,
first caption each, ids coco_0000..) and reference images ref/<id>.jpg for the first N (default 1000; FID reference).
env_dom.  python scripts/v9_coco_data.py [N]
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/raw/dom/coco1k"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
(D / "ref").mkdir(parents=True, exist_ok=True)
T = pd.read_parquet(hf_hub_download("yerevann/coco-karpathy", "data/test-00000-of-00001.parquet", repo_type="dataset"))
rows = [dict(id=f"coco_{i:04d}", caption=str(r.sentences[0]).strip(), url=r.url, cocoid=r.cocoid) for i, (_, r) in enumerate(T.iterrows())]
pd.DataFrame(rows).to_csv(D / "captions.tsv", sep="\t", index=False)


def get(r):
    p = D / "ref" / f"{r['id']}.jpg"
    if p.exists():
        return
    for _ in range(3):
        try:
            x = requests.get(r["url"], timeout=60)
            if x.status_code == 200:
                p.write_bytes(x.content)
                return
        except requests.RequestException:
            pass


with ThreadPoolExecutor(16) as ex:
    list(ex.map(get, rows[:N]))
print(len(rows), "captions;", len(list((D / "ref").glob("*.jpg"))), "reference images")
