"""v9 §1.2 data: AudioCaps test, all clips (OpenSound/AudioCaps, 41 parquet shards): one caption per distinct
audiocap clip (first occurrence) + reference audio -> data/raw/dom/audiocaps_full/{captions.tsv, ref/<id>.wav}.
env_dom.  python scripts/v9_audiocaps_data.py
"""
import io
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/raw/dom/audiocaps_full"
(D / "ref").mkdir(parents=True, exist_ok=True)
def fetch(shard):
    for k in range(10):  # flaky connection: hf_hub_download resumes the partial file on retry
        try:
            return hf_hub_download("OpenSound/AudioCaps", f"data/test-{shard:05d}-of-00041.parquet", repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print("retry", shard, k, type(e).__name__, flush=True)
            time.sleep(10)
    raise RuntimeError(f"shard {shard} failed")


rows, seen = [], set()
for shard in range(41):
    f = fetch(shard)
    for _, r in pd.read_parquet(f).iterrows():
        key = r["youtube_id"]
        if key in seen:
            continue
        seen.add(key)
        cid = f"acf_{len(rows):04d}"
        y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        sf.write(str(D / "ref" / f"{cid}.wav"), y, sr)
        rows.append(dict(id=cid, caption=str(r["caption"]).strip(), youtube_id=key, audiocap_id=r["audiocap_id"], sr=sr))
    print(shard, len(rows), flush=True)
pd.DataFrame(rows).to_csv(D / "captions.tsv", sep="\t", index=False)
print("clips", len(rows))
