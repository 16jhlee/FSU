"""Parallel, resumable HTTP range download (used for MUSAN, v8 §2.3).
python dl_parallel.py <out_file> <n_parts> <url1> [url2 ...]   (parts are spread over the mirror URLs; each part
retries/resumes on failure; parts are concatenated at the end)."""
import sys
import threading
import time
from pathlib import Path

import requests

out, n, urls = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
size = int(requests.head(urls[0], allow_redirects=True, timeout=30).headers["Content-Length"])
step = -(-size // n)
parts = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(n)]


def worker(i, a, b):
    p = out.with_suffix(out.suffix + f".part{i}")
    url = urls[i % len(urls)]
    while True:
        have = p.stat().st_size if p.exists() else 0
        if a + have > b:
            return
        try:
            with requests.get(url, headers={"Range": f"bytes={a + have}-{b}"}, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(p, "ab") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"part {i} retry after {type(e).__name__}", flush=True)
            url = urls[(urls.index(url) + 1) % len(urls)]
            time.sleep(5)


ts = [threading.Thread(target=worker, args=pt) for pt in parts]
for t in ts:
    t.start()
while any(t.is_alive() for t in ts):
    got = sum(out.with_suffix(out.suffix + f".part{i}").stat().st_size for i, _, _ in parts
              if out.with_suffix(out.suffix + f".part{i}").exists())
    print(f"{time.strftime('%H:%M:%S')} {got / 1e6:.0f} / {size / 1e6:.0f} MB", flush=True)
    time.sleep(60)
with open(out, "wb") as f:
    for i, _, _ in parts:
        p = out.with_suffix(out.suffix + f".part{i}")
        f.write(p.read_bytes())
        p.unlink()
print("done", out.stat().st_size == size, flush=True)
