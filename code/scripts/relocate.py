"""Re-point path-dependent artefacts after cloning the project to a new location / machine.

1. Regenerates data/lists*/ (meta.lst + zipvoice.tsv hold absolute prompt-wav paths) from
   data/subset_ids.txt and the Seed-TTS-eval test set (deterministic).
2. Re-keys prompt-embedding caches data/cache/*.npz (keys = normalised absolute prompt paths) and
   rewrites data/cache/prompt_resynth/*/index.tsv to the new root.
Run once with the tools env after data/raw/seedtts is in place:  python scripts/relocate.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "raw" / "seedtts" / "seedtts_testset"
MARK = "/data/raw/seedtts/"


def norm(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def new_key(k):
    i = k.find(MARK)
    return k if i < 0 else norm(str(ROOT)) + k[i:]


def main():
    assert SEED.exists(), f"missing {SEED} (run scripts/download_assets.py first)"
    mk = [sys.executable, str(ROOT / "scripts" / "make_lists.py")]
    sub = str(ROOT / "data" / "subset_ids.txt")
    for name, n in [("en", "all"), ("zh", "all"), ("hard", "all"), ("en", "100"), ("en", "20"), ("en", "3")]:
        subprocess.check_call(mk + [str(SEED), sub, name, n, str(ROOT / "data" / "lists")])
    subprocess.check_call(mk + [str(SEED), "-", "en", "all", str(ROOT / "data" / "lists_full")])
    for f in (ROOT / "data" / "cache").glob("*.npz"):
        d = dict(np.load(f))
        nd = {new_key(k): v for k, v in d.items()}
        if list(nd) != list(d):
            np.savez(f, **nd)
            print("re-keyed", f.name, len(nd))
    for idx in (ROOT / "data" / "cache" / "prompt_resynth").glob("*/index.tsv"):
        lines = []
        for l in idx.read_text(encoding="utf-8").splitlines():
            p, w = l.split("\t")
            q = p.replace("\\", "/")
            i = q.lower().find(MARK)
            lines.append((str(ROOT) + q[i:] if i >= 0 else p) + "\t" + w)
        idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("rewrote", idx)
    print("done; root =", ROOT)


if __name__ == "__main__":
    main()
