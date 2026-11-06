"""Build per-repo test lists from Seed-TTS-eval meta.lst for a subset.

Usage: make_lists.py <seedtts_root> <subset_ids.txt> <set:en|zh|hard> <n|all> <out_dir>
Writes:
  <out_dir>/<set>_<n>.meta.lst   (seed-tts-eval / F5 format, absolute prompt paths)
  <out_dir>/<set>_<n>.zipvoice.tsv (name\tprompt_text\tprompt_wav\ttext)
"""
import sys
from pathlib import Path

root, subset, name, n, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], Path(sys.argv[5])
meta = root / ("zh/hardcase.lst" if name == "hard" else f"{name}/meta.lst")
base = meta.parent
rows = {}
for l in meta.read_text(encoding="utf-8").splitlines():
    if l.strip():
        f = l.split("|")
        rows[f[0]] = f
if str(subset) == "-":  # whole test set in meta order
    ids = list(rows)
else:
    ids = [l.split("\t")[1] for l in subset.read_text(encoding="utf-8").splitlines()
           if l and not l.startswith("#") and l.split("\t")[0] == name]
if n != "all":
    ids = ids[: int(n)]
out.mkdir(parents=True, exist_ok=True)
tag = f"{name}_{n}"
with open(out / f"{tag}.meta.lst", "w", encoding="utf-8", newline="\n") as fm, \
     open(out / f"{tag}.zipvoice.tsv", "w", encoding="utf-8", newline="\n") as fz:
    for u in ids:
        f = rows[u]
        pw = str((base / f[2]).resolve())
        fm.write("|".join([u, f[1], pw, f[3]] + f[4:]) + "\n")
        fz.write("\t".join([u, f[1], pw, f[3]]) + "\n")
print(tag, len(ids))
