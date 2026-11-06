"""Fixed-seed subset of Seed-TTS-eval: test-en 300, test-zh 300, hard 100.

The first 100 en ids (in listed order) are the E0 reproduction set.
"""
import random
import sys
from pathlib import Path

SEED = 20261017
ROOT = Path(sys.argv[1])  # .../seedtts_testset
OUT = Path(sys.argv[2])   # data/subset_ids.txt

def ids(path):
    return [l.split("|")[0] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

sets = {
    "en": (ids(ROOT / "en" / "meta.lst"), 300),
    "zh": (ids(ROOT / "zh" / "meta.lst"), 300),
    "hard": (ids(ROOT / "zh" / "hardcase.lst"), 100),
}
lines = [f"# seed={SEED}; format: <set>\t<utt_id>; en first 100 = E0 set"]
for name, (all_ids, n) in sets.items():
    rng = random.Random(f"{SEED}-{name}")
    pick = rng.sample(sorted(all_ids), n)
    lines += [f"{name}\t{u}" for u in pick]
    print(name, len(all_ids), "->", len(pick))
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
