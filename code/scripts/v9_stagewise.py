"""v9.1 §1.1 Stage-wise CFG reproduction (2509.22007 Table 2): SD3.5-medium, COCO 2017 val 5k, 1024 px, omega = 9,
NFE 5 / 10, rules: vanilla (baseline) / interval-CFG (sqt24/tvcfg interval, left 0.15, right 0.78, normalised) /
TV-CFG (tvcfg symmetric) / R (ours) / R + TV-CFG. Same sampler code as v9_image.py (FlowMatchEuler, official shift).
python scripts/v9_stagewise.py embed            # text embeddings for the 5k captions
python scripts/v9_stagewise.py gen [n_cap=5000] # FID set (seed 0)
python scripts/v9_stagewise.py div [n_cap=100]  # diversity: 16 seeds x n_cap prompts (paper: 1,000 prompts; reduced, reported)
Outputs out/E9/stagewise/sd35/<cond>/s<k>/<id>.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v9_image as V  # noqa: E402

V.CAPS = V.ROOT / "data/raw/dom/coco2017val/captions.tsv"
V.EMB = V.ROOT / "cache" / "e9_c17"
V.OUT = V.ROOT / "out" / "E9" / "stagewise"
GRID = [(9.0, n, r) for n in (5, 10) for r in ("baseline", "intv", "tvs", "R", "Rtv")]
_orig = V.grid
V.grid = lambda name: GRID if name == "stagewise" else _orig(name)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "embed":
        V.embed("sd35", 5000)
    elif cmd == "gen":
        V.gen("sd35", "stagewise", [0], int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
    elif cmd == "div":
        V.gen("sd35", "stagewise", list(range(1, 16)), int(sys.argv[2]) if len(sys.argv) > 2 else 100)
