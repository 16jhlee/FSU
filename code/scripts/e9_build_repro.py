"""E9 (v7 §3.5) reproduction package -> repro/
- minimal R (ours) patches generated on fresh checkouts of the pinned commits (CosyVoice: CosyVoice2/3; index-tts:
  IndexTTS-2): one attribute `cfg_t_min` (default 0.0 = official behaviour); w = 0 for steps starting at t < cfg_t_min.
- copies: full experiment patch chain, pins / lock files, data manifest + sample ids, all result CSVs, figure/table
  script list, samples index.  Run with the tools env python (needs git on PATH).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "repro"
PINS = {"CosyVoice": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc", "index-tts": "d9e41aac89fd00b3d71497fddb287b7f24613712"}
EDITS = {
    "CosyVoice": ("cosyvoice/flow/flow_matching.py",
                  "            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)\n",
                  "            # R: no CFG for steps that start near noise (t < cfg_t_min; t=0 is noise). 0.0 = official.\n"
                  "            w = 0.0 if float(t) < getattr(self, 'cfg_t_min', 0.0) else self.inference_cfg_rate\n"
                  "            dphi_dt = ((1.0 + w) * dphi_dt - w * cfg_dphi_dt)\n"),
    "index-tts": ("indextts/s2mel/modules/flow_matching.py",
                  "            if inference_cfg_rate > 0:\n",
                  "            # R: no CFG for steps that start near noise (t < cfg_t_min; t=0 is noise). 0.0 = official.\n"
                  "            w = 0.0 if float(t) < getattr(self, 'cfg_t_min', 0.0) else inference_cfg_rate\n"
                  "            if w > 0:\n"),
}
EDITS2 = {"index-tts": ("                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt\n",
                        "                dphi_dt = (1.0 + w) * dphi_dt - w * cfg_dphi_dt\n")}


def git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8").stdout


def minimal_patches():
    tmp = Path(tempfile.mkdtemp())
    for repo, sha in PINS.items():
        d = tmp / repo
        git("clone", "-q", "--no-checkout", str(ROOT / "repos" / repo), str(d))
        git("checkout", "-q", sha, cwd=d)
        f, old, new = EDITS[repo]
        p = d / f
        txt = p.read_text(encoding="utf-8")
        assert txt.count(old) == 1, (repo, "anchor not unique")
        txt = txt.replace(old, new)
        if repo in EDITS2:
            o2, n2 = EDITS2[repo]
            assert txt.count(o2) == 1
            txt = txt.replace(o2, n2)
        p.write_text(txt, encoding="utf-8", newline="\n")
        name = {"CosyVoice": "CosyVoice_R_minimal.diff", "index-tts": "IndexTTS_R_minimal.diff"}[repo]
        (R / "patches" / name).write_text(git("diff", cwd=d), encoding="utf-8", newline="\n")
        # verify on a second fresh checkout
        d2 = tmp / (repo + "_check")
        git("clone", "-q", "--no-checkout", str(ROOT / "repos" / repo), str(d2))
        git("checkout", "-q", sha, cwd=d2)
        git("apply", "--check", str(R / "patches" / name), cwd=d2)
        print("minimal patch ok:", name)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    if R.exists():
        for sub in ["patches", "results", "data", "env"]:
            shutil.rmtree(R / sub, ignore_errors=True)
    for sub in ["patches", "results", "data", "env"]:
        (R / sub).mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "patches").glob("*.diff"):
        shutil.copy2(p, R / "patches" / p.name)
    minimal_patches()
    for p in (ROOT / "results").glob("*.csv"):
        shutil.copy2(p, R / "results" / p.name)
    for f in ["MANIFEST.md", "sample_ids.txt", "subset_ids.txt"]:
        shutil.copy2(ROOT / "data" / f, R / "data" / f)
    for p in list((ROOT / "env").glob("lock_*.txt")) + [ROOT / "env" / "pins.txt"]:
        shutil.copy2(p, R / "env" / p.name)
    idx = ["# Listening samples index", "", "Built by `scripts/pick_samples.py <exp>` (copies of generated wavs; wavs are not in git).",
           "Open `index.html` in each folder.", "", "| folder | content |", "|---|---|"]
    desc = {"E1": "NFE sweep (all models)", "E7": "R (ours) vs baseline, CosyVoice2 / ZipVoice", "E8p": "E8' generalisation, branches, solvers, languages",
            "E8p8": "operating points: baseline / R / R′ / mid+R / teacher (3 models)", "E8p7": "external baselines",
            "CMOS": "blind CMOS test (240 trials; key in _key/, do not distribute)"}
    for d in sorted((ROOT / "samples").iterdir()):
        if d.is_dir():
            idx.append(f"| `samples/{d.name}/` | {desc.get(d.name, '')} |")
    (R / "samples_index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print("repro built:", sum(1 for _ in R.rglob("*") if _.is_file()), "files")


if __name__ == "__main__":
    main()
