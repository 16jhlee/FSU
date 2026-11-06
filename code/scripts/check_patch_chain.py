"""Verify that patches/ apply in order on fresh checkouts at the pinned commits and reproduce the working copies.
Run with the tools env python (needs git on PATH)."""
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = {"CosyVoice": ("074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
                       ["CosyVoice_sampler", "CosyVoice_dump_states", "CosyVoice_E5", "CosyVoice_E7", "CosyVoice_branch",
                        "CosyVoice_E8p7"],
                       ["cosyvoice/flow/flow_matching.py"]),
         "ZipVoice": ("2f7326fbfe999a3ad179e3f1af82a424d4a62819", ["ZipVoice_seed", "ZipVoice_E5", "ZipVoice_E7"],
                      ["zipvoice/models/modules/solver.py", "zipvoice/bin/infer_zipvoice.py"]),
         "DMOSpeech2": ("10c2255a59b96c311c317536603a18672e69ac84", ["DMOSpeech2_sampler"], ["src/infer.py"]),
         "F5-TTS": ("2f866649b779881af8bd644e76b9a6d95053da29", ["F5TTS_E8p"], ["src/f5_tts/model/cfm.py"]),
         "index-tts": ("d9e41aac89fd00b3d71497fddb287b7f24613712", ["IndexTTS_E8p", "IndexTTS_branch"],
                       ["indextts/s2mel/modules/flow_matching.py"])}


def h(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


tmp = Path(tempfile.mkdtemp())
ok = True
for repo, (sha, patches, files) in CHAIN.items():
    d = tmp / repo
    subprocess.check_call(["git", "clone", "-q", "--no-checkout", str(ROOT / "repos" / repo), str(d)])
    subprocess.check_call(["git", "-C", str(d), "checkout", "-q", sha])
    for p in patches:
        rc = subprocess.call(["git", "-C", str(d), "apply", str(ROOT / "patches" / f"{p}.diff")])
        print(f"{repo}: apply {p} rc={rc}")
        ok &= rc == 0
    for f in files:
        same = h(d / f) == h(ROOT / "repos" / repo / f)
        print(f"{repo}: {f} identical to working copy: {same}")
        ok &= same
shutil.rmtree(tmp, ignore_errors=True)
print("ALL OK" if ok else "MISMATCH")
