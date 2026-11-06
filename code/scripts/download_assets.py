"""Download all external assets at the revisions recorded in data/MANIFEST.md (tools env: huggingface_hub, gdown).

  python scripts/download_assets.py            # everything
  python scripts/download_assets.py --only data
Checks the SHA256 of the two Google-Drive files against the manifest.
ModelScope assets (wetext TN resources, CAM++, ERes2Net-en, Paraformer) are fetched by setup scripts / on first use.
"""
import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "ckpt"
SHA = {"seedtts_testset.tar": "65B49DF7A5F4714042F9849FA216326F0D77DFE37496878B9B2240A0D2A2F4F9",
       "wavlm_large_finetune.pth": "51F07E3B94D9E0262A6A675EF5A087BE3DD09E8C62E9D886827F44F82FE7F94B"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def gdrive(fid, out):
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([sys.executable, "-m", "gdown", fid, "-O", str(out)])
    got = sha256(out)
    assert got == SHA[out.name], f"{out.name}: sha256 {got} != manifest"
    print("ok", out)


def data():
    tar = ROOT / "data/raw/seedtts/seedtts_testset.tar"
    gdrive("1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP", tar)
    if not (tar.parent / "seedtts_testset").exists():
        with tarfile.open(tar) as t:
            t.extractall(tar.parent)
    gdrive("1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP", ROOT / "data/ckpt/wavlm_large_finetune.pth")
    snapshot_download("MiniMaxAI/TTS-Multilingual-Test-Set", repo_type="dataset",
                      revision="072144006475d33f8263481ab465c00747fe2c74", local_dir=str(ROOT / "data/raw/minimax"))


def ckpts():
    snapshot_download("FunAudioLLM/CosyVoice2-0.5B", revision="eec1ae6c79877dbd9379285cf8789c9e0879293d",
                      local_dir=str(CK / "CosyVoice2-0.5B"))
    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", revision="29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
                      local_dir=str(CK / "Fun-CosyVoice3-0.5B"))
    # IndexTTS-2 (v5 E0'/E8'); its auxiliary models (w2v-BERT, semantic codec, CAM++, BigVGAN) are fetched by the
    # official code on first run into ckpt/IndexTTS-2/hf_cache
    snapshot_download("IndexTeam/IndexTTS-2", revision="740dcaff396282ffb241903d150ac011cd4b1ede",
                      local_dir=str(CK / "IndexTTS-2"))
    snapshot_download("k2-fsa/ZipVoice", revision="889dd609a42f0de9cf9aac335eae42677becb1b9",
                      allow_patterns=["zipvoice/*", "zipvoice_distill/*"], local_dir=str(CK / "ZipVoice"))
    snapshot_download("charactr/vocos-mel-24khz", local_dir=str(CK / "vocos-mel-24khz"))
    for f in ["F5TTS_v1_Base/model_1250000.safetensors", "F5TTS_v1_Base/vocab.txt",
              "F5TTS_Base/model_1200000.safetensors", "F5TTS_Base/vocab.txt"]:
        hf_hub_download("SWivid/F5-TTS", f, revision="84e5a410d9cead4de2f847e7c9369a6440bdfaca",
                        local_dir=str(CK / "F5-TTS"))
    for f in ["model_85000.pt", "model_1500.pt"]:
        hf_hub_download("yl4579/DMOSpeech2", f, revision="b2db04bf6fd5147c487149fe173576eec179f910",
                        local_dir=str(CK / "DMOSpeech2"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["data", "ckpts"])
    a = ap.parse_args()
    if a.only != "ckpts":
        data()
    if a.only != "data":
        ckpts()
