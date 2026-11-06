"""Download the public checkpoints and test data at pinned revisions (environment tools: huggingface_hub, gdown).
    python -m fsu.assets            # everything
    python -m fsu.assets --only data
The SHA-256 of the two Google-Drive files is checked. ModelScope assets (wetext text normalisation, CAM++, ERes2Net,
Paraformer, SenseVoice) and the auxiliary IndexTTS-2 models are fetched by their libraries on first use.
"""
import argparse
import hashlib
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from fsu.common import CKPT, DATA, REPOS

RAW = DATA / "raw"
SHA256 = {"seedtts_testset.tar": "65B49DF7A5F4714042F9849FA216326F0D77DFE37496878B9B2240A0D2A2F4F9",
          "wavlm_large_finetune.pth": "51F07E3B94D9E0262A6A675EF5A087BE3DD09E8C62E9D886827F44F82FE7F94B"}
OPENSLR = {"librispeech": "https://www.openslr.org/resources/12/test-clean.tar.gz",  # LibriSpeech test-clean (CC-BY-4.0)
           "musan": "https://www.openslr.org/resources/17/musan.tar.gz"}  # MUSAN noise (prompt robustness)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def gdrive(file_id, out):
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([sys.executable, "-m", "gdown", file_id, "-O", str(out)])
    got = sha256(out)
    assert got == SHA256[out.name], f"{out.name}: sha256 {got} differs from the pinned value"


def untar(tgz, dest, done):
    if not (dest / done).exists():
        with tarfile.open(tgz) as t:
            t.extractall(dest)


def data():
    from huggingface_hub import snapshot_download
    tar = RAW / "seedtts" / "seedtts_testset.tar"  # Seed-TTS-eval (github.com/BytedanceSpeech/seed-tts-eval)
    gdrive("1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP", tar)
    untar(tar, tar.parent, "seedtts_testset")
    gdrive("1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP", DATA / "ckpt" / "wavlm_large_finetune.pth")  # Seed-TTS-eval SIM model
    snapshot_download("MiniMaxAI/TTS-Multilingual-Test-Set", repo_type="dataset",
                      revision="072144006475d33f8263481ab465c00747fe2c74", local_dir=str(RAW / "minimax"))
    for name, url in OPENSLR.items():
        tgz = RAW / "downloads" / url.rsplit("/", 1)[1]
        if not tgz.exists():
            tgz.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, tgz)
        untar(tgz, RAW / name, "LibriSpeech" if name == "librispeech" else "musan")


def checkpoints():
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download("FunAudioLLM/CosyVoice2-0.5B", revision="eec1ae6c79877dbd9379285cf8789c9e0879293d",
                      local_dir=str(CKPT / "CosyVoice2-0.5B"))
    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", revision="29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
                      local_dir=str(CKPT / "Fun-CosyVoice3-0.5B"))
    snapshot_download("IndexTeam/IndexTTS-2", revision="740dcaff396282ffb241903d150ac011cd4b1ede",
                      local_dir=str(CKPT / "IndexTTS-2"))
    snapshot_download("zai-org/GLM-TTS", revision="b3feb691fa4cf590eec4418f6275f95220e58844",
                      local_dir=str(REPOS / "GLM-TTS" / "ckpt"))  # the official pipeline reads ckpt/ inside its repo
    snapshot_download("k2-fsa/ZipVoice", revision="889dd609a42f0de9cf9aac335eae42677becb1b9",
                      allow_patterns=["zipvoice/*", "zipvoice_distill/*"], local_dir=str(CKPT / "ZipVoice"))
    snapshot_download("charactr/vocos-mel-24khz", local_dir=str(CKPT / "vocos-mel-24khz"))
    for f in ["F5TTS_v1_Base/model_1250000.safetensors", "F5TTS_v1_Base/vocab.txt",
              "F5TTS_Base/model_1200000.safetensors", "F5TTS_Base/vocab.txt"]:
        hf_hub_download("SWivid/F5-TTS", f, revision="84e5a410d9cead4de2f847e7c9369a6440bdfaca", local_dir=str(CKPT / "F5-TTS"))
    for f in ["model_85000.pt", "model_1500.pt"]:
        hf_hub_download("yl4579/DMOSpeech2", f, revision="b2db04bf6fd5147c487149fe173576eec179f910",
                        local_dir=str(CKPT / "DMOSpeech2"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["data", "checkpoints"])
    a = ap.parse_args(argv)
    if a.only != "checkpoints":
        data()
    if a.only != "data":
        checkpoints()


if __name__ == "__main__":
    main()
