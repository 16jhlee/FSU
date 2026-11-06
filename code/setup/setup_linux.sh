#!/usr/bin/env bash
# speechFM — fresh-machine setup (Linux, NVIDIA GPU, conda/miniconda on PATH). Run from the repo root:
#   bash setup/setup_linux.sh
# Same recipe as setup_windows.ps1. The original results were produced on Windows; exact versions in env/lock_*.txt.
# On Linux, CosyVoice's requirements also pull deepspeed / tensorrt / onnxruntime-gpu (Linux-only lines).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
ENVS="$(conda info --base)/envs"
py() { echo "$ENVS/$1/bin/python"; }
newenv() { [ -x "$(py "$1")" ] || conda create -y -n "$1" -c conda-forge --override-channels "python=$2" pip -q; }

newenv tools 3.11
conda install -y -n tools -c conda-forge --override-channels git git-lfs -q
"$(py tools)" -m pip install -q gdown huggingface_hub numpy soundfile
export PATH="$ENVS/tools/bin:$PATH"; git lfs install

mkdir -p repos
clone() { # name url sha patch
  [ -d "repos/$1" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --recursive -q "$2" "repos/$1"
  git -C "repos/$1" checkout -q "$3"; git -C "repos/$1" submodule update --init --recursive -q
  if [ -n "${4:-}" ]; then
    if git -C "repos/$1" apply --check -R "$ROOT/patches/$4" 2>/dev/null; then echo "$1 already patched";
    else git -C "repos/$1" apply "$ROOT/patches/$4"; echo "patched $1"; fi
  fi
}
clone CosyVoice https://github.com/FunAudioLLM/CosyVoice 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc CosyVoice_sampler.diff
clone ZipVoice https://github.com/k2-fsa/ZipVoice 2f7326fbfe999a3ad179e3f1af82a424d4a62819 ZipVoice_seed.diff
clone F5-TTS https://github.com/SWivid/F5-TTS 2f866649b779881af8bd644e76b9a6d95053da29 F5TTS_E8p.diff
clone index-tts https://github.com/index-tts/index-tts d9e41aac89fd00b3d71497fddb287b7f24613712 IndexTTS_E8p.diff
clone DMOSpeech2 https://github.com/yl4579/DMOSpeech2 10c2255a59b96c311c317536603a18672e69ac84 DMOSpeech2_sampler.diff
clone FlashTTS https://github.com/ASLP-lab/FlashTTS d1d9dc0cff1dcb06a4f38d635f33b8f6cfeb3b4d
clone seed-tts-eval https://github.com/BytedanceSpeech/seed-tts-eval 752f4297f090c46bb1a55a1f7439e5944ddefe8d
# incremental CosyVoice patches (on top of CosyVoice_sampler.diff, in order)
for rp in CosyVoice:CosyVoice_dump_states.diff CosyVoice:CosyVoice_E5.diff CosyVoice:CosyVoice_E7.diff \
          CosyVoice:CosyVoice_branch.diff CosyVoice:CosyVoice_E8p7.diff ZipVoice:ZipVoice_E5.diff ZipVoice:ZipVoice_E7.diff index-tts:IndexTTS_branch.diff; do
  r="${rp%%:*}"; p="${rp#*:}"
  git -C "repos/$r" apply --check -R "$ROOT/patches/$p" 2>/dev/null || { git -C "repos/$r" apply "$ROOT/patches/$p"; echo "applied $p"; }
done

CU="--index-url https://download.pytorch.org/whl/cu124"
newenv env_cosy 3.10
"$(py env_cosy)" -m pip install -q "setuptools<81" wheel
"$(py env_cosy)" -m pip install -q --no-build-isolation openai-whisper==20231117
"$(py env_cosy)" -m pip install -q -r repos/CosyVoice/requirements.txt

newenv env_zip 3.10
"$(py env_zip)" -m pip install -q torch==2.5.1 torchaudio==2.5.1 $CU
"$(py env_zip)" -m pip install -q -r repos/ZipVoice/requirements.txt urllib3 requests

newenv env_f5dmo 3.10
"$(py env_f5dmo)" -m pip install -q torch==2.5.1 torchaudio==2.5.1 $CU
"$(py env_f5dmo)" -m pip install -q "setuptools<81"
"$(py env_f5dmo)" -m pip install -q -e repos/F5-TTS "torch==2.5.1" "torchaudio==2.5.1"
"$(py env_f5dmo)" -m pip install -q -r repos/DMOSpeech2/requirements.txt "torch==2.5.1" "torchaudio==2.5.1" ipython

newenv env_index 3.11   # IndexTTS-2 (v5); versions in env/lock_env_index.txt (Windows freeze; drop win-only pins if any)
"$(py env_index)" -m pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$(py env_index)" -m pip install -q -e repos/index-tts

newenv env_eval 3.10
"$(py env_eval)" -m pip install -q torch==2.5.1 torchaudio==2.5.1 $CU
"$(py env_eval)" -m pip install -q "setuptools<81" "transformers==4.46.3" "jiwer==3.0.5" funasr modelscope zhon zhconv \
  speechbrain==1.0.3 librosa soundfile scipy pandas tqdm s3prl "numpy<2" "huggingface_hub<1.0" "datasets<4" sentencepiece \
  accelerate addict simplejson sortedcontainers oss2 pillow pyworld matplotlib "torch==2.5.1" "torchaudio==2.5.1"

"$(py tools)" scripts/download_assets.py
"$(py env_cosy)" -c "from modelscope import snapshot_download; print(snapshot_download('pengzhendong/wetext'))"
"$(py tools)" scripts/relocate.py
echo "setup complete. Next: see README.md"
