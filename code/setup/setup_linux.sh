#!/usr/bin/env bash
# Fresh-machine setup on Linux (NVIDIA GPU, conda on PATH). Same recipe as setup_windows.ps1; the reported results
# were produced on Windows, exact versions in env/lock_<env>.txt. Run from the repository root:  bash setup/setup_linux.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
eval "$(conda shell.bash hook)"
ENVS="$(conda info --base)/envs"
py() { echo "$ENVS/$1/bin/python"; }
newenv() { [ -x "$(py "$1")" ] || conda create -y -q -n "$1" -c conda-forge --override-channels "python=$2" pip; }
pins() { grep -E '^[A-Za-z0-9_.\-\[\]]+==' "env/lock_$1.txt" > "/tmp/constraints_$1.txt"; echo "/tmp/constraints_$1.txt"; }
CU121="--index-url https://download.pytorch.org/whl/cu121"
CU124="--index-url https://download.pytorch.org/whl/cu124"

# 1. tools environment
newenv tools 3.11
conda install -y -q -n tools -c conda-forge --override-channels git git-lfs
"$(py tools)" -m pip install -q gdown huggingface_hub numpy soundfile pyyaml
git lfs install

# 2. upstream repositories at the pinned commits + patches
mkdir -p repos
while read -r name url sha patch; do
  [ -d "repos/$name" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --recursive -q "$url" "repos/$name"
  git -C "repos/$name" checkout -q "$sha"
  git -C "repos/$name" submodule update --init --recursive -q
  if [ "$patch" != "-" ] && ! git -C "repos/$name" apply --check -R "$ROOT/patches/$patch" 2>/dev/null; then
    git -C "repos/$name" apply "$ROOT/patches/$patch" && echo "patched $name"
  fi
done <<'EOF'
CosyVoice https://github.com/FunAudioLLM/CosyVoice 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc cosyvoice.diff
index-tts https://github.com/index-tts/index-tts d9e41aac89fd00b3d71497fddb287b7f24613712 indextts.diff
GLM-TTS https://github.com/zai-org/GLM-TTS 4b944f4be7b6c55454751715081f4dc83992897a glmtts.diff
F5-TTS https://github.com/SWivid/F5-TTS 2f866649b779881af8bd644e76b9a6d95053da29 f5tts.diff
ZipVoice https://github.com/k2-fsa/ZipVoice 2f7326fbfe999a3ad179e3f1af82a424d4a62819 zipvoice.diff
DMOSpeech2 https://github.com/yl4579/DMOSpeech2 10c2255a59b96c311c317536603a18672e69ac84 dmospeech2.diff
seed-tts-eval https://github.com/BytedanceSpeech/seed-tts-eval 752f4297f090c46bb1a55a1f7439e5944ddefe8d -
UTMOSv2 https://github.com/sarulab-speech/UTMOSv2 cc2700db57bb83ee13dc31ebe1b868c254e15d09 -
NISQA https://github.com/gabrielmittag/NISQA fe84f0f252abec382b24367d5b22498a7ce34dbb -
DNS-Challenge https://github.com/microsoft/DNS-Challenge 591184a9fcb2cbdec02520fed81a32bbbf9d73ff -
EOF

# 3. conda environments
newenv env_cosy 3.10
"$(py env_cosy)" -m pip install -q "setuptools<81" wheel pyyaml
"$(py env_cosy)" -m pip install -q --no-build-isolation openai-whisper==20231117
"$(py env_cosy)" -m pip install -q -r repos/CosyVoice/requirements.txt

newenv env_index 3.11
"$(py env_index)" -m pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$(py env_index)" -m pip install -q -e repos/index-tts pyyaml -c "$(pins env_index)"

newenv env_glm 3.10
conda install -y -q -n env_glm -c conda-forge --override-channels pynini
"$(py env_glm)" -m pip install -q torch==2.3.1 torchaudio==2.3.1 $CU121
"$(py env_glm)" -m pip install -q -r repos/GLM-TTS/requirements.txt pyyaml -c "$(pins env_glm)"

newenv env_zip 3.10
"$(py env_zip)" -m pip install -q torch==2.5.1 torchaudio==2.5.1 $CU124
"$(py env_zip)" -m pip install -q -r repos/ZipVoice/requirements.txt urllib3 requests

newenv env_f5dmo 3.10
"$(py env_f5dmo)" -m pip install -q torch==2.5.1 torchaudio==2.5.1 $CU124
"$(py env_f5dmo)" -m pip install -q "setuptools<81" pyyaml
"$(py env_f5dmo)" -m pip install -q -e repos/F5-TTS "torch==2.5.1" "torchaudio==2.5.1"
"$(py env_f5dmo)" -m pip install -q -r repos/DMOSpeech2/requirements.txt "torch==2.5.1" "torchaudio==2.5.1" ipython

newenv env_eval 3.10
"$(py env_eval)" -m pip install -q torch==2.5.1 torchaudio==2.5.1 $CU124
"$(py env_eval)" -m pip install -q "setuptools<81" "transformers==4.46.3" "jiwer==3.0.5" funasr modelscope zhon zhconv \
    speechbrain==1.0.3 librosa soundfile scipy pandas tqdm s3prl "numpy<2" "huggingface_hub<1.0" "datasets<4" sentencepiece \
    accelerate addict simplejson sortedcontainers oss2 pillow pyworld matplotlib pyyaml "torch==2.5.1" "torchaudio==2.5.1"

OBJQ="$ENVS/env_objq"
[ -x "$OBJQ/bin/python" ] || "$(py env_eval)" -m venv --system-site-packages "$OBJQ"
"$OBJQ/bin/python" -m pip install -q -e repos/UTMOSv2 onnxruntime==1.23.2 timm==1.0.30 torchvision==0.20.1

# 4. checkpoints, test data and text-normalisation resources
"$(py tools)" -m fsu.assets
"$(py env_cosy)" -c "from modelscope import snapshot_download; print(snapshot_download('pengzhendong/wetext'))"

# 5. test lists
"$(py env_eval)" -m fsu.datasets build
"$(py env_eval)" -m fsu.datasets build-robust
echo "setup complete. Next: see README.md"
