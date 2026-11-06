# speechFM — fresh-machine setup (Windows, NVIDIA GPU, Miniconda installed).
# Run from the repo root in PowerShell:   powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
# Re-runnable: each step skips work that is already done.
# Exact package versions that produced the reported results: env\lock_<env>.txt (pip freeze, Windows).
$ErrorActionPreference = 'Stop'
$ROOT = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ROOT
$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) { $conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe" }
$envs = Join-Path (Split-Path (Split-Path $conda)) 'envs'
function Py($n) { Join-Path $envs "$n\python.exe" }
function NewEnv($n, $ver) { if (-not (Test-Path (Py $n))) { & $conda create -y -n $n -c conda-forge --override-channels "python=$ver" pip -q } }

# 1. tools env (git, git-lfs, downloaders)
NewEnv 'tools' '3.11'
& $conda install -y -n tools -c conda-forge --override-channels git git-lfs -q
& (Py 'tools') -m pip install -q gdown huggingface_hub numpy soundfile
$env:Path = "$envs\tools\Library\bin;$env:Path"
git lfs install

# 2. upstream repos at pinned commits (env\pins.txt) + our sampler-level patches (patches\*.diff)
New-Item -ItemType Directory -Force repos | Out-Null
$pins = @{ 'CosyVoice' = @('https://github.com/FunAudioLLM/CosyVoice', '074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc', 'CosyVoice_sampler.diff');
           'ZipVoice' = @('https://github.com/k2-fsa/ZipVoice', '2f7326fbfe999a3ad179e3f1af82a424d4a62819', 'ZipVoice_seed.diff');
           'F5-TTS' = @('https://github.com/SWivid/F5-TTS', '2f866649b779881af8bd644e76b9a6d95053da29', 'F5TTS_E8p.diff');
           'index-tts' = @('https://github.com/index-tts/index-tts', 'd9e41aac89fd00b3d71497fddb287b7f24613712', 'IndexTTS_E8p.diff');
           'DMOSpeech2' = @('https://github.com/yl4579/DMOSpeech2', '10c2255a59b96c311c317536603a18672e69ac84', 'DMOSpeech2_sampler.diff');
           'FlashTTS' = @('https://github.com/ASLP-lab/FlashTTS', 'd1d9dc0cff1dcb06a4f38d635f33b8f6cfeb3b4d', $null);
           'seed-tts-eval' = @('https://github.com/BytedanceSpeech/seed-tts-eval', '752f4297f090c46bb1a55a1f7439e5944ddefe8d', $null) }
git config --global core.longpaths true
foreach ($r in $pins.Keys) {
    $url, $sha, $patch = $pins[$r]
    if (-not (Test-Path "repos\$r")) { $env:GIT_LFS_SKIP_SMUDGE = '1'; git clone --recursive -q $url "repos\$r" }
    git -C "repos\$r" checkout -q $sha
    git -C "repos\$r" submodule update --init --recursive -q
    if ($patch) {
        git -C "repos\$r" apply --check -R "$ROOT\patches\$patch" 2>$null
        if ($LASTEXITCODE -ne 0) { git -C "repos\$r" apply "$ROOT\patches\$patch"; "patched $r" } else { "$r already patched" }
    }
}
# incremental CosyVoice patches (applied on top of CosyVoice_sampler.diff, in order)
foreach ($rp in @(@('CosyVoice', 'CosyVoice_dump_states.diff'), @('CosyVoice', 'CosyVoice_E5.diff'),
                  @('CosyVoice', 'CosyVoice_E7.diff'), @('CosyVoice', 'CosyVoice_branch.diff'), @('CosyVoice', 'CosyVoice_E8p7.diff'),
                  @('ZipVoice', 'ZipVoice_E5.diff'), @('ZipVoice', 'ZipVoice_E7.diff'), @('index-tts', 'IndexTTS_branch.diff'))) {
    $r, $p = $rp
    git -C "repos\$r" apply --check -R "$ROOT\patches\$p" 2>$null
    if ($LASTEXITCODE -ne 0) { git -C "repos\$r" apply "$ROOT\patches\$p"; "applied $p" }
}

# 3. conda envs (same recipe as the original run; see README "Environments")
$cu = '--index-url', 'https://download.pytorch.org/whl/cu124'
NewEnv 'env_cosy' '3.10'
& (Py 'env_cosy') -m pip install -q "setuptools<81" wheel
& (Py 'env_cosy') -m pip install -q --no-build-isolation openai-whisper==20231117
& (Py 'env_cosy') -m pip install -q -r repos\CosyVoice\requirements.txt

NewEnv 'env_zip' '3.10'
& (Py 'env_zip') -m pip install -q torch==2.5.1 torchaudio==2.5.1 @cu
& (Py 'env_zip') -m pip install -q -r repos\ZipVoice\requirements.txt urllib3 requests

NewEnv 'env_f5dmo' '3.10'
& (Py 'env_f5dmo') -m pip install -q torch==2.5.1 torchaudio==2.5.1 @cu
& (Py 'env_f5dmo') -m pip install -q "setuptools<81"
& (Py 'env_f5dmo') -m pip install -q -e repos\F5-TTS "torch==2.5.1" "torchaudio==2.5.1"
& (Py 'env_f5dmo') -m pip install -q -r repos\DMOSpeech2\requirements.txt "torch==2.5.1" "torchaudio==2.5.1" ipython

NewEnv 'env_index' '3.11'   # IndexTTS-2 (v5): torch 2.8 cu128, official package editable; versions in env\lock_env_index.txt
& (Py 'env_index') -m pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& (Py 'env_index') -m pip install -q -e repos\index-tts -c env\lock_env_index.txt

NewEnv 'env_eval' '3.10'
& (Py 'env_eval') -m pip install -q torch==2.5.1 torchaudio==2.5.1 @cu
& (Py 'env_eval') -m pip install -q "setuptools<81" "transformers==4.46.3" "jiwer==3.0.5" funasr modelscope zhon zhconv `
    speechbrain==1.0.3 librosa soundfile scipy pandas tqdm s3prl "numpy<2" "huggingface_hub<1.0" "datasets<4" sentencepiece `
    accelerate addict simplejson sortedcontainers oss2 pillow pyworld matplotlib "torch==2.5.1" "torchaudio==2.5.1"

# 4. assets (checkpoints, Seed-TTS-eval, WavLM-SV, MiniMax) + wetext TN resources for CosyVoice
& (Py 'tools') scripts\download_assets.py
& (Py 'env_cosy') -c "from modelscope import snapshot_download; print(snapshot_download('pengzhendong/wetext'))"

# 5. re-point absolute paths (lists, prompt caches) to this checkout
& (Py 'tools') scripts\relocate.py

# 6. optional: restore the large-output backup (wavs, per-dir metrics, prompt resynthesis) if available
#    see README "Restoring generated outputs".
"setup complete. Next: see README.md"
