# Fresh-machine setup on Windows (NVIDIA GPU, Miniconda installed). The reported results were produced on Windows 11.
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
# Re-runnable: each step skips work that is already done. Exact package versions: env\lock_<env>.txt.
$ErrorActionPreference = 'Stop'
$ROOT = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ROOT
$conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
if (-not $conda) { $conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe" }
$envs = Join-Path (Split-Path (Split-Path $conda)) 'envs'
function Py($n) { Join-Path $envs "$n\python.exe" }
function NewEnv($n, $ver) { if (-not (Test-Path (Py $n))) { & $conda create -y -n $n -c conda-forge --override-channels "python=$ver" pip -q } }
function Pins($n) {  # pip constraints from env\lock_<env>.txt (local-build and editable lines removed)
    $c = Join-Path $env:TEMP "constraints_$n.txt"
    Get-Content "env\lock_$n.txt" | Where-Object { $_ -match '^[A-Za-z0-9_.\-\[\]]+==' } | Set-Content -Encoding ascii $c
    return $c
}
$cu121 = '--index-url', 'https://download.pytorch.org/whl/cu121'
$cu124 = '--index-url', 'https://download.pytorch.org/whl/cu124'

# 1. tools environment (git, downloaders, experiment queues)
NewEnv 'tools' '3.11'
& $conda install -y -n tools -c conda-forge --override-channels git git-lfs -q
& (Py 'tools') -m pip install -q gdown huggingface_hub numpy soundfile pyyaml
$env:Path = "$envs\tools\Library\bin;$env:Path"
git lfs install

# 2. upstream repositories at the pinned commits (env\pins.txt) + the sampler-level patches (patches\<repo>.diff)
New-Item -ItemType Directory -Force repos | Out-Null
$repos = [ordered]@{
    'CosyVoice'     = @('https://github.com/FunAudioLLM/CosyVoice', '074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc', 'cosyvoice.diff');
    'index-tts'     = @('https://github.com/index-tts/index-tts', 'd9e41aac89fd00b3d71497fddb287b7f24613712', 'indextts.diff');
    'GLM-TTS'       = @('https://github.com/zai-org/GLM-TTS', '4b944f4be7b6c55454751715081f4dc83992897a', 'glmtts.diff');
    'F5-TTS'        = @('https://github.com/SWivid/F5-TTS', '2f866649b779881af8bd644e76b9a6d95053da29', 'f5tts.diff');
    'ZipVoice'      = @('https://github.com/k2-fsa/ZipVoice', '2f7326fbfe999a3ad179e3f1af82a424d4a62819', 'zipvoice.diff');
    'DMOSpeech2'    = @('https://github.com/yl4579/DMOSpeech2', '10c2255a59b96c311c317536603a18672e69ac84', 'dmospeech2.diff');
    'seed-tts-eval' = @('https://github.com/BytedanceSpeech/seed-tts-eval', '752f4297f090c46bb1a55a1f7439e5944ddefe8d', $null);
    'UTMOSv2'       = @('https://github.com/sarulab-speech/UTMOSv2', 'cc2700db57bb83ee13dc31ebe1b868c254e15d09', $null);
    'NISQA'         = @('https://github.com/gabrielmittag/NISQA', 'fe84f0f252abec382b24367d5b22498a7ce34dbb', $null);
    'DNS-Challenge' = @('https://github.com/microsoft/DNS-Challenge', '591184a9fcb2cbdec02520fed81a32bbbf9d73ff', $null) }
git config --global core.longpaths true
foreach ($r in $repos.Keys) {
    $url, $sha, $patch = $repos[$r]
    if (-not (Test-Path "repos\$r")) {
        $env:GIT_LFS_SKIP_SMUDGE = '1'
        if ($r -eq 'DNS-Challenge') { git clone -q --filter=blob:none --sparse $url "repos\$r"; git -C "repos\$r" sparse-checkout set DNSMOS }
        else { git clone --recursive -q $url "repos\$r" }
    }
    git -C "repos\$r" checkout -q $sha
    git -C "repos\$r" submodule update --init --recursive -q
    if ($patch) {
        git -C "repos\$r" apply --check -R "$ROOT\patches\$patch" 2>$null
        if ($LASTEXITCODE -ne 0) { git -C "repos\$r" apply "$ROOT\patches\$patch"; "patched $r" } else { "$r already patched" }
    }
}

# 3. conda environments (one per decoder family, as in the original runs)
NewEnv 'env_cosy' '3.10'      # CosyVoice 2 / 3
& (Py 'env_cosy') -m pip install -q "setuptools<81" wheel pyyaml
& (Py 'env_cosy') -m pip install -q --no-build-isolation openai-whisper==20231117
& (Py 'env_cosy') -m pip install -q -r repos\CosyVoice\requirements.txt

NewEnv 'env_index' '3.11'     # IndexTTS-2
& (Py 'env_index') -m pip install -q torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& (Py 'env_index') -m pip install -q -e repos\index-tts pyyaml -c (Pins 'env_index')

NewEnv 'env_glm' '3.10'       # GLM-TTS (pynini from conda-forge on Windows)
& $conda install -y -n env_glm -c conda-forge --override-channels pynini -q
& (Py 'env_glm') -m pip install -q torch==2.3.1 torchaudio==2.3.1 @cu121
& (Py 'env_glm') -m pip install -q -r repos\GLM-TTS\requirements.txt pyyaml -c (Pins 'env_glm')

NewEnv 'env_zip' '3.10'       # ZipVoice
& (Py 'env_zip') -m pip install -q torch==2.5.1 torchaudio==2.5.1 @cu124
& (Py 'env_zip') -m pip install -q -r repos\ZipVoice\requirements.txt urllib3 requests

NewEnv 'env_f5dmo' '3.10'     # F5-TTS and DMOSpeech 2
& (Py 'env_f5dmo') -m pip install -q torch==2.5.1 torchaudio==2.5.1 @cu124
& (Py 'env_f5dmo') -m pip install -q "setuptools<81" pyyaml
& (Py 'env_f5dmo') -m pip install -q -e repos\F5-TTS "torch==2.5.1" "torchaudio==2.5.1"
& (Py 'env_f5dmo') -m pip install -q -r repos\DMOSpeech2\requirements.txt "torch==2.5.1" "torchaudio==2.5.1" ipython

NewEnv 'env_eval' '3.10'      # ASR, speaker embedders, UTMOS, prosody features, analysis
& (Py 'env_eval') -m pip install -q torch==2.5.1 torchaudio==2.5.1 @cu124
& (Py 'env_eval') -m pip install -q "setuptools<81" "transformers==4.46.3" "jiwer==3.0.5" funasr modelscope zhon zhconv `
    speechbrain==1.0.3 librosa soundfile scipy pandas tqdm s3prl "numpy<2" "huggingface_hub<1.0" "datasets<4" sentencepiece `
    accelerate addict simplejson sortedcontainers oss2 pillow pyworld matplotlib pyyaml "torch==2.5.1" "torchaudio==2.5.1"

$objq = Join-Path $envs 'env_objq'  # objective quality: a venv on top of env_eval + UTMOSv2 / onnxruntime
if (-not (Test-Path "$objq\Scripts\python.exe")) { & (Py 'env_eval') -m venv --system-site-packages $objq }
& "$objq\Scripts\python.exe" -m pip install -q -e repos\UTMOSv2 onnxruntime==1.23.2 timm==1.0.30 torchvision==0.20.1

# 4. checkpoints and test data (pinned revisions), wetext normalisation resources for CosyVoice
& (Py 'tools') -m fsu.assets
& (Py 'env_cosy') -c "from modelscope import snapshot_download; print(snapshot_download('pengzhendong/wetext'))"

# 5. test lists (GPU needed for the prompt-robustness transcripts)
& (Py 'env_eval') -m fsu.datasets build
& (Py 'env_eval') -m fsu.datasets build-robust
"setup complete. Next: see README.md"
