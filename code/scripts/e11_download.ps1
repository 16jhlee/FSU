# E11: resume-download the remaining GLM-TTS files (HF zai-org/GLM-TTS, pinned revision) with curl.
$rev = 'b3feb691fa4cf590eec4418f6275f95220e58844'
$dst = 'C:\speechFM_store\ckpt\GLM-TTS'
$files = @('llm/model-00001-of-00002.safetensors', 'speech_tokenizer/model.safetensors', 'vocos2d/generator_jit.ckpt',
           'vq32k-phoneme-tokenizer/tokenizer.model', 'vq32k-phoneme-tokenizer/tokenizer_config.json')
foreach ($f in $files) {
    $out = Join-Path $dst ($f -replace '/', '\')
    New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
    for ($i = 0; $i -lt 40; $i++) {
        curl.exe -sL -C - --speed-limit 100000 --speed-time 60 -o $out "https://huggingface.co/zai-org/GLM-TTS/resolve/$rev/$f"
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep 5
    }
    "{0} rc={1} {2:N1} MB" -f $f, $LASTEXITCODE, ((Get-Item $out).Length / 1MB)
}
'done'
