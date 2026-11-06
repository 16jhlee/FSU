# E17 (v10.9): one generation queue (GLM heavy first, then CosyVoice 3 tokens + decodes) and ONE evaluation queue
# (GPU limit: generation 1 + evaluation 1; the heavy GLM decodes and the evaluations exclude each other via gpu_lock).
Set-Location D:\research\speechFM
$py = 'C:\Users\Rose\miniconda3\envs\env_eval\python.exe'
"$(Get-Date -Format s) start" | Out-File -Append -Encoding utf8 logs\chain_e17.log
$g = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList 'scripts/queue_gen.py e17_jobs' -RedirectStandardOutput logs\E17_gen.out -RedirectStandardError logs\E17_gen.err
Start-Sleep 60
$e = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList 'scripts/queue_eval.py e17_jobs' -RedirectStandardOutput logs\E17_eval.out -RedirectStandardError logs\E17_eval.err
$g.WaitForExit()
"$(Get-Date -Format s) gen done" | Out-File -Append -Encoding utf8 logs\chain_e17.log
$e.WaitForExit()
# the evaluation queue exits when nothing is ready yet: rerun it until every folder carries _qdone
for ($i = 0; $i -lt 3; $i++) {
  $e = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList 'scripts/queue_eval.py e17_jobs' -RedirectStandardOutput "logs\E17_eval_r$i.out" -RedirectStandardError "logs\E17_eval_r$i.err"
  $e.WaitForExit()
}
"$(Get-Date -Format s) all done" | Out-File -Append -Encoding utf8 logs\chain_e17.log