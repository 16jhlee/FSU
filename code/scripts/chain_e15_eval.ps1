# E15 evaluation re-split (2026-11-04): the hard-set evaluation needs ~20 GB RAM, so it must not run beside another evaluation
# (32 GB machine). Two parallel queues on disjoint prefixes (full_idx / full_glm, ~10 GB each), then the hard_ groups
# (hard 400 + LibriSpeech-PC) alone.
Set-Location D:\research\speechFM
$py = 'C:\Users\Rose\miniconda3\envs\env_eval\python.exe'
$a = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList 'scripts/queue_eval.py e15_jobs full_cv full_idx' -RedirectStandardOutput logs\E15_evalA2.out -RedirectStandardError logs\E15_evalA2.err
Start-Sleep 5
$env:QUEUE_EVAL_LOCK = 'eval_job2'
$b = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList 'scripts/queue_eval.py e15_jobs full_glm' -RedirectStandardOutput logs\E15_evalB2.out -RedirectStandardError logs\E15_evalB2.err
Remove-Item Env:QUEUE_EVAL_LOCK
$a.WaitForExit(); $b.WaitForExit()
"$(Get-Date -Format s) full done" | Out-File -Append -Encoding utf8 logs\chain_e15_eval.log
$c = Start-Process -PassThru -WindowStyle Hidden $py -ArgumentList 'scripts/queue_eval.py e15_jobs' -RedirectStandardOutput logs\E15_evalC.out -RedirectStandardError logs\E15_evalC.err
$c.WaitForExit()
"$(Get-Date -Format s) all done" | Out-File -Append -Encoding utf8 logs\chain_e15_eval.log
