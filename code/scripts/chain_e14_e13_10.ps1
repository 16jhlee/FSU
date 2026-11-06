# E14 tail -> E13-10: wait for e14_objq.py to exit, then E14-9 throughput alone on the GPU (env_cosy),
# then the E13-10 generation + evaluation queues (queue_gen / queue_eval e13_jobs; ga_midR is heavy -> serialised).
Set-Location D:\research\speechFM
$py = 'C:\Users\Rose\miniconda3\envs'
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*e14_objq.py*' }) { Start-Sleep 30 }
"$(Get-Date -Format s) objq done" | Out-File -Append -Encoding utf8 logs\chain_e14_e13_10.log
$env:E10C_CONDS = 'R (ours) NFE2|mid+R (ours)'; $env:E10C_OUT = 'E14_midR_throughput'; $env:PYTHONIOENCODING = 'utf-8'
& "$py\env_cosy\python.exe" scripts/e10_concurrency.py > logs\E14_throughput.out 2> logs\E14_throughput.err
"$(Get-Date -Format s) throughput rc=$LASTEXITCODE" | Out-File -Append -Encoding utf8 logs\chain_e14_e13_10.log
Remove-Item Env:E10C_CONDS, Env:E10C_OUT
$g = Start-Process -PassThru -WindowStyle Hidden "$py\env_eval\python.exe" -ArgumentList 'scripts/queue_gen.py e13_jobs' -RedirectStandardOutput logs\E13_10_gen.out -RedirectStandardError logs\E13_10_gen.err
Start-Sleep 20
$e = Start-Process -PassThru -WindowStyle Hidden "$py\env_eval\python.exe" -ArgumentList 'scripts/queue_eval.py e13_jobs' -RedirectStandardOutput logs\E13_10_eval.out -RedirectStandardError logs\E13_10_eval.err
$g.WaitForExit(); $e.WaitForExit()
"$(Get-Date -Format s) e13_10 queues done gen=$($g.ExitCode) eval=$($e.ExitCode)" | Out-File -Append -Encoding utf8 logs\chain_e14_e13_10.log
