"""Run commands one after another after a process exits (keeps GPU jobs strictly sequential).
python run_sequence.py <wait_pid|0> <log> -- cmd1 args :: cmd2 args :: ...
Each command's output is appended to <log>; a failing command does not stop the sequence (its rc is logged)."""
import ctypes
import subprocess
import sys
import time

k32 = ctypes.windll.kernel32

pid, log = int(sys.argv[1]), sys.argv[2]
rest = sys.argv[sys.argv.index("--") + 1:]
cmds, cur = [], []
for a in rest:
    if a == "::":
        cmds.append(cur)
        cur = []
    else:
        cur.append(a)
if cur:
    cmds.append(cur)
if pid:
    h = k32.OpenProcess(0x00100000, False, pid)
    if h:
        k32.WaitForSingleObject(h, 0xFFFFFFFF)
        k32.CloseHandle(h)
with open(log, "a", encoding="utf-8") as f:
    for c in cmds:
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] start: {' '.join(c)}\n")
        f.flush()
        rc = subprocess.call(c, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] rc={rc}: {' '.join(c[:3])}\n")
        f.flush()
