"""Wait for a running process to exit, then run a command (used to pick up configs added to a jobs module
after its queue was started; queues are resumable, so re-running only does the new work).
python queue_chain.py <pid> <log> -- <cmd ...>"""
import ctypes
import subprocess
import sys
import time

pid, log = int(sys.argv[1]), sys.argv[2]
cmd = sys.argv[sys.argv.index("--") + 1:]
k32 = ctypes.windll.kernel32
h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
if h:
    k32.WaitForSingleObject(h, 0xFFFFFFFF)
    k32.CloseHandle(h)
with open(log, "a", encoding="utf-8") as f:
    f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] chain: pid {pid} exited, running {' '.join(cmd)}\n")
    f.flush()
    sys.exit(subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT))
