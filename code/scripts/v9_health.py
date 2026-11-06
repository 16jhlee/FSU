"""v9 health check (no GPU use).  python scripts/v9_health.py
Prints: running v9 python processes (by command line), GPU memory/utilisation, newest output file per stream and its age,
last lines of the sequence logs, and failures (rc != 0, Traceback) since the last check (offset kept in logs/_health_state.json).
STALL = newest output older than 30 min while its generator process is alive.
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ["logs/v9_gen_line5.log", "logs/v9_eval_loop.log", "logs/e8p10_jobs_gen.log"]
STATE = ROOT / "logs/_health_state.json"


def procs():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        if "\t" in line and any(k in line for k in ("v9_", "run_sequence", "queue_", "run_cosy", "infer_zipvoice", "eval_metrics")):
            pid, cmd = line.split("\t", 1)
            key = cmd.split("scripts\\", 1)[-1].split("scripts/", 1)[-1][:90]
            rows.append((pid, key))
    return rows


def newest(d, pat):
    best = None
    for p in Path(d).glob(pat):
        m = p.stat().st_mtime
        if best is None or m > best[0]:
            best = (m, p)
    return best


def main():
    now = time.time()
    print(f"== v9 health {time.strftime('%m-%d %H:%M')} ==")
    P = procs()
    for pid, k in P:
        print(f"proc {pid}: {k}")
    if not P:
        print("proc: NONE running")
    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print("gpu:", g)
    workers = [k for _, k in P if not k.startswith("run_sequence")]  # sequence runners carry every queued command line
    alive = {"image": any(k.startswith(("v9_image.py gen", "v9_stagewise.py")) for k in workers),
             "audio": any(k.startswith(("v9_audio.py gen", "v9_sao.py gen")) for k in workers),
             "speech": any(("run_cosy" in k or "infer_zipvoice" in k) for k in workers)}
    for name, d, pat in [("image", ROOT / "out/E9", "*/*/*/s*/*.png"), ("audio", ROOT / "out/E9/audio", "*/*/s*/*.wav"),
                         ("speech", ROOT / "out/E8p10", "*/utt*/*/*.wav")]:
        if name == "speech":  # the LLM token stage writes cache/*.pt, not wavs
            tok = newest(ROOT / "cache", "cosy*_tokens/*/*.pt")
            b0 = newest(d, pat)
            b = max([x for x in (b0, tok) if x], key=lambda x: x[0], default=None)
            if b:
                age = (now - b[0]) / 60
                flag = " STALL?" if age > 30 and alive[name] else ("" if alive[name] else " (idle)")
                print(f"newest {name}: {b[1].relative_to(ROOT)} ({age:.0f} min ago){flag}")
            continue
        b = newest(d, pat)
        if b:
            age = (now - b[0]) / 60
            flag = " STALL?" if age > 30 and alive[name] else ("" if alive[name] else " (idle)")
            print(f"newest {name}: {b[1].relative_to(ROOT / 'out')} ({age:.0f} min ago){flag}")
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    for lf in LOGS:
        p = ROOT / lf
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        off = st.get(lf, 0)
        new = lines[off:] if off <= len(lines) else lines
        bad = [l for l in new if ("rc=" in l and "rc=0" not in l) or "Traceback" in l or "Error" in l and "Warning" not in l]
        prog = [l for l in lines if l.startswith("[") and ("done" in l or "start" in l or "rc=" in l)]
        print(f"log {lf}: last = {prog[-1][:140] if prog else '-'}")
        for b_ in bad[-6:]:
            print(f"   FAIL? {b_[:160]}")
        st[lf] = len(lines)
    STATE.write_text(json.dumps(st))


if __name__ == "__main__":
    main()
