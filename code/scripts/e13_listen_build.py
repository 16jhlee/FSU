"""E13-7 (v10.5) listening-test package (built, not run: running needs raters — user decision) -> samples/E13/listening/
Design (task document §E13-7, minimum design):
  models CosyVoice2, IndexTTS-2 x languages en, zh; conditions baseline 2-step, R (ours), mid+R (ours), teacher (official N);
  40 utterances per model x language from the 300 analysis subset, stratified by the utterance's 2-step SIM loss
  (teacher - baseline, ERes2Net): 10 quantile bins x 4 utterances (fixed-seed draw). Seed-0 wavs are copied, never regenerated.
  Task 1 SMOS : prompt + stimulus, "same speaker?" 1 (definitely different) .. 5 (definitely same); 160 trials + 2 checks
  Task 2 ABX  : X = prompt, A / B = teacher and one condition (baseline / R / mid+R), sides randomised; "which is closer to X";
                120 trials + 2 checks
  Task 3 CMOS : naturalness of R (ours) vs teacher, -3..+3 (B better than A), sides randomised; 40 trials + 2 checks
  Attention checks: SMOS prompt vs itself (expect >= 4) and vs another speaker's teacher output (expect <= 2); ABX with B =
  another speaker (expect the teacher); CMOS identical pair (expect 0 +- 1).
Files are renamed to opaque ids; the mapping is in _key/ (git-ignored). Per task x model x language: trials CSV + a
self-contained HTML page that plays the trials in a per-rater random order and downloads the responses as CSV.
python scripts/e13_listen_build.py   (env_eval);  analysis: scripts/e13_listen_analyze.py
"""
import csv
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e8p8_jobs as J8  # noqa: E402

ROOT = J8.ROOT
OUT = ROOT / "samples" / "E13" / "listening"
LISTS = {"en": "data/lists/en_all", "zh": "data/lists/zh_all"}
MODELS = {"cv2": dict(base="e82_cv2_all_n2_def", R="e82_cv2_all_n2_R", midR="A_cv2_midR", teacher="e82_cv2_all_n10_def"),
          "idx": dict(base="e81_idx_n2_def", R="e81_idx_n2_R", midR="A_idx_midR", teacher="e81_idx_n25_def")}
LAB = {"base": "baseline 2-step", "R": "R (ours)", "midR": "mid+R (ours)", "teacher": "teacher"}


def hid(*a):
    return hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()[:10]


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{font-family:system-ui,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;line-height:1.5}}
.t{{border:1px solid #ccc;border-radius:8px;padding:12px;margin:12px 0}} audio{{width:100%}} button{{margin:4px}}</style></head>
<body><h1>{title}</h1><p>{instr}</p><p>Rater ID: <input id=rid></p><div id=box></div><p id=prog></p>
<button onclick="dl()">Download responses (CSV)</button>
<script>const T={trials};const OPTS={opts};let order=T.map((_,i)=>i);
function seed(s){{let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))>>>0;return()=>{{h=(h*1103515245+12345)>>>0;return h/4294967296}}}}
let R={{}},i=0;function start(){{const r=seed(document.getElementById('rid').value||'x');for(let k=order.length-1;k>0;k--){{const j=Math.floor(r()*(k+1));[order[k],order[j]]=[order[j],order[k]]}}show()}}
function show(){{if(i>=order.length){{document.getElementById('box').innerHTML='<p>Done. Please download your responses.</p>';return}}
const t=T[order[i]];let h='<div class=t>';for(const [lab,src] of t.audio)h+='<p><b>'+lab+'</b><br><audio controls src="'+src+'"></audio></p>';
h+='<p>'+OPTS.map(o=>'<button onclick="ans(\\''+o[0]+'\\')">'+o[1]+'</button>').join('')+'</p></div>';
document.getElementById('box').innerHTML=h;document.getElementById('prog').textContent=(i+1)+' / '+order.length}}
function ans(v){{R[T[order[i]].id]=v;i++;show()}}
function dl(){{let s='rater,trial_id,response\\n';for(const k in R)s+=document.getElementById('rid').value+','+k+','+R[k]+'\\n';
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}}));a.download='responses_{tag}_'+(document.getElementById('rid').value||'x')+'.csv';a.click()}}
document.getElementById('rid').addEventListener('change',start);</script></body></html>"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "audio").mkdir(parents=True)
    (OUT / "_key").mkdir()
    rng = np.random.default_rng(20261102)
    key = []
    for m, cfg in MODELS.items():
        for s, lst in LISTS.items():
            meta = {l.split("|")[0]: l.split("|") for l in (ROOT / (lst + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()}
            T, B = EA.load_metrics(J8.wav_dir(cfg["teacher"], 0, s)), EA.load_metrics(J8.wav_dir(cfg["base"], 0, s))
            loss = (T.sim_eres2net - B.sim_eres2net).dropna().sort_values()
            bins = np.array_split(loss.index.values, 10)
            utts = [u for b in bins for u in rng.choice(b, 4, replace=False)]
            tag = f"{m}_{s}"

            def put(path, *idparts):
                i = hid(*idparts)
                dst = OUT / "audio" / f"{i}.wav"
                if not dst.exists():
                    shutil.copyfile(path, dst)
                return f"../audio/{i}.wav", i
            smos, abx, cmos = [], [], []
            for u in utts:
                p_src, p_id = put(meta[u][2], "prompt", u)
                stim = {c: put(J8.wav_dir(cfg[c], 0, s) / f"{u}.wav", tag, c, u) for c in ("base", "R", "midR", "teacher")}
                for c, (src, i) in stim.items():
                    t = hid("smos", tag, c, u)
                    smos.append(dict(id=t, audio=[["Prompt", p_src], ["Sample", src]]))
                    key.append(dict(task="smos", model=m, set=s, trial_id=t, utt=u, cond=c, loss=float(loss[u]), check=""))
                for c in ("base", "R", "midR"):
                    t = hid("abx", tag, c, u)
                    flip = bool(rng.integers(2))
                    a, b = (stim[c][0], stim["teacher"][0]) if flip else (stim["teacher"][0], stim[c][0])
                    abx.append(dict(id=t, audio=[["X (prompt)", p_src], ["A", a], ["B", b]]))
                    key.append(dict(task="abx", model=m, set=s, trial_id=t, utt=u, cond=c, loss=float(loss[u]),
                                    check="", cond_side="A" if flip else "B"))
                t = hid("cmos", tag, u)
                flip = bool(rng.integers(2))
                a, b = (stim["R"][0], stim["teacher"][0]) if flip else (stim["teacher"][0], stim["R"][0])
                cmos.append(dict(id=t, audio=[["A", a], ["B", b]]))
                key.append(dict(task="cmos", model=m, set=s, trial_id=t, utt=u, cond="R", loss=float(loss[u]), check="",
                                cond_side="A" if flip else "B"))
            # attention checks
            u0, u1 = utts[0], utts[-1]
            p0, _ = put(meta[u0][2], "prompt", u0)
            other = put(J8.wav_dir(cfg["teacher"], 0, s) / f"{u1}.wav", tag, "teacher", u1)[0]
            for t, aud, chk in ((hid("chk1", tag), [["Prompt", p0], ["Sample", p0]], "same>=4"),
                                (hid("chk2", tag), [["Prompt", p0], ["Sample", other]], "different<=2")):
                smos.append(dict(id=t, audio=aud))
                key.append(dict(task="smos", model=m, set=s, trial_id=t, utt=u0, cond="check", check=chk))
            tch = put(J8.wav_dir(cfg["teacher"], 0, s) / f"{u0}.wav", tag, "teacher", u0)[0]
            for n_, (a, b, side) in enumerate(((tch, other, "A"), (other, tch, "B"))):
                t = hid("chkabx", tag, n_)
                abx.append(dict(id=t, audio=[["X (prompt)", p0], ["A", a], ["B", b]]))
                key.append(dict(task="abx", model=m, set=s, trial_id=t, utt=u0, cond="check", check=f"answer={side}"))
            for n_ in range(2):
                t = hid("chkcmos", tag, n_)
                cmos.append(dict(id=t, audio=[["A", tch], ["B", tch]]))
                key.append(dict(task="cmos", model=m, set=s, trial_id=t, utt=u0, cond="check", check="abs<=1"))
            lang = "English" if s == "en" else "Chinese (Mandarin)"
            for task, trials, opts, instr in (
                    ("smos", smos, [[str(i), str(i)] for i in range(1, 6)],
                     f"{lang}. Listen to the prompt and the sample. Is the sample spoken by the same person as the prompt? "
                     "1 = definitely different, 3 = unsure, 5 = definitely the same. Ignore audio quality and content."),
                    ("abx", abx, [["A", "A is closer"], ["B", "B is closer"]],
                     f"{lang}. Listen to X, then A and B. Which of A and B sounds more like the same speaker as X?"),
                    ("cmos", cmos, [[str(i), str(i)] for i in range(-3, 4)],
                     f"{lang}. Compare the naturalness of A and B. -3 = A much better, 0 = equal, +3 = B much better.")):
                d = OUT / task
                d.mkdir(exist_ok=True)
                (d / f"{tag}.html").write_text(PAGE.format(title=html.escape(f"{task.upper()} {tag}"), instr=html.escape(instr),
                                                           trials=json.dumps(trials), opts=json.dumps(opts), tag=f"{task}_{tag}"),
                                               encoding="utf-8")
    with open(OUT / "_key" / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "set", "trial_id", "utt", "cond", "loss", "check", "cond_side"])
        w.writeheader()
        w.writerows(key)
    n = {t: sum(1 for k in key if k["task"] == t) for t in ("smos", "abx", "cmos")}
    print("trials:", n, "| audio files:", len(list((OUT / "audio").glob("*.wav"))))


if __name__ == "__main__":
    main()
