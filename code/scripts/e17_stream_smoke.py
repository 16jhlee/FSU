"""E17 smoke (v10.10 §1): in the official CosyVoice2 chunk-aware streaming path (run_cosy.py --stream 1), is FSU-Mid
(one midpoint step, unguided at t = 0, guided at t = 0.5) applied to every chunk's flow call? long_en, first 3 utterances,
seed 0, with the offline path as a control. The decoder's state dump (patches/CosyVoice_dump_states.diff) records every
evaluation; per flow call (a new call starts at t = 0) we report the evaluation times and w_eff =
<v - v_c, v_c - v_u> / ||v_c - v_u||^2. Expected on every call: times (0, 0.5), w_eff (0, 0.7).
python scripts/e17_stream_smoke.py   (env_cosy) -> logs/e17_stream_smoke.txt
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cosy as RC  # noqa: E402

MID = ["--nfe", "1", "--solver", "midpoint", "--cfg_interval", "0.1,1"]
META = ROOT / "out/smoke/E17_stream/long_en_3.meta.lst"
OUT = []


def main():
    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text("\n".join((ROOT / "data/lists/long_en.meta.lst").read_text(encoding="utf-8").splitlines()[:3]) + "\n",
                    encoding="utf-8")
    orig = RC.stage_decode

    def wrapped(cv, rows, cache_dir, out_dir, a):
        dec = cv.model.flow.decoder
        dec.dump_states = []
        orig(cv, rows, cache_dir, out_dir, a)
        calls, cur = [], None
        for d in dec.dump_states:
            if d["t"] == 0.0 or cur is None:
                cur = []
                calls.append(cur)
            if d.get("vu") is None:  # unguided evaluation: no unconditional call
                cur.append((round(d["t"], 3), 0.0))
                continue
            vc, vu, v = d["vc"].double(), d["vu"].double(), d["v"].double()
            diff = vc - vu
            den = float((diff * diff).sum())
            cur.append((round(d["t"], 3), float(((v - vc) * diff).sum()) / den if den > 0 else 0.0))
        dec.dump_states = None
        sched = sorted({tuple(t for t, _ in c) for c in calls})
        ws = sorted({tuple(round(w, 3) for _, w in c) for c in calls})
        ok = sched == [(0.0, 0.5)] and ws == [(0.0, 0.7)]
        OUT.append(f"{Path(a.out_dir).name}: flow calls {len(calls)}, evaluation times per call {sched}, "
                   f"distinct w_eff tuples {ws} -> {'OK' if ok else 'MISMATCH'}")
        print(OUT[-1], flush=True)
    RC.stage_decode = wrapped
    for mode, st in (("offline", []), ("streaming", ["--stream", "1"])):
        sys.argv = ["run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", str(META), "--stage", "decode",
                    "--cache_dir", "cache/cosy2_tokens/long_en", "--noise", "utt:0", "--out_dir",
                    str(ROOT / "out/smoke/E17_stream" / mode), "--cfg", "0.7", *MID, *st]
        RC.main()
    (ROOT / "logs/e17_stream_smoke.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
