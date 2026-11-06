"""E15 smoke (v10.8 §2): in the official CosyVoice2 chunk-aware streaming path (run_cosy.py --stream 1), is each rule applied
to every chunk's flow call with the same step schedule? long_en, first 3 utterances, NFE 2, seed 0. The decoder's state dump
(patches/CosyVoice_dump_states.diff) records every evaluation; per flow call (a new call starts at t = 0) we report the
evaluation times and the effective weight w_eff = <v - v_c, v_c - v_u> / ||v_c - v_u||^2.
Expected: tvsf w_eff = (g_k - 1) of tv_scales on every call; seltext 0.7 with the speaker-null row; mambo4 w exp(-4 r) < 0.7.
python scripts/e15_stream_smoke.py   (env_cosy) -> logs/e15_stream_smoke.txt
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cosy as RC  # noqa: E402

RULES = {"tvsf": ["--tv", "symmetric_fix"], "seltext": ["--cfg_branch", "speaker"], "mambo4": ["--mambo_alpha", "4"]}
META = ROOT / "out/smoke/E15_stream/long_en_3.meta.lst"
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
            vc, vu, v = d["vc"].double(), d["vu"].double(), d["v"].double()
            diff = vc - vu
            cur.append((round(d["t"], 3), float(((v - vc) * diff).sum() / (diff * diff).sum())))
        dec.dump_states = None
        sched = {tuple(t for t, _ in c) for c in calls}
        ws = [tuple(round(w, 3) for _, w in c) for c in calls]
        OUT.append(f"{Path(a.out_dir).name}: flow calls {len(calls)}, step-start times per call {sorted(sched)}, "
                   f"w_eff first calls {ws[:4]} ... last {ws[-1]}; distinct w_eff tuples {len(set(ws))}")
        print(OUT[-1], flush=True)
    RC.stage_decode = wrapped
    for rule, args in RULES.items():
        sys.argv = ["run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", str(META), "--stage", "decode",
                    "--cache_dir", "cache/cosy2_tokens/long_en", "--noise", "utt:0", "--out_dir",
                    str(ROOT / "out/smoke/E15_stream" / rule), "--cfg", "0.7", "--nfe", "2", "--stream", "1", *args]
        RC.main()
    (ROOT / "logs/e15_stream_smoke.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
