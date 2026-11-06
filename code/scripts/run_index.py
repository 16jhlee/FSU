"""IndexTTS-2 runner (v5 E0' / E8'), env_index. Uses patches/IndexTTS_E8p.diff.

Stage `cond`: official IndexTTS2.infer (GPT sampling seeded per utterance; emotion = speaker prompt, the
  default) -> saves the official wav (25 steps, w=0.7; = E0' reproduction) AND caches, per text segment, the
  exact inputs of the s2mel CFM call (cat_condition, x_lens, ref_mel, style) -> cache/index2/<set>/<utt>.pt.
  The s2mel content condition is therefore fixed for every later decode (paired comparisons, fixed length).
Stage `decode`: s2mel CFM only from the cache with --nfe, --cfg, --cfg_interval/--cfg_ramp (step start time,
  repo time t=0 noise; official schedule linspace(0,1,N+1)), per-utterance noise seed hash(utt|noise|k),
  then BigVGAN; segments joined with the official 200 ms silence and fade_out_tail.
"""
import argparse
import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "index-tts"
sys.path.insert(0, str(REPO))


def tv_scales(t_span, w, kind):  # scripts/tvcfg.py, loaded by path (scripts/ on sys.path would shadow the repo's modules)
    import importlib.util
    spec = importlib.util.spec_from_file_location("tvcfg", ROOT / "scripts" / "tvcfg.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.tv_scales(t_span, w, kind)


def h(s, mod=2**31 - 1):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16) % mod


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_meta(p):
    return [l.split("|")[:4] for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_model():
    from indextts.infer_v2 import IndexTTS2
    md = str(ROOT / "ckpt" / "IndexTTS-2")
    return IndexTTS2(cfg_path=os.path.join(md, "config.yaml"), model_dir=md, use_fp16=False,
                     use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=False)


def stage_cond(tts, rows, cache, out):
    cfm = tts.s2mel.models["cfm"]
    orig = cfm.inference
    captured = []

    def capture(mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        captured.append(dict(mu=mu.detach().cpu(), x_lens=x_lens.detach().cpu(), prompt=prompt.detach().cpu(),
                             style=style.detach().cpu(), n_timesteps=n_timesteps, cfg=inference_cfg_rate))
        return orig(mu, x_lens, prompt, style, f0, n_timesteps, temperature=temperature,
                    inference_cfg_rate=inference_cfg_rate)
    cfm.inference = capture
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for utt, ptext, pwav, text in rows:
        if (cache / f"{utt}.pt").exists() and (out / f"{utt}.wav").exists():
            continue
        captured.clear()
        seed_all(h(utt))
        tts.infer(spk_audio_prompt=pwav, text=text, output_path=str(out / f"{utt}.wav"), verbose=False)
        torch.save(captured.copy(), cache / f"{utt}.pt")
        print(utt, len(captured), [c["mu"].shape[1] for c in captured], flush=True)
    cfm.inference = orig


def install_schedule_solver(cfm, t_span=None, solver="euler"):
    """E8'-8: explicit schedule (--t_span, repo time) and/or midpoint solver built from single Euler-step calls of the
    (patched) original solve_euler, so CFG interval / branch code is shared. midpoint: v1@(x,t), v2@(x+dt/2 v1,t+dt/2)."""
    orig_solve = cfm.solve_euler
    ts_fixed = [float(v) for v in t_span.split(",")] if t_span else None

    def solve(x, x_lens, prompt, mu, style, f0, ts, cfg):
        if ts_fixed is not None:
            ts = torch.tensor(ts_fixed, device=x.device)
        if solver == "euler":
            return orig_solve(x, x_lens, prompt, mu, style, f0, ts, cfg)
        P = prompt.size(-1)
        x = x.clone()
        x[..., :P] = 0

        def vel(z, t):  # one Euler step of length 1 from time t: returns v(z, t) (prompt region zeroed)
            y = orig_solve(z.clone(), x_lens, prompt, mu, style, f0, torch.tensor([t, t + 1.0], device=z.device), cfg)
            v = y - z
            v[..., :P] = 0
            return v
        tl = [float(v) for v in ts]
        for i, (t0, t1) in enumerate(zip(tl[:-1], tl[1:])):
            d = t1 - t0
            v1 = vel(x, t0)
            if solver == "mid1" and i > 0:  # E10-4: midpoint for the first step only, Euler afterwards (as run_cosy)
                x = x + d * v1
            else:
                x = x + d * vel(x + 0.5 * d * v1, t0 + 0.5 * d)
            x[..., :P] = 0
        return x
    cfm.solve_euler = solve


def stage_decode(tts, rows, cache, out, a):
    from indextts.utils.common import fade_out_tail
    cfm = tts.s2mel.models["cfm"]
    cfm.cfg_interval = [float(x) for x in a.cfg_interval.split(",")] if a.cfg_interval else None
    cfm.cfg_ramp = a.cfg_ramp
    cfm.cfg_branch = a.cfg_branch
    cfm.cfg_steps = [float(x) for x in a.cfg_steps.split(",")] if a.cfg_steps else None  # E13: per-step w_k
    if cfm.cfg_steps is not None:
        assert len(cfm.cfg_steps) == a.nfe, (cfm.cfg_steps, a.nfe)
    if a.t_span or a.solver != "euler":
        install_schedule_solver(cfm, a.t_span, a.solver)
    # E14R-2 external rules (patches/IndexTTS_E14.diff; formulas as patches/CosyVoice_E8p7.diff / CosyVoice_E8p10.diff)
    cfm.cfg_branch_switch_t = a.branch_switch_t  # Selective-CFG def_text: all-null while t < this, then cfg_branch
    cfm.mambo_alpha = a.mambo_alpha  # MAMBO-G: w_k = w exp(-alpha ||v_c - v_u|| / ||v_u||)
    cfm.cfg_zero_star = bool(a.cfg_zero_star)  # CFG-Zero*: v_u scaled by s* = <v_c,v_u>/||v_u||^2
    cfm.zero_init_steps = a.zero_init  # CFG-Zero*: velocity 0 for the first K steps
    cfm.tfr = bool(a.tfr)  # Terminal-Fitted Repair
    cfm.jrr = tuple(float(x) for x in a.jrr.split(",")) if a.jrr else None
    ts = [float(x) for x in a.t_span.split(",")] if a.t_span else torch.linspace(0, 1, a.nfe + 1).tolist()  # official grid
    cfm.tv_scales = tv_scales(ts, a.cfg, a.tv) if a.tv else None
    if cfm.tv_scales:
        print("TV-CFG scales g_k:", [round(x, 3) for x in cfm.tv_scales], flush=True)
    k = a.noise.split(":")[1] if ":" in a.noise else "0"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "_lengths.tsv", "a", encoding="utf-8")
    for utt, *_ in rows:
        if (out / f"{utt}.wav").exists():
            continue
        segs = torch.load(cache / f"{utt}.pt")
        wavs = []
        for i, s in enumerate(segs):
            seed_all(h(f"{utt}|noise|{k}|{i}"))
            dev = tts.device
            with torch.inference_mode():
                vc = cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev),
                                   None, a.nfe, inference_cfg_rate=a.cfg)
                vc = vc[:, :, s["prompt"].size(-1):]
                wav = tts.bigvgan(vc.float()).squeeze().unsqueeze(0)
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0).squeeze(1)
            wavs.append(wav.cpu())
        wavs = tts.insert_interval_silence(wavs, sampling_rate=22050, interval_silence=200)
        wav = fade_out_tail(torch.cat(wavs, dim=1), 22050)
        torchaudio.save(str(out / f"{utt}.wav"), (wav / 32767.0).float(), 22050)
        log.write(f"{utt}\t{a.nfe}\t{a.cfg}\t{a.cfg_interval}\t{a.cfg_ramp}\t{a.noise}\t{a.cfg_branch}\t{a.t_span}\t{a.solver}\t"
                  f"{sum(int(s['mu'].shape[1]) for s in segs)}\t{wav.shape[-1]}\n")
        log.flush()
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stage", choices=["cond", "decode"], required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nfe", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--cfg_interval", default=None)
    ap.add_argument("--cfg_ramp", type=float, default=None)
    ap.add_argument("--cfg_steps", default=None, help="E13: per-step w 'w1,w2,...' (one per step; overrides --cfg)")
    ap.add_argument("--noise", default="utt:0")
    ap.add_argument("--cfg_branch", default="all", choices=["all", "speaker", "token"],
                    help="E8'-8: unconditional row erases all (official) / speaker (prompt mel+style) / token (mu)")
    ap.add_argument("--t_span", default=None, help="E8'-8: explicit schedule 'a,b,...' (overrides --nfe linspace)")
    ap.add_argument("--solver", default="euler", choices=["euler", "midpoint", "mid1"],
                    help="E8'-8: --nfe / --t_span give the STEP grid; midpoint = 2 evaluations per step")
    ap.add_argument("--branch_switch_t", type=float, default=None, help="E14R Sel-CFG def_text threshold (0.08)")
    ap.add_argument("--mambo_alpha", type=float, default=None, help="E14R MAMBO-G alpha")
    ap.add_argument("--cfg_zero_star", type=int, default=0, help="E14R CFG-Zero* optimised scale s*")
    ap.add_argument("--zero_init", type=int, default=0, help="E14R CFG-Zero* zero-init: first K steps velocity 0")
    ap.add_argument("--tfr", type=int, default=0, help="E14R Terminal-Fitted Repair (2607.07665)")
    ap.add_argument("--tv", default=None, choices=["symmetric", "stepup", "symmetric_fix", "stepup_fix"],
                    help="E14R TV-CFG schedule (2509.22007, sqt24/tvcfg)")
    ap.add_argument("--jrr", default=None, help="E14R JRR-CFG weights 'gT,gS,gI' (CosyVoice2 0.70,1.20,0.95)")
    a = ap.parse_args()
    os.chdir(REPO)
    tts = load_model()
    rows = read_meta(ROOT / a.meta if not os.path.isabs(a.meta) else a.meta)
    if a.stage == "cond":
        stage_cond(tts, rows, ROOT / a.cache_dir, ROOT / a.out_dir)
    else:
        stage_decode(tts, rows, ROOT / a.cache_dir, ROOT / a.out_dir, a)


if __name__ == "__main__":
    main()
