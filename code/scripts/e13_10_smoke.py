"""E13-10 smoke (v10.6 GLM all-null, section 1): en 3 utterances, seed 0, per network evaluation of the E11 sampler:
t, whether the call's tokens / condition / speaker embedding are all zero, and the effective guidance
w_eff = <v - vc, vc - vu> / |vc - vu|^2 (0 when no unconditional call). Checks
  ga_10   : step 1 w = 0.7 (all-null row), step 2 w = 0
  ga_midR : midpoint, eval t = 0 unguided (R interval), eval t = 0.5 guided by the all-null row (w = 0.7)
python scripts/e13_10_smoke.py   (env_glm)  -> logs/e13_10_smoke.txt
"""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_glm import REPO, h, read_meta  # noqa: E402

CONFS = {"ga_10": dict(cfg_steps=[0.7, 0.0], cfg_interval=None, solver="euler", nfe=2),
         "ga_midR": dict(cfg_steps=None, cfg_interval=[0.1, 1.0], solver="midpoint", nfe=1)}


def main():
    rows = read_meta(ROOT / "data/lists/en_all.meta.lst")[:3]
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from utils import seed_util, tts_model_util, yaml_util
    flow = yaml_util.load_flow_model(os.path.join("ckpt", "flow", "flow.pt"), os.path.join("ckpt", "flow", "config.yaml"), "cuda")
    t2w = tts_model_util.Token2Wav(flow, sample_rate=24000, device="cuda")
    calls = []
    est = flow.estimator
    orig = est.forward

    def rec(*a, **k):
        z = lambda x: bool((x == 0).all())  # noqa: E731
        calls.append(dict(t=float(k["time_step_1d"][0]), tok0=z(k["text"]), cond0=z(k["condition_btd"]), spk0=z(k["spkr_emb_bd"])))
        return orig(*a, **k)
    est.forward = rec
    out = []
    for name, c in CONFS.items():
        flow.inference_cfg_rate, flow.cfg_branch = 0.7, "allnull"
        flow.cfg_steps, flow.cfg_interval, flow.solver, flow.t_span_override = c["cfg_steps"], c["cfg_interval"], c["solver"], None
        for utt, *_ in rows:
            segs = torch.load(ROOT / "cache/glm_tokens/en" / f"{utt}.pt")
            g = torch.Generator().manual_seed(h(f"{utt}|noise|0"))
            flow.noise_override = torch.randn([1, 6000, flow.mel_dim], generator=g)
            s = segs[0]
            calls.clear()
            flow.dump_states = []
            seed_util.set_seed(h(f"{utt}|hift|0"))
            with torch.inference_mode():
                t2w.token2wav_with_cache(s["token"], n_timesteps=c["nfe"], prompt_token=s["prompt_token"],
                                         prompt_feat=s["prompt_feat"], embedding=s["embedding"])
            ci = iter(calls)
            for d in flow.dump_states:
                cc = next(ci)
                line = f"{name} {utt} eval t={d['t']:.3f} cond-call zero(tok,cond,spk)={cc['tok0'], cc['cond0'], cc['spk0']}"
                if d["vu"] is not None:
                    cu = next(ci)
                    diff = (d["vc"] - d["vu"]).double()
                    w = float(((d["v"] - d["vc"]).double() * diff).sum() / (diff * diff).sum())
                    line += f" | uncond-call zero(tok,cond,spk)={cu['tok0'], cu['cond0'], cu['spk0']} w_eff={w:.4f}"
                else:
                    line += " | no uncond call, w_eff=0"
                out.append(line)
                print(line, flush=True)
            flow.dump_states = None
    (ROOT / "logs/e13_10_smoke.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
