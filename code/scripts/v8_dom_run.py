"""v8 §3.1 domain cross-check (exploration): step-wise CFG rules on official FM samplers outside speech.
python scripts/v8_dom_run.py audio|image   (env_dom, GPU)
Rules (repo time t = 1 - sigma, t=0 noise; decided at each step's start):
  baseline : CFG at every step (official)
  R        : no CFG (conditional prediction only) for steps with t < 0.1          [R (ours)]
  kynk     : CFG only for t in [lo, hi)  (Kynkäänniemi 2024 interval; image: EDM sigma (0.28, 5.42] -> t in [0.156, 0.781))
audio : TangoFlux (declare-lab, FlowMatchEulerDiscrete, sigmas linspace(1, 1/N, N), shift 1), official g = 4.5,
        10 s, AudioCaps test 100 captions. Grid: g 4.5 x NFE {2,3,4,6,8,25} x {baseline, R}; g 7 x NFE {6,8} x {baseline, R}.
image : SD3-medium (diffusers StableDiffusion3Pipeline, FlowMatchEulerDiscrete shift 3), official g = 7, 1024 px,
        COCO Karpathy test 100 captions. Grid: g 7 x NFE {4,6,8,28} x {baseline, R, kynk}; g 10 x NFE {6,8} x {baseline, R}.
Seeds: hash(id|dom) per caption (same noise for every rule / NFE). Outputs out/E8v8dom/<domain>/<cond>/<id>.{wav,png}
(+ _steps.json with each condition's CFG mask). Resumable (skips existing files).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CK = Path(r"C:\speechFM_store\ckpt")
OUT = ROOT / "out" / "E8v8dom"


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) % (2**31 - 1)


def use_cfg(rule, t):
    if rule == "baseline":
        return True
    if rule == "R":
        return t >= 0.1
    lo, hi = rule[1]
    return lo <= t < hi


def grid(domain):
    if domain == "audio":
        g = [(4.5, n, r) for n in (2, 3, 4, 6, 8, 25) for r in ("baseline", "R")]
        g += [(7.0, n, r) for n in (6, 8) for r in ("baseline", "R")]
    else:
        g = [(7.0, n, r) for n in (4, 6, 8, 28) for r in ("baseline", "R", "kynk")]
        g += [(10.0, n, r) for n in (6, 8) for r in ("baseline", "R")]
    return g


def rule_obj(r):
    return ("kynk", (0.156, 0.781)) if r == "kynk" else r


def audio():
    sys.path.insert(0, str(CK / "TangoFlux"))
    from diffusers import AutoencoderOobleck
    from model import TangoFlux, retrieve_timesteps
    from safetensors.torch import load_file
    import soundfile as sf
    cfg = json.loads((CK / "TangoFlux/config.json").read_text())
    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(str(CK / "TangoFlux/vae.safetensors")))
    model = TangoFlux(cfg)
    model.load_state_dict(load_file(str(CK / "TangoFlux/tangoflux.safetensors")), strict=False)
    vae.to("cuda").eval()
    model.to("cuda").eval()
    caps = pd.read_csv(ROOT / "data/raw/dom/audiocaps/captions.tsv", sep="\t")
    dur = 10
    for g, n, r in grid("audio"):
        cond = f"g{g}_n{n}_{r}"
        od = OUT / "audio" / cond
        od.mkdir(parents=True, exist_ok=True)
        masks = None
        for _, c in caps.iterrows():
            p = od / f"{c.id}.wav"
            if p.exists():
                continue
            with torch.no_grad():
                torch.manual_seed(h(f"{c.id}|audio"))
                duration = torch.tensor([dur], device="cuda")
                dh = model.encode_duration(duration).repeat(2, 1, 1)
                ehs, mask = model.encode_text_classifier_free([c.caption], num_samples_per_prompt=1)
                me = mask.unsqueeze(-1).expand_as(ehs)
                pooled = torch.nanmean(torch.where(me, ehs, torch.tensor(float("nan"), device=ehs.device)), dim=1)
                pp = model.fc(pooled)
                ehs = torch.cat([ehs, dh], dim=1)
                sched = model.noise_scheduler
                sig = np.linspace(1.0, 1 / n, n)
                ts, _ = retrieve_timesteps(sched, n, "cuda", None, sig)
                lat = torch.randn(1, model.audio_seq_len, 64).to("cuda")
                txt_ids = torch.zeros(2, ehs.shape[1], 3, device="cuda")
                aid = torch.arange(model.audio_seq_len).unsqueeze(0).unsqueeze(-1).repeat(2, 1, 3).to("cuda")
                masks = []
                for i, t in enumerate(ts):
                    tr = 1.0 - float(sched.sigmas[i])
                    on = use_cfg(rule_obj(r), tr)
                    masks.append(int(on))
                    npred = model.transformer(hidden_states=torch.cat([lat] * 2), timestep=torch.tensor([t / 1000], device="cuda"),
                                              guidance=None, pooled_projections=pp, encoder_hidden_states=ehs, txt_ids=txt_ids,
                                              img_ids=aid, return_dict=False)[0]
                    nu, nt = npred.chunk(2)
                    npred = nu + g * (nt - nu) if on else nt
                    lat = sched.step(npred, t, lat).prev_sample
                wave = vae.decode(lat.transpose(2, 1)).sample.cpu()[0][:, : int(dur * vae.config.sampling_rate)]
            sf.write(str(p), wave.T.float().numpy(), vae.config.sampling_rate)
        if masks is not None:
            (od / "_steps.json").write_text(json.dumps(dict(mask="".join(map(str, masks)), g=g, n=n, rule=r)))
        print(cond, "done", flush=True)


def image():
    from diffusers import StableDiffusion3Pipeline
    caps = pd.read_csv(ROOT / "data/raw/dom/coco/captions.tsv", sep="\t")
    pipe = StableDiffusion3Pipeline.from_pretrained(str(CK / "sd3-medium"), torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    emb = {}
    with torch.no_grad():
        for _, c in caps.iterrows():
            pe, ne, pp, npool = pipe.encode_prompt(prompt=c.caption, prompt_2=None, prompt_3=None, device="cuda",
                                                   do_classifier_free_guidance=True)
            emb[c.id] = (pe.cpu(), ne.cpu(), pp.cpu(), npool.cpu())
    for nm in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
        setattr(pipe, nm, None)
    torch.cuda.empty_cache()
    tr, vae, sched = pipe.transformer, pipe.vae, pipe.scheduler
    for g, n, r in grid("image"):
        cond = f"g{g}_n{n}_{r}"
        od = OUT / "image" / cond
        od.mkdir(parents=True, exist_ok=True)
        masks = None
        for _, c in caps.iterrows():
            p = od / f"{c.id}.png"
            if p.exists():
                continue
            pe, ne, pp, npool = (x.to("cuda") for x in emb[c.id])
            with torch.no_grad():
                gen = torch.Generator("cpu").manual_seed(h(f"{c.id}|image"))
                lat = torch.randn((1, tr.config.in_channels, 128, 128), generator=gen).to("cuda", torch.float16)
                sched.set_timesteps(n, device="cuda")
                masks = []
                for i, t in enumerate(sched.timesteps):
                    trp = 1.0 - float(sched.sigmas[i])
                    on = use_cfg(rule_obj(r), trp)
                    masks.append(int(on))
                    out = tr(hidden_states=torch.cat([lat] * 2), timestep=t.expand(2), encoder_hidden_states=torch.cat([ne, pe]),
                             pooled_projections=torch.cat([npool, pp]), return_dict=False)[0]
                    nu, nt = out.chunk(2)
                    npred = nu + g * (nt - nu) if on else nt
                    lat = sched.step(npred, t, lat, return_dict=False)[0]
                x = (lat / vae.config.scaling_factor) + vae.config.shift_factor
                img = vae.decode(x, return_dict=False)[0]
                img = pipe.image_processor.postprocess(img, output_type="pil")[0]
            img.save(p)
        if masks is not None:
            (od / "_steps.json").write_text(json.dumps(dict(mask="".join(map(str, masks)), g=g, n=n, rule=r)))
        print(cond, "done", flush=True)


if __name__ == "__main__":
    {"audio": audio, "image": image}[sys.argv[1]]()
