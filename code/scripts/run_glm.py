"""E11 (v10.2): GLM-TTS runner (official pipeline from repos/GLM-TTS/glmtts_inference.py; patches/GLMTTS_E11.diff in flow/flow.py).
Stage `tokens`: official text frontend + speech tokenizer + campplus embedding + LLM (official generate_long, seeded per
  utterance with hash(utt) instead of the official constant 0) -> cache/<dir>/<utt>.pt = list of segments
  {token, prompt_token, prompt_feat, embedding}. Flow is not run in this stage.
Stage `decode`: official Token2Wav.token2wav_with_cache (flow DiT + HiFT 24 kHz) per segment, segments concatenated
  as in generate_long. Options (defaults = official: NFE 10, cosine schedule, Euler, CFG 0.7, all-null branch):
    --nfe N  --cfg w  --noise utt:K (noise hash(utt|noise|K), HiFT seeded hash(utt|hift|i))  --t_span a,b,...
    --cfg_interval tmin,tmax (R (ours) = 0.1,1)  --cfg_steps=w1,w2,...  --cfg_branch all|allnull|speaker|token
    (all = official = speaker-null in the released config: speech_token_cfg False; allnull also zeroes the tokens)
    --solver euler|midpoint|mid1
python scripts/run_glm.py --meta <list.meta.lst> --stage tokens|decode --cache_dir cache/glm_tokens/en [--out_dir ...]   (env_glm)
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
import torchaudio

if not hasattr(torch, "npu"):  # the official frontend calls torch.npu.is_available() (Ascend branch) unconditionally
    from types import SimpleNamespace
    torch.npu = SimpleNamespace(is_available=lambda: False)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "GLM-TTS"


def h(s, mod=2 ** 31 - 1):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16) % mod


def read_meta(path):
    return [l.split("|")[:4] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def tv_scales(t_span, w, kind):  # scripts/tvcfg.py, loaded by path (scripts/ on sys.path would shadow the repo's modules)
    import importlib.util
    spec = importlib.util.spec_from_file_location("tvcfg", ROOT / "scripts" / "tvcfg.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.tv_scales(t_span, w, kind)


def stage_tokens(rows, cache_dir, text_frontend_on=True):
    import glmtts_inference as G
    frontend, text_frontend, _, llm, flow = G.load_models(use_phoneme=False, sample_rate=24000)
    if not text_frontend_on:  # E8'-5 rule for ko/ja/de: raw text (the zh/en normalisers misfire on these languages)
        text_frontend.text_normalize = lambda t: t
    cache_dir.mkdir(parents=True, exist_ok=True)
    for utt, ptext, pwav, text in rows:
        out = cache_dir / f"{utt}.pt"
        if out.exists():
            continue
        segs = []

        def capture(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
            segs.append(dict(token=torch.tensor(token_list, dtype=torch.long)[None], prompt_token=prompt_speech_tokens.cpu(),
                             prompt_feat=speech_feat.cpu(), embedding=embedding.cpu()))
            return torch.zeros(1, 1), None
        # as jsonl_generate()
        prompt_text = text_frontend.text_normalize(ptext)
        synth_text = text_frontend.text_normalize(text)
        prompt_text_token = frontend._extract_text_token(prompt_text + " ")
        prompt_speech_token = frontend._extract_speech_token([pwav])
        speech_feat = frontend._extract_speech_feat(pwav, sample_rate=24000)
        embedding = frontend._extract_spk_embedding(pwav)
        cache_speech_token = [prompt_speech_token.squeeze().tolist()]
        flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(G.DEVICE)
        cache = {"cache_text": [prompt_text], "cache_text_token": [prompt_text_token],
                 "cache_speech_token": cache_speech_token, "use_cache": True}
        G.generate_long(frontend=frontend, text_frontend=text_frontend, llm=llm, flow=flow, text_info=[utt, synth_text],
                        cache=cache, embedding=embedding, seed=h(utt), flow_prompt_token=flow_prompt_token,
                        speech_feat=speech_feat, device=G.DEVICE, use_phoneme=False, local_flow_forward=capture)
        torch.save(segs, out)
        print(utt, len(segs), [s["token"].shape[1] for s in segs], flush=True)


def stage_decode(rows, cache_dir, out_dir, a):
    from utils import seed_util, tts_model_util, yaml_util
    dev = "cuda"
    flow = yaml_util.load_flow_model(os.path.join("ckpt", "flow", "flow.pt"), os.path.join("ckpt", "flow", "config.yaml"), dev)
    t2w = tts_model_util.Token2Wav(flow, sample_rate=24000, device=dev)
    flow.inference_cfg_rate = a.cfg
    flow.cfg_interval = [float(x) for x in a.cfg_interval.split(",")] if a.cfg_interval else None
    flow.cfg_steps = [float(x) for x in a.cfg_steps.split(",")] if a.cfg_steps else None
    flow.cfg_branch = a.cfg_branch
    flow.solver = a.solver
    flow.t_span_override = [float(x) for x in a.t_span.split(",")] if a.t_span else None
    nfe = len(flow.t_span_override) - 1 if a.t_span else a.nfe
    # E14R-2 external rules (patches/GLMTTS_E14.diff; formulas as patches/CosyVoice_E8p7.diff / CosyVoice_E8p10.diff)
    flow.cfg_branch_switch_t = a.branch_switch_t  # Selective-CFG def_text: all-null while t < this, then cfg_branch
    flow.mambo_alpha = a.mambo_alpha  # MAMBO-G: w_k = w exp(-alpha ||v_c - v_u|| / ||v_u||)
    flow.cfg_zero_star = bool(a.cfg_zero_star)  # CFG-Zero*: v_u scaled by s* = <v_c,v_u>/||v_u||^2
    flow.zero_init_steps = a.zero_init  # CFG-Zero*: velocity 0 for the first K steps
    flow.tfr = bool(a.tfr)  # Terminal-Fitted Repair
    flow.pci = bool(a.pci)  # Prompt-Consistency Inference (prompt block = first prompt_feat frames)
    flow.jrr = tuple(float(x) for x in a.jrr.split(",")) if a.jrr else None
    ts = flow.t_span_override or (1 - torch.cos(torch.linspace(0, 1, nfe + 1) * 0.5 * torch.pi)).tolist()  # cosine grid
    flow.tv_scales = tv_scales(ts, a.cfg, a.tv) if a.tv else None
    flow.e14_rules = any([a.branch_switch_t is not None, a.mambo_alpha is not None, a.cfg_zero_star, a.zero_init, a.tfr,
                          a.pci, a.jrr, a.tv])
    if flow.tv_scales:
        print("TV-CFG scales g_k:", [round(x, 3) for x in flow.tv_scales], flush=True)
    k = a.noise.split(":")[1] if ":" in a.noise else "0"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "_lengths.tsv", "a", encoding="utf-8")
    for utt, *_ in rows:
        wp = out_dir / f"{utt}.wav"
        if wp.exists():
            continue
        segs = torch.load(cache_dir / f"{utt}.pt")
        g = torch.Generator().manual_seed(h(f"{utt}|noise|{k}"))
        flow.noise_override = torch.randn([1, 6000, flow.mel_dim], generator=g)
        wavs = []
        for i, s in enumerate(segs):
            flow.prompt_frames = int(s["prompt_feat"].shape[1])  # E14R PCI: prompt block = first prompt_feat frames
            seed_util.set_seed(h(f"{utt}|hift|{i}"))
            with torch.inference_mode():
                wav, _ = t2w.token2wav_with_cache(s["token"], n_timesteps=nfe, prompt_token=s["prompt_token"],
                                                  prompt_feat=s["prompt_feat"], embedding=s["embedding"])
            wavs.append(wav.detach().cpu())
        wav = torch.cat(wavs, dim=1)
        torchaudio.save(str(wp), wav, 24000)
        log.write(f"{utt}\t{nfe}\t{a.cfg}\t{a.noise}\t{sum(s['token'].shape[1] for s in segs)}\t{wav.shape[1]}\n")
        log.flush()
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stage", choices=["tokens", "decode"], required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir")
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--noise", default="utt:0")
    ap.add_argument("--t_span", default=None)
    ap.add_argument("--cfg_interval", default=None)
    ap.add_argument("--cfg_steps", default=None)
    ap.add_argument("--cfg_branch", default="all", choices=["all", "allnull", "speaker", "token"])
    ap.add_argument("--solver", default="euler", choices=["euler", "midpoint", "mid1"])
    ap.add_argument("--branch_switch_t", type=float, default=None, help="E14R Sel-CFG def_text threshold (0.08)")
    ap.add_argument("--mambo_alpha", type=float, default=None, help="E14R MAMBO-G alpha")
    ap.add_argument("--cfg_zero_star", type=int, default=0, help="E14R CFG-Zero* optimised scale s*")
    ap.add_argument("--zero_init", type=int, default=0, help="E14R CFG-Zero* zero-init: first K steps velocity 0")
    ap.add_argument("--tfr", type=int, default=0, help="E14R Terminal-Fitted Repair (2607.07665)")
    ap.add_argument("--pci", type=int, default=0, help="E14R Prompt-Consistency Inference (2610.04757)")
    ap.add_argument("--tv", default=None, choices=["symmetric", "stepup", "symmetric_fix", "stepup_fix"],
                    help="E14R TV-CFG schedule (2509.22007, sqt24/tvcfg)")
    ap.add_argument("--jrr", default=None, help="E14R JRR-CFG weights 'gT,gS,gI' (CosyVoice2 0.70,1.20,0.95)")
    ap.add_argument("--text_frontend", type=int, default=1, help="0 = raw text, no normalisation (E8'-5 rule, ko/ja/de)")
    a = ap.parse_args()
    meta = Path(a.meta) if os.path.isabs(a.meta) else ROOT / a.meta
    cache_dir = Path(a.cache_dir) if os.path.isabs(a.cache_dir) else ROOT / a.cache_dir
    out_dir = (Path(a.out_dir) if os.path.isabs(a.out_dir) else ROOT / a.out_dir) if a.out_dir else None
    rows = read_meta(meta)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    if a.stage == "tokens":
        stage_tokens(rows, cache_dir, bool(a.text_frontend))
    else:
        stage_decode(rows, cache_dir, out_dir, a)


if __name__ == "__main__":
    main()
