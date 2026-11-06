"""CosyVoice2/3 runner with LLM-token caching (family A).

Stage `tokens`: official frontend + LLM (seeded per utterance) -> cache tokens & flow inputs.
Stage `decode`: flow (+ HiFT) only from cached tokens, with sampler controls:
    --nfe N            Euler steps (official 10)
    --cfg w            inference_cfg_rate (official 0.7)
    --noise default|utt|utt:K   default = official fixed rand_noise buffer;
                       utt = per-utterance noise seeded by hash(utt_id); utt:K = K-th noise seed
    --t_span a,b,...   explicit schedule in repo time (0 = noise, 1 = data); overrides --nfe
Mel length = 2 x token count, so it is identical across all decode settings by construction.
"""
import argparse
import hashlib
import sys
import uuid as uuidlib
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "CosyVoice"
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "Matcha-TTS")]
# wetext re-queries modelscope on every start and silently falls back to no text normalisation
# when the request fails; pin it to the already-downloaded resource directory instead.
import wetext.wetext  # noqa: E402


def _wetext_dir():
    import os
    bases = [os.environ.get("MODELSCOPE_CACHE"), Path.home() / ".cache" / "modelscope" / "hub",
             Path.home() / ".cache" / "modelscope" / "hub" / "models"]
    for b in bases:
        if b and (Path(b) / "pengzhendong" / "wetext" / "en" / "tn" / "tagger.fst").exists():
            return Path(b) / "pengzhendong" / "wetext"
    d = Path(wetext.wetext.snapshot_download("pengzhendong/wetext"))  # first run on a new machine
    assert (d / "en" / "tn" / "tagger.fst").exists(), d
    return d


WETEXT_DIR = _wetext_dir()
wetext.wetext.snapshot_download = lambda *a, **k: str(WETEXT_DIR)
from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402


def h(s, mod=2**31 - 1):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16) % mod


def read_meta(p):
    out = []
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        if l.strip():
            f = l.split("|")
            out.append((f[0], f[1], f[2], f[3]))
    return out


def stage_tokens(cv, rows, cache_dir, prompt_prefix="", text_frontend=True):
    fe, m = cv.frontend, cv.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    for utt, ptext, pwav, text in rows:
        out = cache_dir / f"{utt}.pt"
        if out.exists():
            continue
        set_all_random_seed(h(utt))
        ptext_n = fe.text_normalize(prompt_prefix + ptext, split=False, text_frontend=text_frontend)
        segs = []
        for seg in fe.text_normalize(text, split=True, text_frontend=text_frontend):
            mi = fe.frontend_zero_shot(seg, ptext_n, pwav, cv.sample_rate, "")
            u = str(uuidlib.uuid1())
            m.tts_speech_token_dict[u], m.llm_end_dict[u] = [], False
            m.llm_job(mi["text"], mi["prompt_text"], mi["llm_prompt_speech_token"], mi["llm_embedding"], u)
            tok = torch.tensor(m.tts_speech_token_dict.pop(u)).unsqueeze(0)
            m.llm_end_dict.pop(u)
            segs.append(dict(text=seg, token=tok.cpu(),
                             prompt_token=mi["flow_prompt_speech_token"].cpu(),
                             prompt_feat=mi["prompt_speech_feat"].cpu(),
                             embedding=mi["flow_embedding"].cpu()))
        torch.save(segs, out)
        print(utt, [s["token"].shape[1] for s in segs], flush=True)


def restart_decode(m, dec, s, utt, i, a):
    """A4 (E5): run the configured sampler, re-noise its full-length output to time t_r with fresh noise
    hash(utt|restart|k), take one more step of the same sampler from t_r to 1, then vocode."""
    k = a.noise.split(":")[1] if ":" in a.noise else "0"
    kw = dict(token=s["token"].to(m.device, dtype=torch.int32),
              token_len=torch.tensor([s["token"].shape[1]], dtype=torch.int32).to(m.device),
              prompt_token=s["prompt_token"].to(m.device),
              prompt_token_len=torch.tensor([s["prompt_token"].shape[1]], dtype=torch.int32).to(m.device),
              prompt_feat=s["prompt_feat"].to(m.device),
              prompt_feat_len=torch.tensor([s["prompt_feat"].shape[1]], dtype=torch.int32).to(m.device),
              embedding=s["embedding"].to(m.device), streaming=False, finalize=True)
    dec.dump_states = []
    with torch.inference_mode():
        m.flow.inference(**kw)
    st, args = dec.dump_states, dec.dump_args
    dec.dump_states = None
    x_full = (st[-1]["x"] + (1 - st[-1]["t"]) * st[-1]["v"]).to(args["mu"].device, args["mu"].dtype)
    g = torch.Generator().manual_seed(h(f"{utt}|restart|{k}|{i}"))
    noise = torch.randn(x_full.shape, generator=g).to(x_full.device, x_full.dtype)
    tr = a.restart_tr
    z = tr * x_full + (1 - tr) * noise
    with torch.inference_mode():
        x = dec.solve_euler(z, t_span=torch.tensor([tr, 1.0], device=z.device, dtype=z.dtype), mu=args["mu"],
                            mask=args["mask"], spks=args["spks"], cond=args["cond"], streaming=args["streaming"])
        mel = x[:, :, s["prompt_feat"].shape[1]:].float()
        set_all_random_seed(h(f"{utt}|hift|{i}"))
        sp, _ = m.hift.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
    return sp.cpu()


def stream_decode(m, s, uid, fixed_hop=0):
    """v8 §2.2: official CosyVoice2 streaming token2wav loop (CosyVoice2Model.tts, stream=True) on cached tokens:
    token_hop 25 -> x2 per chunk (max 100), first chunk padded to the prompt-token grid, pre-lookahead tokens, chunk-aware
    causal flow (streaming=True), HiFT cache + fade. Returns (speech, chunk_end_samples).
    E12-5: fixed_hop H > 0 keeps every chunk at H tokens (no doubling); the first-chunk pad stays on the official 25 grid."""
    tokens = s["token"]
    hop, max_hop = (fixed_hop, fixed_hop) if fixed_hop else (25, 100)
    pad = int(np.ceil(s["prompt_token"].shape[1] / 25) * 25 - s["prompt_token"].shape[1])
    look = m.flow.pre_lookahead_len
    offset, out, ends = 0, [], []
    m.hift_cache_dict[uid] = None
    with torch.inference_mode():
        while True:
            this_hop = hop + pad if offset == 0 else hop
            if tokens.shape[1] - offset < this_hop + look:
                break
            sp = m.token2wav(token=tokens[:, :offset + this_hop + look], prompt_token=s["prompt_token"],
                             prompt_feat=s["prompt_feat"], embedding=s["embedding"], token_offset=offset, uuid=uid,
                             stream=True, finalize=False)
            out.append(sp.cpu())
            ends.append(sum(x.shape[1] for x in out))
            offset += this_hop
            hop = min(max_hop, hop * 2)
        sp = m.token2wav(token=tokens, prompt_token=s["prompt_token"], prompt_feat=s["prompt_feat"],
                         embedding=s["embedding"], token_offset=offset, uuid=uid, finalize=True)
        out.append(sp.cpu())
    m.hift_cache_dict.pop(uid)
    return torch.cat(out, dim=1), ends


def install_solver(dec, kind):
    """E8'-4: 2nd-order solvers built from single Euler-step calls of the (patched) original solver, so every
    network evaluation still goes through the same CFG / interval / branch code. t_span = step grid; each step
    costs 2 evaluations. midpoint: v1@(x,t), v2@(x+dt/2 v1, t+dt/2), x+=dt v2.
    heun: v1@(x,t), xp=x+dt v1, v2@(xp,t+dt), x+=dt/2 (v1+v2)."""
    orig = dec.solve_euler

    def vel(x, t, kw):  # one Euler step of length 1 from time t returns x + v(x,t)
        y = orig(x, t_span=torch.tensor([t, t + 1.0], device=x.device, dtype=x.dtype), **kw)
        return (y.to(x.dtype) - x)

    def solve(x, t_span, mu, mask, spks, cond, streaming=False):
        kw = dict(mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming)
        ts = [float(v) for v in t_span]
        for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:])):
            d = t1 - t0
            v1 = vel(x, t0, kw)
            if kind == "mid1" and i > 0:  # E8'-9b: midpoint for the first step only, Euler afterwards
                x = x + d * v1
            elif kind in ("midpoint", "mid1"):
                v2 = vel(x + 0.5 * d * v1, t0 + 0.5 * d, kw)
                x = x + d * v2
            else:
                v2 = vel(x + d * v1, t1, kw)
                x = x + 0.5 * d * (v1 + v2)
        return x.float()
    dec.solve_euler = solve


def tv_scales(t_span, w, kind):
    """E8'-10 TV-CFG (2509.22007), sqt24/tvcfg guidance_schedulers.py transcribed. Guidance scale g = 1 + w.
    timesteps normalised by their max = sigma_i = 1 - t_i at each step start (sigma_0 = 1); delta_i = sigma_i - sigma_{i+1}
    with a final 0.0 appended (official). 'symmetric' = symmetric(low=1, high=2g-1, normalize=True) (generate.py call);
    'stepup' = stepup(low=g/3, high=g, tau=0.25, normalize=False). '*_fix' (our variant, appendix): last delta = sigma_{N-1}
    (the official final 0.0 drops the last step from the normalisation, which inflates few-step scales)."""
    g = 1.0 + w
    sig = np.array([1.0 - t for t in t_span[:-1]])
    n = len(sig)
    delta = np.append(sig[:-1] - sig[1:], sig[-1] if kind.endswith("_fix") else 0.0)
    base = kind.replace("_fix", "")
    if base == "symmetric":
        low, high, half = 1.0, 2 * g - 1.0, n // 2
        if n % 2 == 0:
            s = np.concatenate([np.linspace(low, high, half, endpoint=False), np.linspace(high, low, half, endpoint=True)])
        else:
            s = np.concatenate([np.linspace(low, high, half + 1, endpoint=True), np.linspace(high, low, half + 1, endpoint=True)[1:]])
        return (s * g / np.sum(s * delta)).tolist()
    if base == "stepup":
        steps = np.linspace(0, 1, n)
        return np.where(steps < 0.25, g / 3, g).tolist()
    raise ValueError(kind)


def stage_decode(cv, rows, cache_dir, out_dir, a):
    m = cv.model
    dec = m.flow.decoder
    if a.solver != "euler":
        install_solver(dec, a.solver)
    official_noise = dec.rand_noise.clone()
    dec.inference_cfg_rate = a.cfg
    # E5 sampler options (patches/CosyVoice_E5.diff); defaults reproduce the official sampler
    dec.sampler_mode = a.sampler
    dec.cfg_mode = a.cfg_mode
    dec.cfg_interval = [float(x) for x in a.cfg_interval.split(",")] if a.cfg_interval else None
    dec.cfg_ramp = a.cfg_ramp  # E7 (patches/CosyVoice_E7.diff)
    dec.cfg_branch = a.cfg_branch  # E8'-2 (patches/CosyVoice_branch.diff)
    # E8'-7 external baselines (patches/CosyVoice_E8p7.diff; formulas and sources in reports/E8p7_sources.md)
    dec.cfg_branch_switch_t = a.branch_switch_t  # Selective-CFG def_text: all-null while t < this, then cfg_branch
    dec.mambo_alpha = a.mambo_alpha  # MAMBO-G: w_k = w exp(-alpha ||v_c - v_u|| / ||v_u||)
    dec.cfg_zero_star = bool(a.cfg_zero_star)  # CFG-Zero*: v_u scaled by s* = <v_c,v_u>/||v_u||^2
    dec.zero_init_steps = a.zero_init  # CFG-Zero*: velocity 0 for the first K steps
    nfe = a.nfe
    if a.t_span:
        dec.t_span_override = [float(x) for x in a.t_span.split(",")]
        nfe = len(dec.t_span_override) - 1
    else:
        # official cosine schedule with N steps, computed exactly as in flow_matching.py
        t = torch.linspace(0, 1, nfe + 1)
        dec.t_span_override = (1 - torch.cos(t * 0.5 * torch.pi)).tolist()
    # E8'-10 2026 baselines (patches/CosyVoice_E8p10.diff; formulas in reports/E8p10_baselines2026.md)
    dec.tfr = bool(a.tfr)  # Terminal-Fitted Repair
    dec.pci = bool(a.pci)  # Prompt-Consistency Inference
    dec.tv_scales = tv_scales(dec.t_span_override, a.cfg, a.tv) if a.tv else None
    if a.cfg_steps:  # E10-3: explicit per-step w_k through the same per-step scale hook (g_k = 1 + w_k)
        ws = [float(x) for x in a.cfg_steps.split(",")]
        assert len(ws) == nfe, (ws, nfe)
        dec.tv_scales = [1.0 + x for x in ws]
    dec.jrr = tuple(float(x) for x in a.jrr.split(",")) if a.jrr else None
    dec.jrr_steps = a.jrr_steps if a.jrr_steps else 10 ** 6
    if dec.tv_scales:
        print("TV-CFG scales g_k:", [round(x, 3) for x in dec.tv_scales], flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "_lengths.tsv", "a", encoding="utf-8")
    for utt, *_ in rows:
        wav_path = out_dir / f"{utt}.wav"
        if wav_path.exists():
            continue
        segs = torch.load(cache_dir / f"{utt}.pt")
        if a.noise == "default":
            dec.rand_noise = official_noise
        else:
            k = a.noise.split(":")[1] if ":" in a.noise else "0"
            g = torch.Generator().manual_seed(h(f"{utt}|noise|{k}"))
            dec.rand_noise = torch.randn([1, 80, 50 * 300], generator=g)
        speech = []
        for i, s in enumerate(segs):
            if a.restart_tr is not None:
                speech.append(restart_decode(m, dec, s, utt, i, a))
                continue
            if a.stream:
                set_all_random_seed(h(f"{utt}|hift|{i}"))
                sp, ends = stream_decode(m, s, str(uuidlib.uuid1()), a.stream_hop)
                base = sum(x.shape[1] for x in speech)
                with open(out_dir / "_chunks.tsv", "a", encoding="utf-8") as cf:
                    cf.write(f"{utt}\t{i}\t" + ",".join(str(base + e) for e in ends) + "\n")
                speech.append(sp)
                continue
            dec.prompt_frames = int(s["prompt_feat"].shape[1])  # E8'-10 PCI: prompt block = first mel_len1 frames
            u = str(uuidlib.uuid1())
            m.hift_cache_dict[u] = None
            set_all_random_seed(h(f"{utt}|hift|{i}"))  # CV2 HiFT is stochastic
            with torch.inference_mode():
                sp = m.token2wav(token=s["token"], prompt_token=s["prompt_token"], prompt_feat=s["prompt_feat"],
                                 embedding=s["embedding"], token_offset=0, uuid=u, finalize=True)
            m.hift_cache_dict.pop(u)
            speech.append(sp.cpu())
        wav = torch.cat(speech, dim=1)
        torchaudio.save(str(wav_path), wav, cv.sample_rate)
        ntok = sum(s["token"].shape[1] for s in segs)
        log.write(f"{utt}\t{nfe}\t{a.cfg}\t{a.noise}\t{ntok}\t{wav.shape[1]}\n")
        log.flush()
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stage", choices=["tokens", "decode"], required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir")
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--noise", default="default")
    ap.add_argument("--t_span", default=None)
    ap.add_argument("--sampler", default="euler", choices=["euler", "a1"], help="E5 A1 = on-path re-interpolation")
    ap.add_argument("--cfg_mode", default="standard", choices=["standard", "apg"], help="E5 A2-b")
    ap.add_argument("--cfg_interval", default=None, help="E5 A2-c: 'tmin,tmax' (repo time, t=0 noise)")
    ap.add_argument("--restart_tr", type=float, default=None, help="E5 A4: re-noise time t_r (repo time)")
    ap.add_argument("--cfg_ramp", type=float, default=None, help="E7: w_k = w * min(1, t_k / t*) with t* given")
    ap.add_argument("--solver", default="euler", choices=["euler", "midpoint", "heun", "mid1"],
                    help="E8'-4: --nfe/--t_span give the STEP grid; midpoint/heun use 2 evaluations per step")
    ap.add_argument("--stream", type=int, default=0, help="v8 §2.2: official chunk-aware streaming decode (CosyVoice2)")
    ap.add_argument("--stream_hop", type=int, default=0, help="E12-5: fixed token hop per chunk (0 = official 25->50->100)")
    ap.add_argument("--branch_switch_t", type=float, default=None, help="E8'-7 Sel-CFG def_text threshold (0.08)")
    ap.add_argument("--mambo_alpha", type=float, default=None, help="E8'-7 MAMBO-G alpha")
    ap.add_argument("--cfg_zero_star", type=int, default=0, help="E8'-7 CFG-Zero* optimised scale s*")
    ap.add_argument("--zero_init", type=int, default=0, help="E8'-7 CFG-Zero* zero-init: first K steps velocity 0")
    ap.add_argument("--cfg_branch", default="all", choices=["all", "speaker", "token"],
                    help="E8'-2: what the unconditional branch erases (all = official)")
    ap.add_argument("--tfr", type=int, default=0, help="E8'-10 Terminal-Fitted Repair (2607.07665)")
    ap.add_argument("--pci", type=int, default=0, help="E8'-10 Prompt-Consistency Inference (2610.04757)")
    ap.add_argument("--tv", default=None, choices=["symmetric", "stepup", "symmetric_fix", "stepup_fix"],
                    help="E8'-10 TV-CFG schedule (2509.22007, sqt24/tvcfg)")
    ap.add_argument("--jrr", default=None, help="E8'-10 JRR-CFG weights 'gT,gS,gI' (2606.25672; CosyVoice2 0.70,1.20,0.95)")
    ap.add_argument("--cfg_steps", default=None, help="E10-3: per-step w 'w1,w2,...' (one per step; overrides --cfg)")
    ap.add_argument("--jrr_steps", type=int, default=0, help="E8'-10 JRR sparse: JRR on the first K steps only (0 = all)")
    ap.add_argument("--prompt_prefix", default="",
                    help="CosyVoice3 official: 'You are a helpful assistant.<|endofprompt|>'")
    ap.add_argument("--text_frontend", type=int, default=1,
                    help="E8'-5: 0 = raw text (official text_frontend=False path). The zh/en normalisers misfire on "
                         "ko/ja/de (wetext-en crashes on Korean numerals with counters, ja kanji -> zh TN, de digits -> English words)")
    a = ap.parse_args()
    cv = AutoModel(model_dir=a.model_dir)
    # ttsfrd is Linux-only; wetext is the official fallback. Refuse the silent '' fallback.
    assert cv.frontend.text_frontend in ("ttsfrd", "wetext"), f"text frontend = {cv.frontend.text_frontend!r}"
    print("text_frontend:", cv.frontend.text_frontend, flush=True)
    rows = read_meta(a.meta)
    if a.stage == "tokens":
        stage_tokens(cv, rows, Path(a.cache_dir), a.prompt_prefix, bool(a.text_frontend))
    else:
        stage_decode(cv, rows, Path(a.cache_dir), Path(a.out_dir), a)


if __name__ == "__main__":
    main()
