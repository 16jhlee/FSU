"""CosyVoice 2 / Fun-CosyVoice 3 runner (environment env_cosy; upstream repos/CosyVoice + patches/cosyvoice.diff).

prepare (--stage tokens): official frontend + LLM, seeded per utterance -> cached speech tokens and flow inputs.
decode: flow-matching decoder (+ HiFT vocoder) from the cached tokens. The mel length is twice the token count, so it is
identical for every sampler setting.
    python -m fsu.decoders.cosyvoice --model_dir ckpt/CosyVoice2-0.5B --meta data/lists/seedtts_en.lst \
        --stage decode --cache_dir cache/tokens/cosyvoice2/seedtts_en --out_dir out/... --nfe 2 --cfg_interval 0.1,1
"""
import os
import sys
import uuid as uuidlib
from pathlib import Path

import numpy as np
import torch
import torchaudio

from fsu.common import REPOS, noise_index, seed_of
from fsu.decoders.base import Decoder, tv_scales

REPO = REPOS / "CosyVoice"
set_all_random_seed = None  # cosyvoice.utils.common.set_all_random_seed, bound by load_upstream()


def load_upstream():
    """Import the CosyVoice package. wetext re-queries ModelScope on every start and silently falls back to no text
    normalisation when the request fails, so it is pinned to the already-downloaded resource directory."""
    global set_all_random_seed
    sys.path[:0] = [str(REPO), str(REPO / "third_party" / "Matcha-TTS")]
    import wetext.wetext
    d = None
    for b in [os.environ.get("MODELSCOPE_CACHE"), Path.home() / ".cache" / "modelscope" / "hub",
              Path.home() / ".cache" / "modelscope" / "hub" / "models"]:
        if b and (Path(b) / "pengzhendong" / "wetext" / "en" / "tn" / "tagger.fst").exists():
            d = Path(b) / "pengzhendong" / "wetext"
            break
    if d is None:
        d = Path(wetext.wetext.snapshot_download("pengzhendong/wetext"))  # first run on a new machine
        assert (d / "en" / "tn" / "tagger.fst").exists(), d
    wetext.wetext.snapshot_download = lambda *a, **k: str(d)
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed as seed_fn
    set_all_random_seed = seed_fn
    return AutoModel


class CosyVoice(Decoder):
    """CosyVoice 2 / 3 decoder runner."""
    defaults = dict(nfe=10, cfg=0.7, noise="default")
    solvers = ("euler", "midpoint", "heun", "mid1")

    @classmethod
    def add_arguments(cls, ap):
        ap.add_argument("--model_dir", required=True)
        ap.add_argument("--sampler", default="euler", choices=["euler", "a1"], help="a1 = on-path re-interpolation")
        ap.add_argument("--cfg_mode", default="standard", choices=["standard", "apg"])
        ap.add_argument("--restart_tr", type=float, default=None, help="re-noise the output to time t_r and take one more step")
        ap.add_argument("--cfg_ramp", type=float, default=None, help="w_k = w * min(1, t_k / t*) with t* given")
        ap.add_argument("--stream", type=int, default=0, help="official chunk-aware streaming decode (CosyVoice 2)")
        ap.add_argument("--stream_hop", type=int, default=0, help="fixed token hop per chunk (0 = official 25->50->100)")
        ap.add_argument("--pci", type=int, default=0, help="Prompt-consistency inference")
        ap.add_argument("--jrr_steps", type=int, default=0, help="JRR-CFG on the first K steps only (0 = all)")
        ap.add_argument("--prompt_prefix", default="", help="CosyVoice 3: 'You are a helpful assistant.<|endofprompt|>'")
        ap.add_argument("--text_frontend", type=int, default=1,
                        help="0 = raw text (official text_frontend=False path); the zh/en normalisers misfire on ko/ja/de")

    def __init__(self, args):
        super().__init__(args)
        self.cv = load_upstream()(model_dir=args.model_dir)
        # ttsfrd is Linux-only and wetext is the official fallback; refuse the silent '' fallback
        assert self.cv.frontend.text_frontend in ("ttsfrd", "wetext"), f"text frontend = {self.cv.frontend.text_frontend!r}"
        print("text_frontend:", self.cv.frontend.text_frontend, flush=True)

    # ------------------------------------------------------------------ prepare
    def prepare(self, rows, cache_dir, out_dir=None):
        fe, m = self.cv.frontend, self.cv.model
        prefix, text_frontend = self.args.prompt_prefix, bool(self.args.text_frontend)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for utt, ptext, pwav, text in rows:
            out = cache_dir / f"{utt}.pt"
            if out.exists():
                continue
            set_all_random_seed(seed_of(utt))
            ptext_n = fe.text_normalize(prefix + ptext, split=False, text_frontend=text_frontend)
            segs = []
            for seg in fe.text_normalize(text, split=True, text_frontend=text_frontend):
                mi = fe.frontend_zero_shot(seg, ptext_n, pwav, self.cv.sample_rate, "")
                u = str(uuidlib.uuid1())
                m.tts_speech_token_dict[u], m.llm_end_dict[u] = [], False
                m.llm_job(mi["text"], mi["prompt_text"], mi["llm_prompt_speech_token"], mi["llm_embedding"], u)
                tok = torch.tensor(m.tts_speech_token_dict.pop(u)).unsqueeze(0)
                m.llm_end_dict.pop(u)
                segs.append(dict(text=seg, token=tok.cpu(), prompt_token=mi["flow_prompt_speech_token"].cpu(),
                                 prompt_feat=mi["prompt_speech_feat"].cpu(), embedding=mi["flow_embedding"].cpu()))
            torch.save(segs, out)
            print(utt, [s["token"].shape[1] for s in segs], flush=True)

    # ------------------------------------------------------------------ decode
    def configure(self, dec):
        """Set the sampler options on the patched flow decoder; returns the number of solver steps."""
        a = self.args
        if a.solver != "euler":
            self.install_solver(dec, a.solver)
        dec.inference_cfg_rate = a.cfg
        dec.sampler_mode = a.sampler
        dec.cfg_mode = a.cfg_mode
        dec.cfg_interval = self.floats(a.cfg_interval)
        dec.cfg_ramp = a.cfg_ramp
        dec.cfg_branch = a.cfg_branch
        dec.cfg_branch_switch_t = a.branch_switch_t
        dec.mambo_alpha = a.mambo_alpha
        dec.cfg_zero_star = bool(a.cfg_zero_star)
        dec.zero_init_steps = a.zero_init
        nfe = a.nfe
        if a.t_span:
            dec.t_span_override = self.floats(a.t_span)
            nfe = len(dec.t_span_override) - 1
        else:  # official cosine schedule with N steps, computed exactly as in flow_matching.py
            t = torch.linspace(0, 1, nfe + 1)
            dec.t_span_override = (1 - torch.cos(t * 0.5 * torch.pi)).tolist()
        dec.tfr = bool(a.tfr)
        dec.pci = bool(a.pci)
        dec.tv_scales = tv_scales(dec.t_span_override, a.cfg, a.tv) if a.tv else None
        if a.cfg_steps:  # explicit per-step w_k through the same per-step scale hook (g_k = 1 + w_k)
            ws = self.floats(a.cfg_steps)
            assert len(ws) == nfe, (ws, nfe)
            dec.tv_scales = [1.0 + x for x in ws]
        dec.jrr = tuple(self.floats(a.jrr)) if a.jrr else None
        dec.jrr_steps = a.jrr_steps if a.jrr_steps else 10 ** 6
        if dec.tv_scales:
            print("TV-CFG scales g_k:", [round(x, 3) for x in dec.tv_scales], flush=True)
        return nfe

    def decode(self, rows, cache_dir, out_dir):
        a, m = self.args, self.cv.model
        dec = m.flow.decoder
        official_noise = dec.rand_noise.clone()
        nfe = self.configure(dec)
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
                g = torch.Generator().manual_seed(seed_of(f"{utt}|noise|{noise_index(a.noise)}"))
                dec.rand_noise = torch.randn([1, 80, 50 * 300], generator=g)
            speech = []
            for i, s in enumerate(segs):
                if a.restart_tr is not None:
                    speech.append(self.restart_segment(dec, s, utt, i))
                elif a.stream:
                    set_all_random_seed(seed_of(f"{utt}|hift|{i}"))
                    sp, ends = self.stream_segment(s, str(uuidlib.uuid1()), a.stream_hop)
                    base = sum(x.shape[1] for x in speech)
                    with open(out_dir / "_chunks.tsv", "a", encoding="utf-8") as cf:
                        cf.write(f"{utt}\t{i}\t" + ",".join(str(base + e) for e in ends) + "\n")
                    speech.append(sp)
                else:
                    dec.prompt_frames = int(s["prompt_feat"].shape[1])  # PCI: prompt block = first prompt frames
                    speech.append(self.offline_segment(s, utt, i))
            wav = torch.cat(speech, dim=1)
            torchaudio.save(str(wav_path), wav, self.cv.sample_rate)
            ntok = sum(s["token"].shape[1] for s in segs)
            log.write(f"{utt}\t{nfe}\t{a.cfg}\t{a.noise}\t{ntok}\t{wav.shape[1]}\n")
            log.flush()
        log.close()

    def offline_segment(self, s, utt, i):
        m = self.cv.model
        u = str(uuidlib.uuid1())
        m.hift_cache_dict[u] = None
        set_all_random_seed(seed_of(f"{utt}|hift|{i}"))  # the CosyVoice 2 HiFT vocoder is stochastic
        with torch.inference_mode():
            sp = m.token2wav(token=s["token"], prompt_token=s["prompt_token"], prompt_feat=s["prompt_feat"],
                             embedding=s["embedding"], token_offset=0, uuid=u, finalize=True)
        m.hift_cache_dict.pop(u)
        return sp.cpu()

    def stream_segment(self, s, uid, fixed_hop=0):
        """Official CosyVoice 2 streaming token2wav loop (CosyVoice2Model.tts, stream=True) on cached tokens: token hop 25,
        doubled per chunk up to 100, first chunk padded to the prompt-token grid, pre-lookahead tokens, chunk-aware causal
        flow, HiFT cache and fade. fixed_hop H > 0 keeps every chunk at H tokens. Returns (speech, chunk end samples)."""
        m = self.cv.model
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

    def restart_segment(self, dec, s, utt, i):
        """Run the configured sampler, re-noise its full-length output to time t_r with fresh noise
        seed_of(f"{utt}|restart|{k}|{i}"), take one more step of the same sampler from t_r to 1, then vocode."""
        m, a = self.cv.model, self.args
        k = noise_index(a.noise)
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
        g = torch.Generator().manual_seed(seed_of(f"{utt}|restart|{k}|{i}"))
        noise = torch.randn(x_full.shape, generator=g).to(x_full.device, x_full.dtype)
        tr = a.restart_tr
        z = tr * x_full + (1 - tr) * noise
        with torch.inference_mode():
            x = dec.solve_euler(z, t_span=torch.tensor([tr, 1.0], device=z.device, dtype=z.dtype), mu=args["mu"],
                                mask=args["mask"], spks=args["spks"], cond=args["cond"], streaming=args["streaming"])
            mel = x[:, :, s["prompt_feat"].shape[1]:].float()
            set_all_random_seed(seed_of(f"{utt}|hift|{i}"))
            sp, _ = m.hift.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
        return sp.cpu()

    @staticmethod
    def install_solver(dec, kind):
        """Second-order solvers built from single Euler-step calls of the patched solver, so every network evaluation
        goes through the same guidance code. t_span is the step grid and each step costs 2 evaluations.
        midpoint: v1@(x,t), v2@(x+dt/2 v1, t+dt/2), x += dt v2.  heun: v1@(x,t), v2@(x+dt v1, t+dt), x += dt/2 (v1+v2).
        mid1: midpoint on the first step, Euler afterwards."""
        orig = dec.solve_euler

        def vel(x, t, kw):  # one Euler step of length 1 from time t returns x + v(x, t)
            y = orig(x, t_span=torch.tensor([t, t + 1.0], device=x.device, dtype=x.dtype), **kw)
            return y.to(x.dtype) - x

        def solve(x, t_span, mu, mask, spks, cond, streaming=False):
            kw = dict(mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming)
            ts = [float(v) for v in t_span]
            for i, (t0, t1) in enumerate(zip(ts[:-1], ts[1:])):
                d = t1 - t0
                v1 = vel(x, t0, kw)
                if kind == "mid1" and i > 0:
                    x = x + d * v1
                elif kind in ("midpoint", "mid1"):
                    v2 = vel(x + 0.5 * d * v1, t0 + 0.5 * d, kw)
                    x = x + d * v2
                else:
                    v2 = vel(x + d * v1, t1, kw)
                    x = x + 0.5 * d * (v1 + v2)
            return x.float()
        dec.solve_euler = solve


if __name__ == "__main__":
    CosyVoice.main()
