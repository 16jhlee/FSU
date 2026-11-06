"""IndexTTS-2 runner (environment env_index; upstream repos/index-tts + patches/indextts.diff).

prepare (--stage cond): official IndexTTS2.infer (GPT sampling seeded per utterance; emotion = speaker prompt, the
default). It writes the official output (25 steps, w = 0.7) to --out_dir and caches, per text segment, the exact inputs
of the s2mel CFM call (content condition, lengths, reference mel, style), so the content condition is fixed for every
later decode.
decode: s2mel CFM from the cache (official grid linspace(0, 1, N+1)), per-utterance noise seed_of(f"{utt}|noise|{k}|{i}")
per segment, then BigVGAN; segments are joined with the official 200 ms silence and fade-out.
"""
import os
import random
import sys

import numpy as np
import torch
import torchaudio

from fsu.common import CKPT, REPOS, noise_index, seed_of
from fsu.decoders.base import Decoder, tv_scales

REPO = REPOS / "index-tts"
sys.path.insert(0, str(REPO))


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class IndexTTS2(Decoder):
    """IndexTTS-2 s2mel decoder runner."""
    prepare_stage = "cond"
    defaults = dict(nfe=25, cfg=0.7, noise="utt:0")

    @classmethod
    def add_arguments(cls, ap):
        ap.add_argument("--cfg_ramp", type=float, default=None)

    def __init__(self, args):
        super().__init__(args)
        os.chdir(REPO)
        from indextts.infer_v2 import IndexTTS2 as Model
        md = str(CKPT / "IndexTTS-2")
        self.tts = Model(cfg_path=os.path.join(md, "config.yaml"), model_dir=md, use_fp16=False,
                         use_cuda_kernel=False, use_deepspeed=False, use_qwen_emo=False)

    def prepare(self, rows, cache_dir, out_dir=None):
        cfm = self.tts.s2mel.models["cfm"]
        orig = cfm.inference
        captured = []

        def capture(mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
            captured.append(dict(mu=mu.detach().cpu(), x_lens=x_lens.detach().cpu(), prompt=prompt.detach().cpu(),
                                 style=style.detach().cpu(), n_timesteps=n_timesteps, cfg=inference_cfg_rate))
            return orig(mu, x_lens, prompt, style, f0, n_timesteps, temperature=temperature,
                        inference_cfg_rate=inference_cfg_rate)
        cfm.inference = capture
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        for utt, ptext, pwav, text in rows:
            if (cache_dir / f"{utt}.pt").exists() and (out_dir / f"{utt}.wav").exists():
                continue
            captured.clear()
            seed_all(seed_of(utt))
            self.tts.infer(spk_audio_prompt=pwav, text=text, output_path=str(out_dir / f"{utt}.wav"), verbose=False)
            torch.save(captured.copy(), cache_dir / f"{utt}.pt")
            print(utt, len(captured), [c["mu"].shape[1] for c in captured], flush=True)
        cfm.inference = orig

    @staticmethod
    def install_schedule_solver(cfm, t_span=None, solver="euler"):
        """Explicit step grid and/or midpoint solver built from single Euler-step calls of the patched solve_euler, so the
        guidance code is shared. midpoint: v1@(x,t), v2@(x+dt/2 v1, t+dt/2); mid1: midpoint on the first step only."""
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

            def vel(z, t):  # one Euler step of length 1 from time t: v(z, t) with the prompt region zeroed
                y = orig_solve(z.clone(), x_lens, prompt, mu, style, f0, torch.tensor([t, t + 1.0], device=z.device), cfg)
                v = y - z
                v[..., :P] = 0
                return v
            tl = [float(v) for v in ts]
            for i, (t0, t1) in enumerate(zip(tl[:-1], tl[1:])):
                d = t1 - t0
                v1 = vel(x, t0)
                if solver == "mid1" and i > 0:
                    x = x + d * v1
                else:
                    x = x + d * vel(x + 0.5 * d * v1, t0 + 0.5 * d)
                x[..., :P] = 0
            return x
        cfm.solve_euler = solve

    def configure(self, cfm):
        a = self.args
        cfm.cfg_interval = self.floats(a.cfg_interval)
        cfm.cfg_ramp = a.cfg_ramp
        cfm.cfg_branch = a.cfg_branch
        cfm.cfg_steps = self.floats(a.cfg_steps)
        if cfm.cfg_steps is not None:
            assert len(cfm.cfg_steps) == a.nfe, (cfm.cfg_steps, a.nfe)
        if a.t_span or a.solver != "euler":
            self.install_schedule_solver(cfm, a.t_span, a.solver)
        cfm.cfg_branch_switch_t = a.branch_switch_t
        cfm.mambo_alpha = a.mambo_alpha
        cfm.cfg_zero_star = bool(a.cfg_zero_star)
        cfm.zero_init_steps = a.zero_init
        cfm.tfr = bool(a.tfr)
        cfm.jrr = tuple(self.floats(a.jrr)) if a.jrr else None
        ts = self.floats(a.t_span) if a.t_span else torch.linspace(0, 1, a.nfe + 1).tolist()  # official grid
        cfm.tv_scales = tv_scales(ts, a.cfg, a.tv) if a.tv else None
        if cfm.tv_scales:
            print("TV-CFG scales g_k:", [round(x, 3) for x in cfm.tv_scales], flush=True)

    def decode(self, rows, cache_dir, out_dir):
        from indextts.utils.common import fade_out_tail
        a, tts = self.args, self.tts
        cfm = tts.s2mel.models["cfm"]
        self.configure(cfm)
        k = noise_index(a.noise)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "_lengths.tsv", "a", encoding="utf-8")
        for utt, *_ in rows:
            if (out_dir / f"{utt}.wav").exists():
                continue
            segs = torch.load(cache_dir / f"{utt}.pt")
            wavs = []
            for i, s in enumerate(segs):
                seed_all(seed_of(f"{utt}|noise|{k}|{i}"))
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
            torchaudio.save(str(out_dir / f"{utt}.wav"), (wav / 32767.0).float(), 22050)
            log.write(f"{utt}\t{a.nfe}\t{a.cfg}\t{a.cfg_interval}\t{a.cfg_ramp}\t{a.noise}\t{a.cfg_branch}\t{a.t_span}\t"
                      f"{a.solver}\t{sum(int(s['mu'].shape[1]) for s in segs)}\t{wav.shape[-1]}\n")
            log.flush()
        log.close()


if __name__ == "__main__":
    IndexTTS2.main()
