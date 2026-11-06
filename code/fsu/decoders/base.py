"""Common interface of the decoder runners.

Every runner is executed inside the decoder's own conda environment as `python -m fsu.decoders.<name> ...`. It has two
stages:
  prepare  everything upstream of the flow-matching decoder (text frontend, LLM / GPT, speaker features), seeded per
           utterance and cached to <cache_dir>/<utt>.pt, so every decoder configuration sees the same conditions
  decode   the flow-matching decoder (+ vocoder) from the cache with the sampler options below -> <out_dir>/<utt>.wav
Both stages skip utterances whose output already exists, so an interrupted run can simply be restarted.

Sampler options shared by the decoders (repository time convention: t = 0 is noise, t = 1 is data):
  --nfe N                 number of solver steps on the decoder's official grid
  --cfg w                 guidance weight in v = (1 + w) v_c - w v_u
  --noise utt:K           K-th per-utterance noise draw, seeded with seed_of(f"{utt}|noise|{K}")
  --t_span a,b,...        explicit step grid (overrides --nfe)
  --cfg_interval lo,hi    guidance only for steps that start in [lo, hi]; FSU = 0.1,1
  --cfg_steps w1,w2,...   explicit per-step weights
  --cfg_branch            what the unconditional branch removes (all = official)
  --solver                euler | midpoint | mid1 (midpoint on the first step only); FSU-Mid = --nfe 1 --solver midpoint
  external guidance rules --tv, --mambo_alpha, --branch_switch_t, --cfg_zero_star, --zero_init, --tfr, --pci, --jrr
"""
import argparse

import numpy as np

from fsu.common import read_list, resolve


def tv_scales(t_span, w, kind):
    """TV-CFG per-step guidance scales g_k (arXiv 2509.22007, transcribed from sqt24/tvcfg guidance_schedulers.py),
    with g = 1 + w. Timesteps are normalised by their maximum, sigma_i = 1 - t_i at each step start (sigma_0 = 1), and
    delta_i = sigma_i - sigma_{i+1} with a final 0.0 appended as in the official code.
    'symmetric' = symmetric(low=1, high=2g-1, normalize=True) as called by generate.py;
    'stepup' = stepup(low=g/3, high=g, tau=0.25, normalize=False);
    '*_fix' (corrected variant): the last delta is sigma_{N-1}. The official final 0.0 drops the last step from the
    normalisation, which inflates the scales at few steps."""
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


class Decoder:
    """Base class: subclasses set the stage names and argument defaults and implement prepare() / decode()."""
    prepare_stage = "tokens"
    defaults = dict(nfe=10, cfg=0.7, noise="utt:0")
    branches = ("all", "speaker", "token")
    solvers = ("euler", "midpoint", "mid1")

    def __init__(self, args):
        self.args = args

    @classmethod
    def parser(cls):
        ap = argparse.ArgumentParser(description=cls.__doc__)
        ap.add_argument("--meta", required=True, help="test list (utt|prompt_text|prompt_wav|text)")
        ap.add_argument("--stage", choices=[cls.prepare_stage, "decode"], required=True)
        ap.add_argument("--cache_dir", required=True)
        ap.add_argument("--out_dir")
        ap.add_argument("--nfe", type=int, default=cls.defaults["nfe"])
        ap.add_argument("--cfg", type=float, default=cls.defaults["cfg"])
        ap.add_argument("--noise", default=cls.defaults["noise"])
        ap.add_argument("--t_span", default=None)
        ap.add_argument("--cfg_interval", default=None)
        ap.add_argument("--cfg_steps", default=None)
        ap.add_argument("--cfg_branch", default="all", choices=list(cls.branches))
        ap.add_argument("--solver", default="euler", choices=list(cls.solvers))
        ap.add_argument("--branch_switch_t", type=float, default=None,
                        help="Selective-CFG (default): all-null branch while t < this (0.08), then --cfg_branch")
        ap.add_argument("--mambo_alpha", type=float, default=None, help="MAMBO-G: w_k = w exp(-alpha |v_c - v_u| / |v_u|)")
        ap.add_argument("--cfg_zero_star", type=int, default=0, help="CFG-Zero*: v_u scaled by <v_c,v_u>/|v_u|^2")
        ap.add_argument("--zero_init", type=int, default=0, help="CFG-Zero*: zero velocity for the first K steps")
        ap.add_argument("--tfr", type=int, default=0, help="Terminal-fitted repair")
        ap.add_argument("--tv", default=None, choices=["symmetric", "stepup", "symmetric_fix", "stepup_fix"],
                        help="TV-CFG schedule (symmetric_fix = corrected normalisation)")
        ap.add_argument("--jrr", default=None, help="JRR-CFG weights 'gT,gS,gI'")
        cls.add_arguments(ap)
        return ap

    @classmethod
    def add_arguments(cls, ap):
        """Decoder-specific arguments."""

    def prepare(self, rows, cache_dir, out_dir=None):
        raise NotImplementedError

    def decode(self, rows, cache_dir, out_dir):
        raise NotImplementedError

    @staticmethod
    def floats(s):
        return [float(x) for x in s.split(",")] if s else None

    @classmethod
    def main(cls, argv=None):
        a = cls.parser().parse_args(argv)
        rows = read_list(resolve(a.meta))
        cache_dir = resolve(a.cache_dir)
        out_dir = resolve(a.out_dir) if a.out_dir else None
        dec = cls(a)  # may change the working directory to the upstream repository
        if a.stage == "decode":
            dec.decode(rows, cache_dir, out_dir)
        else:
            dec.prepare(rows, cache_dir, out_dir)
