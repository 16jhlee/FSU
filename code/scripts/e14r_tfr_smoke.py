"""E14R-2 TFR alias check at the decoder output (v10.7 §2: mel max |difference| < 1e-4), en 3 utterances, seed 0, N = 2,
in one process: TFR (w = 0.7) vs w = 0. IndexTTS-2: cfm.inference mel (generated region). GLM-TTS (all-null): final
flow state x_N from the dumped last evaluation (x + dt v). Also the same w = 0 run twice (process noise floor) and, for
IndexTTS-2, the w = 0 velocity through the batched (CFG) estimator call (batch-size kernel effect).
python scripts/e14r_tfr_smoke.py idx|glm   (env_index | env_glm) -> logs/e14r_tfr_smoke_<model>.txt
"""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
UTTS = [l.split("|")[0] for l in (ROOT / "data/lists/en_3.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()]


def idx():
    from run_index import REPO, h, load_model, seed_all
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    dev = tts.device
    cfm.cfg_interval = cfm.cfg_ramp = cfm.cfg_steps = None
    cfm.cfg_branch = "all"
    out = []
    for utt in UTTS:
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]

        def run(w, tfr):
            cfm.tfr = tfr
            seed_all(h(f"{utt}|noise|0|0"))
            with torch.inference_mode():
                m = cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev), None, 2,
                                  inference_cfg_rate=w)
            return m[:, :, s["prompt"].size(-1):].double().cpu()
        a, b, b2 = run(0.7, True), run(0.0, False), run(0.0, False)
        out.append(f"{utt} TFR vs w0 max|d| {float((a - b).abs().max()):.3e} (rel {float((a - b).abs().max() / b.abs().max()):.2e})"
                   f" | w0 twice {float((b - b2).abs().max()):.3e}")
        print(out[-1], flush=True)
    return out


def glm():
    from run_glm import REPO, h
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from utils import seed_util, tts_model_util, yaml_util
    flow = yaml_util.load_flow_model(os.path.join("ckpt", "flow", "flow.pt"), os.path.join("ckpt", "flow", "config.yaml"), "cuda")
    t2w = tts_model_util.Token2Wav(flow, sample_rate=24000, device="cuda")
    flow.cfg_branch, flow.cfg_interval, flow.cfg_steps, flow.solver, flow.t_span_override = "allnull", None, None, "euler", None
    out = []
    for utt in UTTS:
        s = torch.load(ROOT / "cache/glm_tokens/en" / f"{utt}.pt")[0]

        def run(w, tfr):
            flow.inference_cfg_rate, flow.tfr, flow.e14_rules = w, tfr, tfr
            g = torch.Generator().manual_seed(h(f"{utt}|noise|0"))
            flow.noise_override = torch.randn([1, 6000, flow.mel_dim], generator=g)
            flow.dump_states = []
            seed_util.set_seed(h(f"{utt}|hift|0"))
            with torch.inference_mode():
                t2w.token2wav_with_cache(s["token"], n_timesteps=2, prompt_token=s["prompt_token"],
                                         prompt_feat=s["prompt_feat"], embedding=s["embedding"])
            d = flow.dump_states[-1]
            flow.dump_states = None
            return (d["x"] + (1.0 - d["t"]) * d["v"]).double()
        a, b, b2 = run(0.7, True), run(0.0, False), run(0.0, False)
        out.append(f"{utt} TFR vs w0 max|d| {float((a - b).abs().max()):.3e} (rel {float((a - b).abs().max() / b.abs().max()):.2e})"
                   f" | w0 twice {float((b - b2).abs().max()):.3e}")
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    key = sys.argv[1]
    lines = {"idx": idx, "glm": glm}[key]()
    (ROOT / f"logs/e14r_tfr_smoke_{key}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
