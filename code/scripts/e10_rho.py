"""E10-7: direct measurement of rho_c (condition (C) of the method section) — en first 100 utterances (data/lists/en_100),
the E8'-9e procedure, mel / feature space of each model's decoder output (generated region only, RMS = sqrt(mean a^2)).
Only official sampler outputs are used (no hooks into the network):
  T_k      = teacher (official NFE, official w) output, noise seeds k = 0..7 (hash(utt|noise|k) as everywhere)
  xbar_k   = NFE 1, w = 0 output = z_k + v_c(z_k, 0)            (estimate of the conditional mean x_c)
  g_k      = [NFE 1, w = 1 output] - xbar_k = v_c - v_u at t = 0 (CFG (1+w)v_c - w v_u, all-null branch)
             ZipVoice: its official CFG at t <= 0.5 uses the text-null branch (speech prompt kept) with the scale doubled,
             so g = [NFE 1, w = 0.5] - xbar = v_c - v_u(text-null) — recorded as the official-branch guidance.
Per utterance and k in {0, 1}:
  rho_nn   = min_j RMS(xbar_k - T_j)        distance from the conditional-mean estimate to the teacher cloud (support)
  rho_mean = RMS(xbar_k - mean_j T_j)       distance to the cloud centre (E8'-9e d_T(0))
  sigma    = mean over pairs RMS(T_j - T_j') teacher seed-to-seed distance
  g_rms    = RMS(g_k)
python scripts/e10_rho.py cv2|cv3|idx|f5|zip [n_utt=100]  (env_cosy | env_cosy | env_index | env_f5dmo | env_zip)
Output results/E10_rho_<model>.csv
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# E12-3: E10_RHO_KT = number of teacher seeds (default 8 = E10-7); E10_RHO_OUT = output tag; per-seed distances kept
KT = int(os.environ.get("E10_RHO_KT", "8"))
OUT = os.environ.get("E10_RHO_OUT", "E10_rho")
sys.path.insert(0, str(ROOT / "scripts"))


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))


def h(s, mod=2 ** 31 - 1):
    import hashlib
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16) % mod


def stats(utt, T, X, G):
    rows = []
    Tm = np.mean(T, 0)
    sig = np.mean([rms(a - b) for a, b in itertools.combinations(T, 2)])
    for k in range(len(X)):
        rows.append(dict(utt=utt, k=k, rho_nn=min(rms(X[k] - t) for t in T), rho_mean=rms(X[k] - Tm), sigma=sig,
                         g_rms=rms(G[k]), T_rms=rms(Tm), d_all=";".join(f"{rms(X[k] - t):.5f}" for t in T)))
    return rows


# ---------------- CosyVoice2 / CosyVoice3 ----------------
def cosy(key, utts):
    from v8_theory_cosy import Runner
    R = Runner(key)
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    rows = []
    for utt in utts:
        seg = torch.load(ROOT / R.cache / f"{utt}.pt")[0]
        T = [R.run(seg, utt, k, t10, 0.7)["final"] for k in range(KT)]
        X = [R.run(seg, utt, k, [0.0, 1.0], 0.0)["final"] for k in range(2)]
        G = [R.run(seg, utt, k, [0.0, 1.0], 1.0)["final"] - X[k] for k in range(2)]
        rows += stats(utt, T, X, G)
        print(utt, flush=True)
    return rows


# ---------------- IndexTTS-2 (first text segment, as v8_theory_index) ----------------
def index(utts):
    from run_index import REPO, load_model, seed_all
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    dev = tts.device
    cfm.cfg_interval, cfm.cfg_ramp, cfm.cfg_branch = None, None, "all"

    def out(s, utt, k, n, w):
        seed_all(h(f"{utt}|noise|{k}|0"))
        with torch.inference_mode():
            vc = cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev), None, n,
                               inference_cfg_rate=w)
        return vc[0, :, s["prompt"].size(-1):].double().cpu().numpy()
    rows = []
    for utt in utts:
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]
        T = [out(s, utt, k, 25, 0.7) for k in range(KT)]
        X = [out(s, utt, k, 1, 0.0) for k in range(2)]
        G = [out(s, utt, k, 1, 1.0) - X[k] for k in range(2)]
        rows += stats(utt, T, X, G)
        print(utt, flush=True)
    return rows


# ---------------- F5-TTS v0 (official F5TTS_Base, NFE 32, cfg 2, sway -1, as E1/E8') ----------------
def f5(utts):
    from importlib.resources import files
    from omegaconf import OmegaConf
    from f5_tts.eval.utils_eval import get_inference_prompt, get_seedtts_testset_metainfo
    from f5_tts.infer.utils_infer import load_checkpoint
    from f5_tts.model import CFM
    from f5_tts.model.utils import get_tokenizer
    from hydra.utils import get_class
    device = "cuda:0"
    cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml")))
    model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")
    ms = cfg.model.mel_spec
    vocab_char_map, vocab_size = get_tokenizer(cfg.datasets.name, cfg.model.tokenizer)
    model = CFM(transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=ms.n_mel_channels),
                mel_spec_kwargs=dict(n_fft=ms.n_fft, hop_length=ms.hop_length, win_length=ms.win_length,
                                     n_mel_channels=ms.n_mel_channels, target_sample_rate=ms.target_sample_rate,
                                     mel_spec_type=ms.mel_spec_type),
                odeint_kwargs=dict(method="euler"), vocab_char_map=vocab_char_map).to(device)
    model = load_checkpoint(model, str(ROOT / "ckpt/F5-TTS/F5TTS_Base/model_1200000.safetensors"), device, dtype=None,
                            use_ema=True)
    model.cfg_interval = None
    meta = get_seedtts_testset_metainfo(str(ROOT / "data/lists/en_100.meta.lst"))
    prompts = get_inference_prompt(meta, speed=1.0, tokenizer=cfg.model.tokenizer, target_sample_rate=ms.target_sample_rate,
                                   n_mel_channels=ms.n_mel_channels, hop_length=ms.hop_length,
                                   mel_spec_type=ms.mel_spec_type, target_rms=0.1, use_truth_duration=False,
                                   infer_batch_size=1)
    want = set(utts)
    rows = []
    for us, _, ref_mels, ref_lens, tot_lens, texts in prompts:
        utt = us[0]
        if utt not in want:
            continue

        def out(k, n, w):
            with torch.inference_mode():
                gen, _ = model.sample(cond=ref_mels.to(device), text=texts,
                                      duration=torch.tensor(tot_lens, dtype=torch.long, device=device),
                                      lens=torch.tensor(ref_lens, dtype=torch.long, device=device), steps=n,
                                      cfg_strength=w, sway_sampling_coef=-1.0, seed=h(f"{utt}|noise|{k}"), use_epss=False)
            return gen[0, ref_lens[0]:tot_lens[0], :].double().cpu().numpy().T
        T = [out(k, 32, 2.0) for k in range(KT)]
        X = [out(k, 1, 0.0) for k in range(2)]
        G = [out(k, 1, 1.0) - X[k] for k in range(2)]
        rows += stats(utt, T, X, G)
        print(utt, flush=True)
    return rows


# ---------------- ZipVoice teacher (NFE 16, w 1.0, t-shift 0.5, raw evaluation; as E1/E7) ----------------
def zip_(utts):
    import json
    import zipvoice.bin.infer_zipvoice as Z
    from lhotse.utils import fix_random_seed
    from zipvoice.models.modules import solver as zsolver
    captured = {}

    class Capture:  # stands in for the vocoder: keeps the predicted features, returns silence of the right length
        def decode(self, feats):
            captured["f"] = feats.detach().float().cpu().numpy()[0]
            return torch.zeros(1, 1, feats.shape[-1] * 256)
    # model / tokenizer / feature extractor built exactly as infer_zipvoice.main() (model_name zipvoice, local dir, emilia)
    for k_, v_ in {"mode": "euler", "t_span": None, "restart": None, "cfg_interval": None, "pci": False}.items():
        zsolver.E5[k_] = v_
    md = ROOT / "ckpt/ZipVoice/zipvoice"
    tokenizer = Z.EmiliaTokenizer(token_file=md / "tokens.txt")
    cfgj = json.loads((md / "model.json").read_text())
    model = Z.ZipVoice(**cfgj["model"], vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    Z.load_checkpoint(filename=md / "model.pt", model=model, strict=True)
    device = torch.device("cuda", 0)
    model = model.to(device).eval()
    feature_extractor = Z.VocosFbank()
    meta = {l.split("|")[0]: l.split("|") for l in (ROOT / "data/lists/en_100.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()}
    os.makedirs(ROOT / "out/smoke/E10_rho_zip", exist_ok=True)
    rows = []
    for utt in utts:
        _, ptext, pwav, text = meta[utt][:4]

        def out(k, n, w):
            fix_random_seed(h(f"{utt}|noise|{k}"))  # the --seed-per-utt rule of the patched infer_zipvoice
            with torch.inference_mode():  # as the official generate_list
                Z.generate_sentence_raw_evaluation(save_path=str(ROOT / "out/smoke/E10_rho_zip/tmp.wav"), prompt_text=ptext,
                                               prompt_wav=pwav, text=text, model=model, vocoder=Capture(),
                                               tokenizer=tokenizer, feature_extractor=feature_extractor, device=device,
                                               num_step=n, guidance_scale=w, t_shift=0.5)
            return captured["f"].astype(np.float64)
        T = [out(k, 16, 1.0) for k in range(KT)]
        X = [out(k, 1, 0.0) for k in range(2)]
        G = [out(k, 1, 0.5) - X[k] for k in range(2)]
        rows += stats(utt, T, X, G)
        print(utt, flush=True)
    return rows


def main():
    key = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    utts = [l.split("|")[0] for l in (ROOT / "data/lists/en_100.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    rows = {"cv2": lambda: cosy("cv2", utts), "cv3": lambda: cosy("cv3", utts), "idx": lambda: index(utts),
            "f5": lambda: f5(utts), "zip": lambda: zip_(utts)}[key]()
    import csv  # env_zip has no pandas
    with open(ROOT / f"results/{OUT}_{key}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print({c: round(float(np.mean([r[c] for r in rows])), 4) for c in rows[0] if c not in ("utt", "k", "d_all")})


if __name__ == "__main__":
    main()
