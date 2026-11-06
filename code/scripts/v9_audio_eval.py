"""v9 §1.2 audio metrics. Two stages (resumable, per complete folder out/E9/audio/<model>/<cond>/s<k>/, _steps.json present):
  python scripts/v9_audio_eval.py clap [model ...]    (env_dom)    -> _clap.csv (clap_text, clap_ref), _clap.npy (L2 audio emb)
      laion_clap HTSAT 630k-audioset-best (as v8); text-audio cosine (adherence), cosine to the AudioCaps reference clip.
  python scripts/v9_audio_eval.py toolkit [model ...] (env_aeval)  -> _passt.csv (per-item KL_passt), _aeval.json (FD/KL PANNs, FAD)
      KL_passt: Stability-AI stable-audio-metrics (repos/stable-audio-metrics, PaSST, 32 kHz, 10 s windows, softmax, KL(ref||gen));
      audioldm_eval (cnn14 backbone, 16 kHz): frechet_distance (PANNs), kullback_leibler_divergence_sigmoid/softmax, FAD (VGGish);
      only for folders with all 883 clips (set-level metrics). Reference = data/raw/dom/audiocaps_full/ref.
Default models: tangoflux sao.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "E9" / "audio"
DATA = ROOT / "data/raw/dom/audiocaps_full"
TMP = Path(r"C:\speechFM_store\tmp\aeval")  # same volume as out/E9 (hard links)


def folders(models, marker):
    caps = pd.read_csv(DATA / "captions.tsv", sep="\t")
    for m in models:
        for d in sorted((OUT / m).glob("*/s*")):
            if (d / marker).exists() or not (d / "_steps.json").exists():
                continue
            n = len(list(d.glob("*.wav")))
            ids = list(caps.id[:n])
            if n == 0 or not all((d / f"{i}.wav").exists() for i in ids):
                continue
            yield m, d, ids, caps


def clap(models):
    import torch
    import laion_clap
    m = laion_clap.CLAP_Module(enable_fusion=False)
    m.load_ckpt()
    caps = pd.read_csv(DATA / "captions.tsv", sep="\t")
    nrm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    rt = DATA / "_clap_ref.npz"
    if not rt.exists():
        with torch.no_grad():
            T = np.concatenate([m.get_text_embedding(list(caps.caption[j:j + 64]), use_tensor=False) for j in range(0, len(caps), 64)])
            R = np.concatenate([m.get_audio_embedding_from_filelist([str(DATA / "ref" / f"{i}.wav") for i in caps.id[j:j + 32]], use_tensor=False)
                                for j in range(0, len(caps), 32)])
        np.savez(rt, T=nrm(T), R=nrm(R))
    Z = np.load(rt)
    T, R = Z["T"], Z["R"]
    for model, d, ids, _ in folders(models, "_clap.csv"):
        n = len(ids)
        with torch.no_grad():
            A = np.concatenate([m.get_audio_embedding_from_filelist([str(d / f"{i}.wav") for i in ids[j:j + 32]], use_tensor=False)
                                for j in range(0, n, 32)])
        A = nrm(A)
        np.save(d / "_clap.npy", A)
        ct, cr = (A * T[:n]).sum(1), (A * R[:n]).sum(1)
        pd.DataFrame(dict(id=ids, clap_text=ct, clap_ref=cr)).to_csv(d / "_clap.csv", index=False)
        print(f"{model} {d.parent.name} {d.name} n={n} clap_text={ct.mean():.4f} clap_ref={cr.mean():.4f}", flush=True)
    print("clap pass done", flush=True)


def toolkit(models):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT / "repos/stable-audio-metrics"))
    from src.passt_kld import return_probabilities
    from hear21passt.base import get_basic_model
    passt = get_basic_model(mode="logits").cuda().eval()
    caps = pd.read_csv(DATA / "captions.tsv", sep="\t")
    rp = DATA / "_passt_ref.pt"
    if not rp.exists():
        with torch.no_grad():
            torch.save({i: return_probabilities(passt, str(DATA / "ref" / f"{i}.wav")) for i in caps.id}, rp)
    ref = torch.load(rp)
    helper = None
    for model, d, ids, _ in folders(models, "_passt.csv"):
        with torch.no_grad():
            kl = [float(F.kl_div((ref[i] + 1e-6).log(), return_probabilities(passt, str(d / f"{i}.wav")), reduction="sum",
                                 log_target=False)) for i in ids]
        if len(ids) == len(caps):
            if helper is None:
                from audioldm_eval import EvaluationHelper
                helper = EvaluationHelper(16000, torch.device("cuda"), backbone="cnn14")
                helper.get_current_time = lambda: time.strftime("%Y%m%d-%H%M%S")  # Windows: no ':' in its result filename
            tmp = TMP / f"{model}__{d.parent.name}__{d.name}"  # wav-only view (the toolkit reads every file in the folder)
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            for i in ids:
                os.link(d / f"{i}.wav", tmp / f"{i}.wav")
            out = helper.calculate_metrics(str(tmp), str(DATA / "ref"), same_name=True)
            shutil.rmtree(tmp)
            (d / "_aeval.json").write_text(json.dumps({k: float(v) for k, v in out.items()}))
        pd.DataFrame(dict(id=ids, kl_passt=kl)).to_csv(d / "_passt.csv", index=False)
        print(f"{model} {d.parent.name} {d.name} n={len(ids)} kl_passt={np.mean(kl):.4f}", flush=True)
    print("toolkit pass done", flush=True)


if __name__ == "__main__":
    {"clap": clap, "toolkit": toolkit}[sys.argv[1]](sys.argv[2:] or ["tangoflux", "sao"])
