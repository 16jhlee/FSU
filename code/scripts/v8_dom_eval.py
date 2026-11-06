"""v8 §3.1 exploration metrics.  python scripts/v8_dom_eval.py audio|image   (env_dom, GPU)
audio: CLAP (laion_clap HTSAT, 630k-audioset-best) text-audio cosine (prompt adherence) and audio-audio cosine to the
       AudioCaps reference clip of the same caption; FD on CLAP audio embeddings vs the 100 reference clips (FD-CLAP;
       n = 100, indicative only). Official AudioLDM-eval FAD/KL is left for the main experiment.
image: CLIP score (torchmetrics, openai/clip-vit-large-patch14), ImageReward (v1.0), KID (subset 50) and FID (2048-d)
       vs the 100 COCO reference images (n = 100: FID is biased upward, compare conditions only relative to each other).
Output results/E8v8dom_<domain>.csv (one row per condition) and per-item CSVs in each condition folder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "E8v8dom"


def fd(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


def audio():
    import laion_clap
    caps = pd.read_csv(ROOT / "data/raw/dom/audiocaps/captions.tsv", sep="\t")
    m = laion_clap.CLAP_Module(enable_fusion=False)
    m.load_ckpt()
    with torch.no_grad():
        T = m.get_text_embedding(list(caps.caption), use_tensor=False)
        R = m.get_audio_embedding_from_filelist([str(ROOT / f"data/raw/dom/audiocaps/ref/{i}.wav") for i in caps.id], use_tensor=False)
    nrm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    T, R = nrm(T), nrm(R)
    rows = []
    for d in sorted((OUT / "audio").iterdir()):
        fs = [str(d / f"{i}.wav") for i in caps.id]
        if not all(Path(f).exists() for f in fs):
            continue
        with torch.no_grad():
            A = nrm(m.get_audio_embedding_from_filelist(fs, use_tensor=False))
        clap_t, clap_r = (A * T).sum(1), (A * R).sum(1)
        pd.DataFrame(dict(id=caps.id, clap_text=clap_t, clap_ref=clap_r)).to_csv(d / "_metrics.csv", index=False)
        rows.append(dict(cond=d.name, clap_text=clap_t.mean(), clap_ref=clap_r.mean(), fd_clap=fd(A, R)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/E8v8dom_audio.csv", index=False)


def image():
    from PIL import Image
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.multimodal.clip_score import CLIPScore
    import ImageReward as RM
    caps = pd.read_csv(ROOT / "data/raw/dom/coco/captions.tsv", sep="\t")
    clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")
    rm = RM.load("ImageReward-v1.0", device="cuda")

    def load(p, size=299):
        im = Image.open(p).convert("RGB").resize((size, size))
        return torch.from_numpy(np.array(im)).permute(2, 0, 1)
    ref = torch.stack([load(ROOT / f"data/raw/dom/coco/ref/{i}.jpg") for i in caps.id])
    rows = []
    for d in sorted((OUT / "image").iterdir()):
        fs = [d / f"{i}.png" for i in caps.id]
        if not all(f.exists() for f in fs):
            continue
        X = torch.stack([load(f) for f in fs])
        X224 = torch.stack([load(f, 224) for f in fs])
        cs = [float(clip(X224[i:i + 1].cuda(), [caps.caption[i]])) for i in range(len(fs))]
        ir = [float(rm.score(caps.caption[i], str(fs[i]))) for i in range(len(fs))]
        fid = FrechetInceptionDistance(feature=2048).to("cuda")
        kid = KernelInceptionDistance(subset_size=50).to("cuda")
        for met in (fid, kid):
            met.update(ref.cuda(), real=True)
            met.update(X.cuda(), real=False)
        km, _ = kid.compute()
        pd.DataFrame(dict(id=caps.id, clip=cs, imagereward=ir)).to_csv(d / "_metrics.csv", index=False)
        rows.append(dict(cond=d.name, clip=np.mean(cs), imagereward=np.mean(ir), fid=float(fid.compute()), kid=float(km)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/E8v8dom_image.csv", index=False)


if __name__ == "__main__":
    {"audio": audio, "image": image}[sys.argv[1]]()
