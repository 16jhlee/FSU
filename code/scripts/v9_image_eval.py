"""v9 §1.1 image metrics (env_dom, GPU).  python scripts/v9_image_eval.py [model ...]   (default: sd3 sd35)
Per complete folder out/E9/image/<model>/<cond>/s<k>/ (n = #png must equal the first n COCO ids, n in {1000, 300, 200, 100}):
  _metrics.csv  id, clip (CLIP score = 100 max(cos, 0), openai/clip-vit-large-patch14, as torchmetrics), imagereward (v1.0)
  _clip.npy     L2-normalised CLIP ViT-L/14 image embeddings (diversity / geometry D5)
  _incep.npy    Inception-v3 pool3 2048-d features (torchmetrics FID extractor; KID / FID vs COCO reference images)
Reference features data/raw/dom/coco1k/_incep_ref.npy (first 1000 reference images). Skip-existing (resumable).
Aggregation (KID, FID, paired tests) lives in v9_analyze.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "E9" / "image"
DATA = ROOT / "data/raw/dom/coco1k"
NS = (1000, 300, 200, 100)


def u8(p, size):
    im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1)


class Scorer:
    def __init__(self):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from transformers import CLIPModel, CLIPProcessor
        import ImageReward as RM
        self.inc = FrechetInceptionDistance(feature=2048).inception.to("cuda").eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.rm = RM.load("ImageReward-v1.0", device="cuda")

    @torch.no_grad()
    def incep(self, files):
        out = []
        for j in range(0, len(files), 25):
            x = torch.stack([u8(f, 299) for f in files[j:j + 25]]).cuda()
            out.append(self.inc(x).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def clip_ir(self, files, texts):
        embs, cs, ir = [], [], []
        for j in range(0, len(files), 16):
            ims = [Image.open(f).convert("RGB") for f in files[j:j + 16]]
            b = self.proc(text=texts[j:j + 16], images=ims, return_tensors="pt", padding=True, truncation=True).to("cuda")
            o = self.clip(**b)
            ie = torch.nn.functional.normalize(o.image_embeds.float(), dim=-1)
            te = torch.nn.functional.normalize(o.text_embeds.float(), dim=-1)
            cs += (100 * (ie * te).sum(-1).clamp(min=0)).tolist()
            embs.append(ie.cpu().numpy())
            ir += [float(self.rm.score(t, str(f))) for t, f in zip(texts[j:j + 16], files[j:j + 16])]
        return np.concatenate(embs), cs, ir


def main(models):
    caps = pd.read_csv(DATA / "captions.tsv", sep="\t")
    S = Scorer()
    rf = DATA / "_incep_ref.npy"
    if not rf.exists():
        np.save(rf, S.incep([DATA / "ref" / f"{i}.jpg" for i in caps.id[:1000]]))
    for model in models:
        for d in sorted((OUT / model).glob("*/s*")):
            if (d / "_metrics.csv").exists() or not (d / "_steps.json").exists():  # _steps.json = generator finished the folder
                continue
            n = len(list(d.glob("*.png")))
            if n not in NS:
                continue
            ids = list(caps.id[:n])
            files = [d / f"{i}.png" for i in ids]
            if not all(f.exists() for f in files):
                continue
            np.save(d / "_incep.npy", S.incep(files))
            emb, cs, ir = S.clip_ir(files, list(caps.caption[:n]))
            np.save(d / "_clip.npy", emb)
            pd.DataFrame(dict(id=ids, clip=cs, imagereward=ir)).to_csv(d / "_metrics.csv", index=False)
            print(f"{model} {d.parent.name} {d.name} n={n} clip={np.mean(cs):.2f} ir={np.mean(ir):.3f}", flush=True)
    print("image eval pass done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["sd3", "sd35"])
