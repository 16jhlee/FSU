"""E10-1′ objective quality predictors for directories of generated wavs (no inference of TTS models).
  utmosv2 : UTMOSv2 (sarulab-speech/UTMOSv2, official pretrained fold 0 / seed 42, predict_dataset 'sarulab', defaults)
            -> utmosv2
  dnsmos  : DNSMOS P.835 (microsoft/DNS-Challenge DNSMOS/sig_bak_ovr.onnx, non-personalised; + P.808 model_v8)
            -> dnsmos_sig, dnsmos_bak, dnsmos_ovrl, dnsmos_p808   (official dnsmos_local.ComputeScore, CPU onnxruntime)
  nisqa   : NISQA v2.0 (gabrielmittag/NISQA weights/nisqa.tar: MOS + noisiness / discontinuity / coloration / loudness)
            and NISQA-TTS naturalness (weights/nisqa_tts.tar) -> nisqa_mos, nisqa_noi, nisqa_dis, nisqa_col, nisqa_loud, nisqa_tts
Output <wav_dir>/_metrics/objq.csv (utt_id + columns; resumable per predictor; separate from metrics.csv).
The only adaptation: dnsmos_local calls librosa.resample(y, sr_in, sr_out) positionally (librosa < 0.10 API); a shim
forwards it as keywords. Model code and weights are unchanged.
  python scripts/objq_metrics.py --meta <list.meta.lst> --wav_dir <dir> [--metrics utmosv2,dnsmos,nisqa]
  python scripts/objq_metrics.py --jobs <file: meta<TAB>wav_dir per line> --metrics ...   (each model loaded once)
(env_objq = venv over env_eval + utmosv2, onnxruntime)
"""
import argparse
import os
import shutil
import sys
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DNS = ROOT / "repos" / "DNS-Challenge" / "DNSMOS"
NISQA = ROOT / "repos" / "NISQA"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
COLS = {"utmosv2": ["utmosv2"], "dnsmos": ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"],
        "nisqa": ["nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud", "nisqa_tts"]}


def read_meta(path):
    return [l.split("|")[0] for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def linked_dir(wav_dir, utts):
    """Temp folder with hard links (same volume) or copies of exactly the listed wavs (NISQA / UTMOSv2 read folders)."""
    store = Path(Path(wav_dir).resolve().anchor) / "speechFM_store" / "tmp"
    tmp = Path(tempfile.mkdtemp(prefix="objq_", dir=str(store) if store.is_dir() else None))
    for u in utts:
        src = Path(wav_dir) / f"{u}.wav"
        try:
            os.link(src, tmp / f"{u}.wav")
        except OSError:
            shutil.copyfile(src, tmp / f"{u}.wav")
    return tmp


def h(s):
    import hashlib
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) % (2 ** 31 - 1)


_W = {}


def _utmos_cfg():
    """The prediction config exactly as utmosv2.create_model + UTMOSv2Model.predict use it (fusion_stage3, phase
    'prediction'). Note: predict() sets cfg.dataset.remove_silent_section and restores the initial value (None) right
    after get_dataset(), before the DataLoader reads any item, and the dataset reads the flag lazily -> in the official
    predict() silent-section removal is never applied. We reproduce that effective behaviour (flag unset)."""
    import importlib
    from types import SimpleNamespace
    from utmosv2._settings import configure_execution
    _cfg = importlib.import_module("utmosv2.config.fusion_stage3")
    cfg = SimpleNamespace(**{k: v for k, v in _cfg.__dict__.items() if not k.startswith("__")})
    configure_execution(cfg)
    return cfg


def _utmos_item(args):
    """Worker: one dataset item (official dataset code), random crop / mix seeded by the utterance id."""
    import numpy as np
    from utmosv2.utils import get_dataset
    item, utt = args
    if "cfg" not in _W:
        torch.set_num_threads(1)
        _W["cfg"] = _utmos_cfg()
    np.random.seed(h(f"utmosv2|{utt}"))
    ds = get_dataset(_W["cfg"], [item], _W["cfg"].phase)
    return ds[0]


class Utmosv2:
    """UTMOSv2Model.predict (num_repetitions 1, batch 16, autocast) with the CPU feature extraction (16 mel spectrograms
    per item, librosa) spread over worker processes. Windows cannot pickle the DataLoader dataset (config module), so
    each worker rebuilds the identical config. The official crop / mix randomness is seeded per utterance
    (hash('utmosv2|utt')) so scores are reproducible and paired conditions share crop positions."""

    def __init__(self):
        import utmosv2
        from concurrent.futures import ProcessPoolExecutor
        self.m = utmosv2.create_model(pretrained=True, device=DEV)
        self.m._model.eval().to(DEV)
        self.pool = ProcessPoolExecutor(4)  # each worker ~2.4 GB RAM (8 workers pushed the page file to 28 GB)

    def __call__(self, utts, tmp):
        from torch.utils.data import default_collate
        data = self.m._prepare_data(None, 16000, None, tmp, None, None, "sarulab")
        stem = [Path(str(d.file_path)).stem for d in data]
        # stream: items arrive in order and are consumed 16 at a time (holding a whole folder of 16-spectrogram items
        # took ~17 MB per utterance -> tens of GB for 2,020-utterance folders)
        pred, buf = [], []

        def flush():
            t = default_collate(buf)
            x = [z.to(DEV, non_blocking=True) for z in t[:-1]]
            with torch.no_grad(), torch.autocast("cuda", enabled=DEV.startswith("cuda")):
                pred.append(self.m._model(*x).squeeze(1).float().cpu().numpy())
            buf.clear()
        for it in self.pool.map(_utmos_item, zip(data, stem), chunksize=4):
            buf.append(it)
            if len(buf) == 16:
                flush()
        if buf:
            flush()
        import numpy as np
        p = dict(zip(stem, np.concatenate(pred).tolist()))
        return {"utmosv2": [p[u] for u in utts]}


class Dnsmos:
    def __init__(self):
        import librosa
        sys.path.insert(0, str(DNS))
        import dnsmos_local as D
        shim = types.ModuleType("librosa_shim")
        shim.__dict__.update({k: getattr(librosa, k) for k in dir(librosa) if not k.startswith("__")})
        shim.resample = lambda y, a, b: librosa.resample(y, orig_sr=a, target_sr=b)
        D.librosa = shim
        self.D = D
        self.cs = D.ComputeScore(str(DNS / "DNSMOS" / "sig_bak_ovr.onnx"), str(DNS / "DNSMOS" / "model_v8.onnx"))
        # same models; one intra-op thread per session so the 6 workers do not oversubscribe the CPU (GPU jobs share it)
        so = D.ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
        self.cs.onnx_sess = D.ort.InferenceSession(str(DNS / "DNSMOS" / "sig_bak_ovr.onnx"), sess_options=so)
        self.cs.p808_onnx_sess = D.ort.InferenceSession(str(DNS / "DNSMOS" / "model_v8.onnx"), sess_options=so)

    def __call__(self, utts, tmp):
        with ThreadPoolExecutor(6) as ex:
            out = list(ex.map(lambda u: self.cs(str(tmp / f"{u}.wav"), self.D.SAMPLING_RATE, False), utts))
        return {"dnsmos_sig": [o["SIG"] for o in out], "dnsmos_bak": [o["BAK"] for o in out],
                "dnsmos_ovrl": [o["OVRL"] for o in out], "dnsmos_p808": [float(o["P808_MOS"]) for o in out]}


class Nisqa:
    W = [("nisqa.tar", {"mos_pred": "nisqa_mos", "noi_pred": "nisqa_noi", "dis_pred": "nisqa_dis", "col_pred": "nisqa_col",
                        "loud_pred": "nisqa_loud"}), ("nisqa_tts.tar", {"mos_pred": "nisqa_tts"})]

    def __init__(self):
        sys.path.insert(0, str(NISQA))
        from nisqa.NISQA_model import nisqaModel
        self.cls = nisqaModel

    def __call__(self, utts, tmp):
        res = {}
        for w, cols in self.W:
            args = dict(mode="predict_dir", pretrained_model=str(NISQA / "weights" / w), data_dir=str(tmp), output_dir=None,
                        num_workers=0, bs=16, ms_channel=None, tr_bs_val=16, tr_num_workers=0, tr_parallel=True, tr_device=None)
            df = self.cls(args).predict().set_index("deg")
            for c, name in cols.items():
                res[name] = [float(df.loc[f"{u}.wav", c]) for u in utts]
        return res


PRED = {"utmosv2": Utmosv2, "dnsmos": Dnsmos, "nisqa": Nisqa}


def fname(m):
    return "objq_dnsmos.csv" if m == "dnsmos" else "objq.csv"  # DNSMOS runs in its own CPU process -> own file


def load_state(meta, wav_dir, m):
    wav_dir = Path(wav_dir)
    utts = [u for u in read_meta(meta) if (wav_dir / f"{u}.wav").exists()]
    out = wav_dir / "_metrics" / fname(m)
    df = pd.DataFrame({"utt_id": utts})
    if out.exists():
        old = pd.read_csv(out)
        if list(old.utt_id) == utts:
            df = old
    return utts, out, df


def run(jobs, metrics):
    for m in metrics:
        todo = []
        for meta, wd in jobs:
            utts, out, df = load_state(meta, wd, m)
            if utts and not all(c in df for c in COLS[m]):
                todo.append((meta, wd))
        if not todo:
            continue
        pred = PRED[m]()
        for meta, wd in todo:
            utts, out, df = load_state(meta, wd, m)
            tmp = linked_dir(wd, utts)
            try:
                for c, v in pred(utts, tmp).items():
                    df[c] = v
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            import gc
            gc.collect()
            torch.cuda.empty_cache()  # NISQA builds its models per folder; release cached blocks (shared GPU with eval)
            print(m, wd, len(utts), round(float(df[COLS[m][-1]].mean()), 3), flush=True)
        del pred
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta")
    ap.add_argument("--wav_dir")
    ap.add_argument("--jobs")
    ap.add_argument("--metrics", default="utmosv2,dnsmos,nisqa")
    a = ap.parse_args()
    if a.jobs:
        jobs = [tuple(l.split("\t")) for l in Path(a.jobs).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        jobs = [(a.meta, a.wav_dir)]
    run(jobs, a.metrics.split(","))


if __name__ == "__main__":
    main()
