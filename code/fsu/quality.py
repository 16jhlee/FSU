"""Objective quality predictors for directories of generated wavs (environment env_objq = env_eval + utmosv2,
onnxruntime). Writes <wav_dir>/_metrics/objq.csv (DNSMOS: objq_dnsmos.csv); resumable per predictor.
  utmosv2  UTMOSv2 (sarulab-speech/UTMOSv2, official pretrained fold 0 / seed 42, predict_dataset 'sarulab')
  dnsmos   DNSMOS P.835 (microsoft/DNS-Challenge sig_bak_ovr.onnx, non-personalised) + P.808 (model_v8.onnx)
  nisqa    NISQA v2.0 (MOS, noisiness, discontinuity, coloration, loudness) and NISQA-TTS naturalness
The only adaptation: dnsmos_local calls librosa.resample(y, sr_in, sr_out) positionally (librosa < 0.10 API), and a shim
forwards it as keywords. Model code and weights are unchanged.
    python -m fsu.quality --meta <list> --wav_dir <dir> [--metrics utmosv2,dnsmos,nisqa]
    python -m fsu.quality --jobs <file with 'list<TAB>wav_dir' lines> --metrics ...   (each model is loaded once)
"""
import argparse
import gc
import os
import shutil
import sys
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fsu.common import CACHE, REPOS, resolve, seed_of

DNS = REPOS / "DNS-Challenge" / "DNSMOS"
NISQA = REPOS / "NISQA"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
COLS = {"utmosv2": ["utmosv2"], "dnsmos": ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_p808"],
        "nisqa": ["nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud", "nisqa_tts"]}


def list_utts(path):
    return [line.split("|")[0] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def linked_dir(wav_dir, utts):
    """Temporary folder with hard links (or copies) of exactly the listed wavs; NISQA and UTMOSv2 read whole folders."""
    tmp_root = CACHE / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="quality_", dir=str(tmp_root)))
    for u in utts:
        src = Path(wav_dir) / f"{u}.wav"
        try:
            os.link(src, tmp / f"{u}.wav")
        except OSError:
            shutil.copyfile(src, tmp / f"{u}.wav")
    return tmp


_WORKER = {}


def _utmos_cfg():
    """The prediction config exactly as utmosv2.create_model + UTMOSv2Model.predict use it (fusion_stage3, phase
    'prediction'). predict() sets cfg.dataset.remove_silent_section and restores the initial value (None) right after
    get_dataset(), before any item is read, and the dataset reads the flag lazily, so the official predict() never removes
    silent sections. This reproduces that effective behaviour (flag unset)."""
    import importlib
    from types import SimpleNamespace
    from utmosv2._settings import configure_execution
    _cfg = importlib.import_module("utmosv2.config.fusion_stage3")
    cfg = SimpleNamespace(**{k: v for k, v in _cfg.__dict__.items() if not k.startswith("__")})
    configure_execution(cfg)
    return cfg


def _utmos_item(args):
    """Worker: one dataset item (official dataset code); the random crop / mix is seeded by the utterance id."""
    from utmosv2.utils import get_dataset
    item, utt = args
    if "cfg" not in _WORKER:
        torch.set_num_threads(1)
        _WORKER["cfg"] = _utmos_cfg()
    np.random.seed(seed_of(f"utmosv2|{utt}"))
    ds = get_dataset(_WORKER["cfg"], [item], _WORKER["cfg"].phase)
    return ds[0]


class Utmosv2:
    """UTMOSv2Model.predict (one repetition, batch 16, autocast) with the CPU feature extraction (16 mel spectrograms
    per item) spread over worker processes. The official crop / mix randomness is seeded per utterance, so scores are
    reproducible and paired conditions share crop positions."""

    def __init__(self):
        import utmosv2
        self.m = utmosv2.create_model(pretrained=True, device=DEV)
        self.m._model.eval().to(DEV)
        self.pool = ProcessPoolExecutor(4)  # each worker needs about 2.4 GB RAM

    def __call__(self, utts, tmp):
        from torch.utils.data import default_collate
        data = self.m._prepare_data(None, 16000, None, tmp, None, None, "sarulab")
        stem = [Path(str(d.file_path)).stem for d in data]
        pred, buf = [], []

        def flush():
            t = default_collate(buf)
            x = [z.to(DEV, non_blocking=True) for z in t[:-1]]
            with torch.no_grad(), torch.autocast("cuda", enabled=DEV.startswith("cuda")):
                pred.append(self.m._model(*x).squeeze(1).float().cpu().numpy())
            buf.clear()
        for it in self.pool.map(_utmos_item, zip(data, stem), chunksize=4):  # streamed, 16 items at a time
            buf.append(it)
            if len(buf) == 16:
                flush()
        if buf:
            flush()
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
        so = D.ort.SessionOptions()  # one thread per session so the 6 workers do not oversubscribe the CPU
        so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
        self.cs.onnx_sess = D.ort.InferenceSession(str(DNS / "DNSMOS" / "sig_bak_ovr.onnx"), sess_options=so)
        self.cs.p808_onnx_sess = D.ort.InferenceSession(str(DNS / "DNSMOS" / "model_v8.onnx"), sess_options=so)

    def __call__(self, utts, tmp):
        with ThreadPoolExecutor(6) as ex:
            out = list(ex.map(lambda u: self.cs(str(tmp / f"{u}.wav"), self.D.SAMPLING_RATE, False), utts))
        return {"dnsmos_sig": [o["SIG"] for o in out], "dnsmos_bak": [o["BAK"] for o in out],
                "dnsmos_ovrl": [o["OVRL"] for o in out], "dnsmos_p808": [float(o["P808_MOS"]) for o in out]}


class Nisqa:
    WEIGHTS = [("nisqa.tar", {"mos_pred": "nisqa_mos", "noi_pred": "nisqa_noi", "dis_pred": "nisqa_dis",
                              "col_pred": "nisqa_col", "loud_pred": "nisqa_loud"}),
               ("nisqa_tts.tar", {"mos_pred": "nisqa_tts"})]

    def __init__(self):
        sys.path.insert(0, str(NISQA))
        from nisqa.NISQA_model import nisqaModel
        self.cls = nisqaModel

    def __call__(self, utts, tmp):
        res = {}
        for w, cols in self.WEIGHTS:
            args = dict(mode="predict_dir", pretrained_model=str(NISQA / "weights" / w), data_dir=str(tmp), output_dir=None,
                        num_workers=0, bs=16, ms_channel=None, tr_bs_val=16, tr_num_workers=0, tr_parallel=True, tr_device=None)
            df = self.cls(args).predict().set_index("deg")
            for c, name in cols.items():
                res[name] = [float(df.loc[f"{u}.wav", c]) for u in utts]
        return res


PREDICTORS = {"utmosv2": Utmosv2, "dnsmos": Dnsmos, "nisqa": Nisqa}


def csv_name(metric):
    return "objq_dnsmos.csv" if metric == "dnsmos" else "objq.csv"  # DNSMOS runs in its own CPU process


def load_state(meta, wav_dir, metric):
    wav_dir = Path(wav_dir)
    utts = [u for u in list_utts(meta) if (wav_dir / f"{u}.wav").exists()]
    out = wav_dir / "_metrics" / csv_name(metric)
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
            utts, _, df = load_state(meta, wd, m)
            if utts and not all(c in df for c in COLS[m]):
                todo.append((meta, wd))
        if not todo:
            continue
        pred = PREDICTORS[m]()
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
            gc.collect()
            torch.cuda.empty_cache()  # NISQA builds its models per folder
            print(m, wd, len(utts), round(float(df[COLS[m][-1]].mean()), 3), flush=True)
        del pred
        torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta")
    ap.add_argument("--wav_dir")
    ap.add_argument("--jobs")
    ap.add_argument("--metrics", default="utmosv2,dnsmos,nisqa")
    a = ap.parse_args(argv)
    if a.jobs:
        jobs = [tuple(line.split("\t")) for line in Path(a.jobs).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        jobs = [(a.meta, a.wav_dir)]
    run([(resolve(m), resolve(w)) for m, w in jobs], a.metrics.split(","))


if __name__ == "__main__":
    main()
