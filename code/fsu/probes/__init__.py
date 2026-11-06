"""Model-direct measurements ("probes"): they load a decoder checkpoint and its cached conditions, call the flow-matching
network directly (one Euler step from noise, state dumps of the patched samplers, teacher samples), and write
results/probes/<name>.csv. No generated wav is read.

Every probe uses the first 100 utterances of the English analysis subset (seedtts_en_subset100), the first text segment
of each utterance, the generated region of the decoder output (prompt frames removed), RMS(a) = sqrt(mean a^2), and the
per-utterance noise draws of the runners (seed_of(f"{utt}|noise|{k}")).

Modules (each runs in its decoder's environment):
    python -m fsu.probes.cosyvoice <measurement> --decoder cosyvoice2|cosyvoice3   env_cosy
    python -m fsu.probes.indextts  <measurement>                                   env_index
    python -m fsu.probes.glmtts    <measurement>                                   env_glm
    python -m fsu.probes.textcond  <measurement> --decoder f5tts|zipvoice          env_f5dmo | env_zip
    python -m fsu.probes.timing    <measurement>                                   env_cosy, idle GPU
Driver (tools environment; one GPU process at a time, skips measurements whose CSV exists):
    python -m fsu.probes list
    python -m fsu.probes run [output names or prefixes ...]
"""
import argparse
import csv
import itertools
import sys
from dataclasses import dataclass

import numpy as np

from fsu import datasets
from fsu.common import LOGS, REPOS, ROOT

RESULTS = ROOT / "results" / "probes"
TEST_SET = "seedtts_en_subset100"


# ---------------------------------------------------------------- shared numerics
def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))


def cosv(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def cos_grid(n):
    """Official CosyVoice / GLM-TTS cosine grid with n steps (numpy float64 values)."""
    return list(1 - np.cos(np.pi / 2 * np.arange(n + 1) / n))


def utterance_rows(n=None, test_set=TEST_SET):
    """(utt, prompt_text, prompt_wav, text) rows of the probe list; the first n if given."""
    return datasets.get(test_set).rows()[:n]


def token_cache(decoder, test_set=TEST_SET):
    from fsu.experiment import DECODERS
    return DECODERS[decoder].token_cache(test_set)


def branch_summary(utt, G):
    """First-step guidance vectors g1 = w (v_c - v_u) per unconditional branch (dict branch -> C x T array):
    E_tot = ||g||^2 / (C T); rms = ||g|| / sqrt(C T); timbre_share = share of ||g||^2 in the frame-mean component (per-bin
    mean over time); temporal_share = 1 - timbre_share; plus the cosines between the three branch vectors."""
    rows = []
    for br, g in G.items():
        mean = g.mean(1, keepdims=True)
        tot = (g ** 2).sum()
        timbre = float((np.broadcast_to(mean, g.shape) ** 2).sum() / tot)
        rows.append(dict(utt=utt, branch=br, E_tot=tot / g.size, rms=float(np.sqrt(tot / g.size)),
                         timbre_share=timbre, temporal_share=1 - timbre))
    cos = lambda a, b: float(a.ravel() @ b.ravel() / np.linalg.norm(a) / np.linalg.norm(b))  # noqa: E731
    if {"all", "speaker", "token"} <= set(G):
        rows.append(dict(utt=utt, branch="cos", cos_speaker_token=cos(G["speaker"], G["token"]),
                         cos_all_speaker=cos(G["all"], G["speaker"]), cos_all_token=cos(G["all"], G["token"])))
    return rows


def cloud_stats(utt, T, X, G):
    """Distance of the conditional-mean estimate to the teacher cloud. T = teacher outputs (official steps and weight,
    seeds 0..K-1), X[k] = 1-step w = 0 output (z_k + v_c), G[k] = 1-step guidance v_c - v_u. Per k:
    rho_nn = min_j RMS(X_k - T_j); rho_mean = RMS(X_k - mean T); sigma = mean pairwise RMS(T_j - T_j');
    g_rms = RMS(G_k); T_rms = RMS(mean T); d_all = every RMS(X_k - T_j)."""
    rows = []
    Tm = np.mean(T, 0)
    sig = np.mean([rms(a - b) for a, b in itertools.combinations(T, 2)])
    for k in range(len(X)):
        rows.append(dict(utt=utt, k=k, rho_nn=min(rms(X[k] - t) for t in T), rho_mean=rms(X[k] - Tm), sigma=sig,
                         g_rms=rms(G[k]), T_rms=rms(Tm), d_all=";".join(f"{rms(X[k] - t):.5f}" for t in T)))
    return rows


W1_GRID = [-1.0, -0.7, -0.4333, -0.2, 0.0, 0.2, 0.35, 0.7, 1.05]  # first-step weights of the w1 sweep
T1_COS2 = float(1 - np.cos(np.pi / 4))  # first step of the 2-step cosine grid


def deviation_rows(rows, model, utt, k, eps, vc, vu, Tm, t1, branch="official"):
    """First-step deviation from the teacher paths: z(w1) = eps + t1 [(1 + w1) v_c - w1 v_u],
    pi_j = (1 - t1) eps + t1 x_j over the teacher samples x_j; dist_nn = min_j RMS(z - pi_j) / t1,
    dist_rms = sqrt(mean_j RMS(z - pi_j)^2) / t1, g_rms = RMS(v_c - v_u)."""
    pi = (1 - t1) * eps[None] + t1 * Tm
    for w in W1_GRID:
        z = eps + t1 * ((1 + w) * vc - w * vu)
        d = np.array([rms(z - p) for p in pi])
        rows.append(dict(model=model, branch=branch, utt=utt, k=k, t1=t1, w1=w, dist_nn=d.min() / t1,
                         dist_rms=float(np.sqrt(np.mean(d ** 2))) / t1, g_rms=rms(vc - vu)))


# ---------------------------------------------------------------- catalogue
@dataclass(frozen=True)
class Measurement:
    module: str  # fsu.probes.<module>
    name: str  # command (= probe method)
    output: str  # results/probes/<output>.csv; {decoder} is filled in
    decoders: tuple
    n_utt: int = None  # default number of utterances (None = the whole list)
    teachers: int = None  # default number of teacher seeds, if the measurement has the option
    writer: str = "frame"  # frame (pandas) | csv (fields of the first row) | csv_sorted (sorted union of fields)
    doc: str = ""

    def out_name(self, decoder, n_utt=None, teachers=None):
        name = self.output.format(decoder=decoder)
        if n_utt is not None and n_utt != self.n_utt:
            name += f"_{n_utt}utt"
        if teachers is not None and teachers != self.teachers:
            name += f"_{teachers}teachers"
        return name


CV = ("cosyvoice2", "cosyvoice3")
CATALOGUE = [
    Measurement("cosyvoice", "theory", "closed_form_checks_{decoder}", CV, 50,
                doc="seed invariance of v_c - v_u at t = 0 (8 seeds), one-step closed form vs an independent g estimate, "
                    "first-state distance to the teacher paths, persistence of the first step, branch decomposition"),
    Measurement("cosyvoice", "branches", "guidance_branches_{decoder}", CV,
                doc="first-step guidance g1 = w (v_c - v_u) per unconditional branch (all / speaker / token)"),
    Measurement("cosyvoice", "steps", "guidance_steps_{decoder}", CV,
                doc="RMS(v_c - v_u) at every evaluation for official grids, branches and first-step lengths"),
    Measurement("cosyvoice", "gamma", "guidance_decay_{decoder}", CV,
                doc="gamma(t) = RMS(v_c - v_u) on the 20-step uniform Euler path"),
    Measurement("cosyvoice", "one_step", "one_step_outputs_{decoder}", ("cosyvoice2",),
                doc="1-step outputs x1(w) vs the teacher mean and seed dependence"),
    Measurement("cosyvoice", "rho", "conditional_mean_distance_{decoder}", CV, 100, 8, writer="csv",
                doc="distance of the 1-step w = 0 output to the teacher cloud (r_c) and first-step guidance size"),
    Measurement("cosyvoice", "deviation_nn", "first_step_deviation_nearest_{decoder}", CV, 50,
                doc="first-step deviation (nearest teacher path) over the w1 grid"),
    Measurement("cosyvoice", "deviation", "first_step_deviation_{decoder}", CV, 50, writer="csv",
                doc="first-step deviation (nearest and RMS over the teacher paths) over the w1 grid"),
    Measurement("cosyvoice", "direction", "displacement_direction_{decoder}", ("cosyvoice2",), 100,
                doc="energy shares of first-step / final displacements in speaker and content bases"),
    Measurement("cosyvoice", "lipschitz", "velocity_lipschitz_{decoder}", ("cosyvoice2",), 50,
                doc="finite-difference Lipschitz ratios of v_c, v_u and the guided field on the teacher path"),
    Measurement("cosyvoice", "crossling", "crossling_guidance_{decoder}", CV,
                doc="first-step RMS(v_c - v_u) on the cross-lingual and same-language lists"),
    Measurement("indextts", "theory", "closed_form_checks_{decoder}", ("indextts2",), 50,
                doc="seed invariance of v_c - v_u at t = 0 and branch decomposition"),
    Measurement("indextts", "branches", "guidance_branches_{decoder}", ("indextts2",),
                doc="first-step guidance g1 per unconditional branch"),
    Measurement("indextts", "steps", "guidance_steps_{decoder}", ("indextts2",),
                doc="RMS(v_c - v_u) at every evaluation"),
    Measurement("indextts", "gamma", "guidance_decay_{decoder}", ("indextts2",),
                doc="gamma(t) on the 20-step uniform Euler path"),
    Measurement("indextts", "rho", "conditional_mean_distance_{decoder}", ("indextts2",), 100, 8, writer="csv",
                doc="distance of the 1-step w = 0 output to the teacher cloud"),
    Measurement("indextts", "deviation", "first_step_deviation_{decoder}", ("indextts2",), 100, writer="csv",
                doc="first-step deviation over the w1 grid (uniform 2-step grid, t1 = 0.5)"),
    Measurement("indextts", "one_step_slope", "one_step_slope_{decoder}", ("indextts2",), 100,
                doc="one-step closed form: slope of x1(w) vs an independent g estimate"),
    Measurement("glmtts", "dumps", "guidance_dumps_{decoder}", ("glmtts",), 100, writer="csv_sorted",
                doc="first-step guidance per branch, gamma(t), 1-step output vs the teacher cloud"),
    Measurement("glmtts", "deviation", "first_step_deviation_{decoder}", ("glmtts",), 50, writer="csv",
                doc="first-step deviation over the w1 grid, official (speaker-null) and all-null branches"),
    Measurement("textcond", "rho", "conditional_mean_distance_{decoder}", ("f5tts", "zipvoice"), 100, 8, writer="csv",
                doc="distance of the 1-step w = 0 output to the teacher cloud"),
    Measurement("textcond", "branches", "text_conditioned_branches_{decoder}", ("f5tts", "zipvoice"), writer="csv",
                doc="first-step guidance size per unconditional branch (all-null / text-null / audio-null)"),
    Measurement("timing", "latency", "streaming_latency_{decoder}", ("cosyvoice2",),
                doc="first-packet latency and decoder time in the official streaming pipeline"),
    Measurement("timing", "concurrency", "streaming_concurrency_{decoder}", ("cosyvoice2",),
                doc="throughput under concurrent requests: teacher, official 2 steps, FSU"),
    Measurement("timing", "concurrency_fsu_mid", "streaming_concurrency_fsu_mid_{decoder}", ("cosyvoice2",),
                doc="throughput under concurrent requests: FSU and FSU-Mid"),
]


def measurements(module):
    return {m.name: m for m in CATALOGUE if m.module == module}


def write_rows(rows, name, writer):
    """Write a list of row dicts (or {suffix: DataFrame}) to results/probes/<name><suffix>.csv."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        for suffix, df in rows.items():
            df.to_csv(RESULTS / f"{name}{suffix}.csv", index=False)
            print("wrote", RESULTS / f"{name}{suffix}.csv", flush=True)
        return
    path = RESULTS / f"{name}.csv"
    if writer == "frame":
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        fields = sorted({k for r in rows for k in r}) if writer == "csv_sorted" else list(rows[0])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print("wrote", path, len(rows), "rows", flush=True)


def parser(module):
    ms = measurements(module)
    ap = argparse.ArgumentParser(prog=f"python -m fsu.probes.{module}",
                                 description="\n".join(f"{m.name}: {m.doc}" for m in ms.values()),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measurement", choices=list(ms))
    decs = sorted({d for m in ms.values() for d in m.decoders})
    ap.add_argument("--decoder", default=decs[0], choices=decs)
    ap.add_argument("--n_utt", type=int, default=None, help="number of utterances (default: the measurement's)")
    ap.add_argument("--teacher_seeds", type=int, default=None, help="rho: number of teacher seeds (default 8)")
    ap.add_argument("--quick", action="store_true", help="timing: functionality test only, nothing is written")
    return ap


def resolve_args(module, a):
    """Measurement, utterance count, teacher count and output name of parsed arguments."""
    m = measurements(module)[a.measurement]
    if a.decoder not in m.decoders:
        raise SystemExit(f"{m.name} is defined for {m.decoders}, not {a.decoder}")
    n = a.n_utt if a.n_utt is not None else m.n_utt
    kt = a.teacher_seeds if a.teacher_seeds is not None else m.teachers
    return m, n, kt, m.out_name(a.decoder, n, kt)


def main_for(module, probe_factory, argv=None):
    """Command line of one probe module: build the probe for --decoder, run the measurement, write its CSV."""
    a = parser(module).parse_args(argv)
    m, n, kt, name = resolve_args(module, a)
    probe = probe_factory(a.decoder)
    opts = dict(n_utt=n)
    if m.teachers is not None:
        opts["teachers"] = kt
    if module == "timing":
        opts["quick"] = a.quick
    rows = getattr(probe, m.name)(**opts)
    if rows is not None:
        write_rows(rows, name, m.writer)


# ---------------------------------------------------------------- driver
JOBS = [  # (measurement module, argv): every probe behind a result of the manuscript
    *[("cosyvoice", [m, "--decoder", d]) for m in ("theory", "branches", "steps", "gamma", "rho", "deviation_nn",
                                                    "deviation", "crossling") for d in CV],
    ("cosyvoice", ["one_step", "--decoder", "cosyvoice2"]),
    ("cosyvoice", ["direction", "--decoder", "cosyvoice2"]),
    ("cosyvoice", ["lipschitz", "--decoder", "cosyvoice2"]),
    ("cosyvoice", ["rho", "--decoder", "cosyvoice2", "--n_utt", "50", "--teacher_seeds", "32"]),
    ("cosyvoice", ["rho", "--decoder", "cosyvoice3", "--n_utt", "50"]),
    *[("indextts", [m, "--decoder", "indextts2"]) for m in ("theory", "branches", "steps", "gamma", "rho", "deviation",
                                                            "one_step_slope")],
    ("indextts", ["rho", "--decoder", "indextts2", "--n_utt", "50"]),
    ("glmtts", ["dumps", "--decoder", "glmtts"]),
    ("glmtts", ["deviation", "--decoder", "glmtts"]),
    *[("textcond", [m, "--decoder", d]) for m in ("rho", "branches") for d in ("f5tts", "zipvoice")],
    *[("textcond", ["rho", "--decoder", d, "--n_utt", "50"]) for d in ("f5tts", "zipvoice")],
    # timing: run on an otherwise idle GPU, so only on request
    ("timing", ["latency", "--decoder", "cosyvoice2"]),
    ("timing", ["concurrency", "--decoder", "cosyvoice2"]),
    ("timing", ["concurrency_fsu_mid", "--decoder", "cosyvoice2"]),
]


def jobs():
    out = []
    for module, argv in JOBS:
        a = parser(module).parse_args(argv)
        m, _, _, name = resolve_args(module, a)
        out.append((name, module, argv, a.decoder))
    return out


def run_jobs(selection):
    """Run the selected jobs one after another in their decoders' environments (skip-existing on the output CSV).
    IndexTTS-2 and GLM-TTS take the heavy_gen lock and wait for a running evaluation job, as the generation queue."""
    import contextlib
    import subprocess
    import time
    from fsu.common import GpuLock, env_python, subprocess_env
    from fsu.experiment import DECODERS
    (LOGS / "probes").mkdir(parents=True, exist_ok=True)
    for name, module, argv, dec in jobs():
        if selection:
            if not any(name.startswith(s) for s in selection):
                continue
        elif module == "timing":
            continue
        if (RESULTS / f"{name}.csv").exists():
            continue
        spec = DECODERS[dec]
        extra = dict(PYTHONPATH=str(REPOS / "ZipVoice")) if dec == "zipvoice" else {}
        t0 = time.time()
        with GpuLock("heavy_gen") if spec.heavy else contextlib.nullcontext():
            if spec.heavy:
                GpuLock.wait_free("eval_job")
            with open(LOGS / "probes" / f"{name}.log", "w", encoding="utf-8") as f:
                rc = subprocess.call([env_python(spec.env), "-m", f"fsu.probes.{module}", *argv], cwd=ROOT,
                                     env=subprocess_env(**extra), stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {name} rc={rc} {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fsu.probes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "run"])
    ap.add_argument("names", nargs="*", help="run: output names or prefixes (default: every probe except timing)")
    a = ap.parse_args(argv)
    if a.command == "list":
        for name, module, argv_, _ in jobs():
            state = "done" if (RESULTS / f"{name}.csv").exists() else "-"
            print(f"{name:52s} {state:5s} python -m fsu.probes.{module} {' '.join(argv_)}")
    else:
        run_jobs(a.names)


if __name__ == "__main__":
    sys.exit(main())
