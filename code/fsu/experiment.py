"""Experiments: decoders, sampling methods and runs, plus the generation and evaluation queues.

An experiment file (configs/experiments/<name>.yaml) lists groups of runs:

    description: ...
    runs:
      - decoders: [cosyvoice2, cosyvoice3]
        sets: [seedtts_en_subset, seedtts_zh_subset]
        methods: [official_n2, fsu_n2, fsu_mid, teacher]     # names from configs/methods.yaml ...
        seeds: [0]
      - decoders: [cosyvoice2]
        sets: [seedtts_en_subset100]
        methods: {w1_-0.20: [--nfe, 2, --cfg_steps=-0.2,0.7]} # ... or defined inline
        seeds: [0, 1, 2, 3, 4, 5, 6, 7]
        metrics: [wavlm, ecapa, eres2net]                     # optional, default depends on seed and language

A run is one (decoder, test set, method, seed). Its output directory depends only on these, so experiments that share a
run share its files:  out/<decoder>/<set>/<method>/seed<k>/<utt>.wav  (+ _metrics/ after evaluation).
Completion markers: _done (all wavs generated), _evaluated (all metrics computed).
"""
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fsu import datasets
from fsu.common import CACHE, CKPT, LOGS, OUT, REPOS, ROOT, GpuLock, env_python, subprocess_env

CONFIGS = ROOT / "configs"


# ---------------------------------------------------------------- decoders
class DecoderSpec:
    """How to run one decoder: conda environment, runner command, token cache and official settings."""
    name = ""
    env = ""
    teacher_steps = 10
    heavy = False  # does not fit on a 16 GB GPU next to an ASR evaluation job

    def token_cache(self, test_set):
        return CACHE / "tokens" / self.name / datasets.get(test_set).token_group

    def prepare_command(self, test_set):
        """Command that caches the conditions of `test_set` (None if the decoder has no separate stage)."""
        return None

    def decode_command(self, test_set, args, seed, out_dir):
        raise NotImplementedError

    def cwd_env(self):
        return ROOT, subprocess_env()


class CosyVoiceSpec(DecoderSpec):
    env = "env_cosy"
    model_dir = ""
    prompt_prefix = ""

    def prepare_command(self, test_set):
        ts = datasets.get(test_set)
        cmd = ["-m", "fsu.decoders.cosyvoice", "--model_dir", self.model_dir, "--meta", rel(ts.path), "--stage", "tokens",
               "--cache_dir", rel(self.token_cache(test_set))]
        if self.prompt_prefix:
            cmd += ["--prompt_prefix", self.prompt_prefix]
        if not ts.text_frontend:
            cmd += ["--text_frontend", "0"]
        return cmd

    def decode_command(self, test_set, args, seed, out_dir):
        return ["-m", "fsu.decoders.cosyvoice", "--model_dir", self.model_dir, "--meta", rel(datasets.get(test_set).path),
                "--stage", "decode", "--cache_dir", rel(self.token_cache(test_set)), "--noise", f"utt:{seed}",
                "--out_dir", rel(out_dir), *args]


class CosyVoice2(CosyVoiceSpec):
    name, model_dir = "cosyvoice2", "ckpt/CosyVoice2-0.5B"


class CosyVoice3(CosyVoiceSpec):
    name, model_dir = "cosyvoice3", "ckpt/Fun-CosyVoice3-0.5B"
    prompt_prefix = "You are a helpful assistant.<|endofprompt|>"


class IndexTTS2Spec(DecoderSpec):
    name, env, teacher_steps, heavy = "indextts2", "env_index", 25, True

    def prepare_command(self, test_set):
        ts = datasets.get(test_set)  # the official 25-step output of the cond stage is kept next to the cache
        return ["-m", "fsu.decoders.indextts", "--meta", rel(ts.path), "--stage", "cond",
                "--cache_dir", rel(self.token_cache(test_set)),
                "--out_dir", rel(OUT / self.name / ts.token_group / "official_pipeline")]

    def decode_command(self, test_set, args, seed, out_dir):
        return ["-m", "fsu.decoders.indextts", "--meta", rel(datasets.get(test_set).path), "--stage", "decode",
                "--cache_dir", rel(self.token_cache(test_set)), "--out_dir", rel(out_dir), "--noise", f"utt:{seed}", *args]


class GLMTTSSpec(DecoderSpec):
    name, env, heavy = "glmtts", "env_glm", True

    def prepare_command(self, test_set):
        ts = datasets.get(test_set)
        cmd = ["-m", "fsu.decoders.glmtts", "--meta", rel(ts.path), "--stage", "tokens",
               "--cache_dir", rel(self.token_cache(test_set))]
        return cmd + ([] if ts.text_frontend else ["--text_frontend", "0"])

    def decode_command(self, test_set, args, seed, out_dir):
        return ["-m", "fsu.decoders.glmtts", "--meta", rel(datasets.get(test_set).path), "--stage", "decode",
                "--cache_dir", rel(self.token_cache(test_set)), "--noise", f"utt:{seed}", "--out_dir", rel(out_dir), *args]


class F5TTSSpec(DecoderSpec):
    """F5-TTS v0 base model (cfg 2, sway -1, EPSS off so every step count uses the uniform grid with sway)."""
    name, env, teacher_steps = "f5tts", "env_f5dmo", 32

    def decode_command(self, test_set, args, seed, out_dir):
        return ["-m", "fsu.decoders.f5tts", "f5", "--exp", "F5TTS_Base",
                "--ckpt", "ckpt/F5-TTS/F5TTS_Base/model_1200000.safetensors", "--meta", rel(datasets.get(test_set).path),
                "--out_dir", rel(out_dir), *args, "--noise", f"utt:{seed}"]


class DMOSpeech2Spec(DecoderSpec):
    name, env, teacher_steps = "dmospeech2", "env_f5dmo", 32

    def decode_command(self, test_set, args, seed, out_dir):
        return ["-m", "fsu.decoders.f5tts", "dmo", "--meta", str(datasets.get(test_set).path), "--out_dir", str(out_dir),
                *args, "--noise", f"utt:{seed}"]


class ZipVoiceSpec(DecoderSpec):
    """ZipVoice through its own command line (repos/ZipVoice/zipvoice/bin/infer_zipvoice.py + patches/zipvoice.diff),
    raw evaluation mode, per-utterance noise seeds."""
    name, env, teacher_steps, model_name = "zipvoice", "env_zip", 16, "zipvoice"

    def decode_command(self, test_set, args, seed, out_dir):
        return ["-m", "zipvoice.bin.infer_zipvoice", "--model-name", self.model_name,
                "--model-dir", str(CKPT / "ZipVoice" / self.model_name), "--vocoder-path", str(CKPT / "vocos-mel-24khz"),
                "--test-list", str(datasets.get(test_set).zipvoice_path), "--res-dir", str(out_dir),
                "--raw-evaluation", "True", *args, "--seed-per-utt", "True", "--noise-key", str(seed),
                "--skip-existing", "True"]

    def cwd_env(self):
        z = REPOS / "ZipVoice"
        return z, subprocess_env(PYTHONPATH=str(z))


class ZipVoiceDistillSpec(ZipVoiceSpec):
    name, teacher_steps, model_name = "zipvoice_distill", 8, "zipvoice_distill"


DECODERS = {c.name: c() for c in (CosyVoice2, CosyVoice3, IndexTTS2Spec, GLMTTSSpec, F5TTSSpec, DMOSpeech2Spec,
                                  ZipVoiceSpec, ZipVoiceDistillSpec)}


def rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- methods and runs
def load_methods():
    """configs/methods.yaml: name -> runner arguments, or {decoder: arguments} for decoder-specific settings."""
    return yaml.safe_load((CONFIGS / "methods.yaml").read_text(encoding="utf-8"))


def method_args(spec, decoder):
    if isinstance(spec, dict):
        if decoder not in spec:
            raise KeyError(f"method not defined for {decoder}: {spec}")
        spec = spec[decoder]
    return [str(x) for x in spec]


@dataclass(frozen=True)
class Run:
    decoder: str
    test_set: str
    method: str
    seed: int
    args: tuple = field(compare=False)
    metrics: tuple = field(default=None, compare=False)
    quality: tuple = field(default=(), compare=False)  # objective quality predictors (fsu.quality), if any

    @property
    def spec(self):
        return DECODERS[self.decoder]

    @property
    def dir(self):
        return OUT / self.decoder / self.test_set / self.method / f"seed{self.seed}"

    @property
    def lang(self):
        return datasets.get(self.test_set).lang

    def command(self):
        return self.spec.decode_command(self.test_set, list(self.args), self.seed, self.dir)

    def metric_list(self):
        if self.metrics:
            return ",".join(self.metrics)
        if self.seed > 0:
            return "wavlm,ecapa,eres2net"
        m = "wer,wavlm,ecapa,eres2net,utmos"
        return m + (",cer2" if self.lang in ("ko", "ja") else "")

    def check_args(self):
        """Refuse to mix two different argument lists in one output directory."""
        f = self.dir / "_args.json"
        cur = dict(decoder=self.decoder, test_set=self.test_set, args=list(self.args), seed=self.seed)
        if f.exists():
            old = json.loads(f.read_text(encoding="utf-8"))
            assert old == cur, f"{self.dir}: arguments changed\n  was {old}\n  now {cur}"
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(cur, indent=1), encoding="utf-8")

    def n_expected(self):
        return len(datasets.get(self.test_set))

    def n_done(self):
        return len(list(self.dir.glob("*.wav")))


class Experiment:
    """The runs of one experiment file."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            self.path = CONFIGS / "experiments" / f"{path}.yaml"
        cfg = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.name = self.path.stem
        self.description = cfg.get("description", "")
        methods = load_methods()
        runs = {}
        for g in cfg["runs"]:
            group_methods = g["methods"]
            if isinstance(group_methods, list):
                group_methods = {m: methods[m] for m in group_methods}
            extra = [str(x) for x in g.get("extra", [])]  # appended to every method of the group
            suffix = g.get("suffix", "")  # e.g. _allnull for the GLM-TTS all-null branch
            for dec in g["decoders"]:
                for ts in g["sets"]:
                    for m, spec in group_methods.items():
                        for k in g.get("seeds", [0]):
                            r = Run(dec, ts, m + suffix, int(k), tuple(method_args(spec, dec) + extra),
                                    tuple(g["metrics"]) if g.get("metrics") else None, tuple(g.get("quality", ())))
                            runs.setdefault(r, r)
        self.runs = list(runs)

    def prepare_jobs(self):
        """(decoder, test set) pairs whose conditions must be cached before decoding."""
        seen = []
        for r in self.runs:
            if (r.decoder, r.test_set) not in seen and r.spec.prepare_command(r.test_set):
                seen.append((r.decoder, r.test_set))
        return seen


# ---------------------------------------------------------------- queues
def _call(env, cmd, cwd, env_vars, log):
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.call([env_python(env), *cmd], cwd=cwd, env=env_vars, stdout=f, stderr=subprocess.STDOUT)


def _stamp(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def generate(exp, only=None):
    """Cache the conditions, then decode every run (seed-major order). Resumable: finished runs have a _done marker and
    the runners skip existing wavs."""
    logs = LOGS / exp.name
    logs.mkdir(parents=True, exist_ok=True)
    for dec, ts in exp.prepare_jobs():
        spec = DECODERS[dec]
        mark = spec.token_cache(ts) / "_done"
        if mark.exists():
            continue
        cwd, env = spec.cwd_env()
        with (GpuLock("heavy_gen") if spec.heavy else _Null()):
            if spec.heavy:
                GpuLock.wait_free("eval_job")
            rc = _call(spec.env, spec.prepare_command(ts), cwd, env, logs / f"prepare_{dec}_{ts}.log")
        _stamp(f"prepare {dec} {ts} rc={rc}")
        if rc == 0:
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.write_text("ok")
    for r in sorted(exp.runs, key=lambda r: r.seed):
        if only and not any(r.method.startswith(p) or r.decoder == p for p in only):
            continue
        if (r.dir / "_done").exists():
            continue
        r.check_args()
        cwd, env = r.spec.cwd_env()
        t0 = time.time()
        with (GpuLock("heavy_gen") if r.spec.heavy else _Null()):
            if r.spec.heavy:  # announce first, then wait for a running evaluation to finish
                GpuLock.wait_free("eval_job")
            rc = _call(r.spec.env, r.command(), cwd, env, logs / f"gen_{r.decoder}_{r.test_set}_{r.method}_seed{r.seed}.log")
        n = r.n_done()
        _stamp(f"{r.decoder} {r.test_set} {r.method} seed{r.seed} rc={rc} {n} wavs {time.time() - t0:.0f}s")
        if rc == 0 and n == r.n_expected():
            (r.dir / "_done").write_text("ok")


def evaluate(exp, wait=True, only=None):
    """Score every generated run (metrics + prosody features). Runs alongside generate(); one evaluation job at a time.
    With wait=True it keeps polling until every run of the experiment is evaluated."""
    logs = LOGS / exp.name
    logs.mkdir(parents=True, exist_ok=True)
    fails = {}
    while True:
        todo = [r for r in exp.runs if not (r.dir / "_evaluated").exists() and fails.get(r, 0) < 2
                and (not only or any(r.method.startswith(p) or r.decoder == p for p in only))]
        if not todo:
            _stamp(f"all evaluated {fails or ''}")
            return
        ready = [r for r in todo if (r.dir / "_done").exists()]
        if not ready:
            if not wait:
                return
            time.sleep(60)
            continue
        r = ready[0]
        ts = datasets.get(r.test_set)
        GpuLock.wait_free("heavy_gen")
        log = logs / f"eval_{r.decoder}_{r.test_set}_{r.method}_seed{r.seed}.log"
        with GpuLock("eval_job"):
            rc1 = _call("env_eval", ["-m", "fsu.metrics", "--meta", rel(ts.eval_path), "--wav_dir", rel(r.dir),
                                     "--lang", ts.lang, "--metrics", r.metric_list()], ROOT, subprocess_env(), log)
            rc2 = _call("env_eval", ["-m", "fsu.metrics", "--acoustic", "--meta", rel(ts.path), "--wav_dir", rel(r.dir),
                                     "--lang", ts.lang], ROOT, subprocess_env(), log)
            rc3 = _call("env_objq", ["-m", "fsu.quality", "--meta", rel(ts.path), "--wav_dir", rel(r.dir),
                                     "--metrics", ",".join(r.quality)], ROOT, subprocess_env(), log) if r.quality else 0
        _stamp(f"eval {r.decoder} {r.test_set} {r.method} seed{r.seed} rc={rc1},{rc2},{rc3}")
        if rc1 == 0 and rc2 == 0 and rc3 == 0:
            (r.dir / "_evaluated").write_text("ok")
        else:
            fails[r] = fails.get(r, 0) + 1


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
