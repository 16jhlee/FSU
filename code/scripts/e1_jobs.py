"""E1 (NFE sweep) + CosyVoice3 full-test-en job list, shared by gen_queue.py and eval_queue.py.

Each job = one (model, setting) over the three subset lists (en 300 / zh 300 / hard 100).
Output: out/E1/<family>/<model>/nfe<N>/cfg<w>/<noise>/<set>/<utt_id>.wav
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_python(name):
    """Python of conda env `name`. Override the envs directory with SPEECHFM_CONDA_ENVS; otherwise it is
    derived from the running interpreter (expected to be the `tools` env: <envs>/tools/python[.exe])."""
    envs = os.environ.get("SPEECHFM_CONDA_ENVS")
    if envs is None:
        exe = Path(sys.executable).resolve()
        envs = exe.parents[1] if os.name == "nt" else exe.parents[2]
    p = Path(envs) / name / ("python.exe" if os.name == "nt" else "bin/python")
    return str(p)


ENVS = {k: _env_python(k) for k in ["env_cosy", "env_zip", "env_f5dmo", "env_eval"]}
SETS = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh"), ("hard", "data/lists/hard_all", "zh")]
NOISE = "utt:0"


def jobs():
    J = []

    def add(name, env, cmd_fn, out, sets=SETS, cwd=ROOT, extra_env=None, pre=None):
        J.append(dict(name=name, env=env, cmd_fn=cmd_fn, out=out, sets=sets, cwd=cwd,
                      extra_env=extra_env or {}, pre=pre))

    # ---- decision 1b: Fun-CosyVoice3 on full test-en (1,088), official settings ----
    full = [("en", "data/lists_full/en_all", "en")]
    pre3 = "You are a helpful assistant.<|endofprompt|>"
    add("cv3_full_tokens", "env_cosy",
        lambda s, m, o: ["scripts/run_cosy.py", "--model_dir", "ckpt/Fun-CosyVoice3-0.5B", "--meta", m + ".meta.lst",
                         "--stage", "tokens", "--cache_dir", f"cache/cosy3_tokens/{s}_full", "--prompt_prefix", pre3],
        out=None, sets=full)
    add("cv3_full_nfe10", "env_cosy",
        lambda s, m, o: ["scripts/run_cosy.py", "--model_dir", "ckpt/Fun-CosyVoice3-0.5B", "--meta", m + ".meta.lst",
                         "--stage", "decode", "--cache_dir", f"cache/cosy3_tokens/{s}_full", "--out_dir", o,
                         "--nfe", "10", "--cfg", "0.7", "--noise", "default"],
        out="out/E0/A/cosyvoice3_full/nfe10/cfg0.7/default", sets=full)

    # ---- A: CosyVoice2, NFE {1,2,3,5,10}, cfg 0.7, per-utt noise ----
    add("cv2_tokens", "env_cosy",
        lambda s, m, o: ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst",
                         "--stage", "tokens", "--cache_dir", f"cache/cosy2_tokens/{s}"], out=None)
    for n in [10, 5, 3, 2, 1]:
        add(f"cv2_nfe{n}", "env_cosy",
            lambda s, m, o, n=n: ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst",
                                  "--stage", "decode", "--cache_dir", f"cache/cosy2_tokens/{s}", "--out_dir", o,
                                  "--nfe", str(n), "--cfg", "0.7", "--noise", NOISE],
            out=f"out/E1/A/cosyvoice2/nfe{n}/cfg0.7/{NOISE.replace(':', '')}")

    # ---- B: ZipVoice teacher {1,2,4,8,16} w1.0; distill {1,2,4,8} w3.0 (raw evaluation) ----
    zipdir = ROOT / "repos" / "ZipVoice"
    for model, nfes, w in [("zipvoice", [16, 8, 4, 2, 1], "1.0"), ("zipvoice_distill", [8, 4, 2, 1], "3.0")]:
        for n in nfes:
            add(f"{model}_nfe{n}", "env_zip",
                lambda s, m, o, model=model, n=n, w=w: [
                    "-m", "zipvoice.bin.infer_zipvoice", "--model-name", model,
                    "--model-dir", str(ROOT / "ckpt" / "ZipVoice" / model),
                    "--vocoder-path", str(ROOT / "ckpt" / "vocos-mel-24khz"),
                    "--test-list", str(ROOT / (m + ".zipvoice.tsv")), "--res-dir", str(ROOT / o),
                    "--raw-evaluation", "True", "--num-step", str(n), "--guidance-scale", w,
                    "--seed-per-utt", "True", "--noise-key", "0", "--skip-existing", "True"],
                out=f"out/E1/B/{model}/nfe{n}/cfg{w}/{NOISE.replace(':', '')}", cwd=zipdir,
                extra_env={"PYTHONPATH": str(zipdir)})

    # ---- C main: DMOSpeech2 internal teacher {1,2,4,8,16,32} (cfg2, sway-1, eta0) ----
    for n in [32, 16, 8, 4, 2, 1]:
        add(f"dmo_teacher_nfe{n}", "env_f5dmo",
            lambda s, m, o, n=n: ["scripts/run_dmo.py", "--meta", str(ROOT / (m + ".meta.lst")), "--out_dir",
                                  str(ROOT / o), "--mode", "teacher", "--teacher_steps", str(n), "--noise", NOISE],
            out=f"out/E1/C/dmo_teacher/nfe{n}/cfg2.0/{NOISE.replace(':', '')}")
    # DMOSpeech2 student-only, grids per decision 4
    grids = {4: "0,0.25,0.5,0.75", 3: "0,0.25,0.5", 2: "0,0.5", 1: "0"}
    for n, g in grids.items():
        add(f"dmo_student_nfe{n}", "env_f5dmo",
            lambda s, m, o, g=g: ["scripts/run_dmo.py", "--meta", str(ROOT / (m + ".meta.lst")), "--out_dir",
                                  str(ROOT / o), "--mode", "student", "--grid", g, "--noise", NOISE],
            out=f"out/E1/C/dmo_student/nfe{n}/cfgfused/{NOISE.replace(':', '')}")
    add("dmo_official_tg", "env_f5dmo",
        lambda s, m, o: ["scripts/run_dmo.py", "--meta", str(ROOT / (m + ".meta.lst")), "--out_dir", str(ROOT / o),
                         "--mode", "official", "--noise", NOISE],
        out=f"out/E1/C/dmo_official_tg/nfe16/cfg2.0/{NOISE.replace(':', '')}")

    # ---- C secondary: public F5-TTS v0 {1,2,4,8,16,32}, cfg2, sway-1, EPSS off (uniform+sway at all NFE) ----
    for n in [32, 16, 8, 4, 2, 1]:
        add(f"f5v0_nfe{n}", "env_f5dmo",
            lambda s, m, o, n=n: ["scripts/run_f5.py", "--exp", "F5TTS_Base",
                                  "--ckpt", "ckpt/F5-TTS/F5TTS_Base/model_1200000.safetensors",
                                  "--meta", m + ".meta.lst", "--out_dir", o, "--nfe", str(n), "--cfg", "2.0",
                                  "--sway", "-1", "--use_epss", "0", "--noise", NOISE],
            out=f"out/E1/C/f5tts_v0/nfe{n}/cfg2.0/{NOISE.replace(':', '')}")
    return J


def n_lines(meta):
    return sum(1 for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip())
