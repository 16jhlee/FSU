"""E1 figures: per test set, SIM (3 embedders) + WER vs NFE, mean with bootstrap 95% CI band.
Teachers / plain step reduction dashed, distilled students solid."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# fixed categorical order (validated default palette, light mode)
SERIES = [  # (label, job prefix, color, dashed)
    ("A CosyVoice2 (step reduction)", "cv2_nfe", "#2a78d6", True),
    ("B ZipVoice teacher (step reduction)", "zipvoice_nfe", "#eb6834", True),
    ("B ZipVoice-Distill (student)", "zipvoice_distill_nfe", "#1baf7a", False),
    ("C DMOSpeech2 teacher (step reduction)", "dmo_teacher_nfe", "#eda100", True),
    ("C DMOSpeech2 student (DMD+SV)", "dmo_student_nfe", "#e87ba4", False),
    ("C F5-TTS v0 public (step reduction)", "f5v0_nfe", "#008300", True),
]
import sys  # noqa: E402
# main embedder set after E1b (CAM++ zh-cn moved to appendix: domain mismatch + CosyVoice cond. encoder)
PANELS = [("sim_wavlm", "SIM-o · WavLM-SV"), ("sim_ecapa", "SIM-o · ECAPA (VoxCeleb)"),
          ("sim_eres2net", "SIM-o · ERes2Net-en (VoxCeleb2)"), ("wer", "WER / CER (%)")]
if "--appendix" in sys.argv:
    PANELS = [("sim_campp", "SIM-o · CAM++ zh-cn (appendix)"), ("simr_wavlm", "SIM-r · WavLM-SV"),
              ("simr_ecapa", "SIM-r · ECAPA"), ("simr_eres2net", "SIM-r · ERes2Net-en")]
rng = np.random.default_rng(0)


def ci(x):
    bs = rng.choice(x, (5000, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


def main():
    d = pd.read_csv(ROOT / "results" / "E1_long.csv")
    out = ROOT / "reports" / "fig"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF,
                         "axes.facecolor": SURF})
    for s in ["en", "zh", "hard"]:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), constrained_layout=True)
        for ax, (m, title) in zip(axes, PANELS):
            for lab, pre, col, dashed in SERIES:
                g = d[(d.set == s) & d.job.str.startswith(pre)]
                if pre == "zipvoice_nfe":
                    g = g[~g.job.str.startswith("zipvoice_distill")]
                if not len(g):
                    continue
                g = g.assign(nfe=g.job.str.extract(r"nfe(\d+)$")[0].astype(int))
                pts = sorted(g.nfe.unique())
                st = np.array([ci(g[g.nfe == n][m].values * (100 if m == "wer" else 1)) for n in pts])
                ax.fill_between(pts, st[:, 1], st[:, 2], color=col, alpha=0.15, linewidth=0)
                ax.plot(pts, st[:, 0], color=col, lw=2, ls="--" if dashed else "-", marker="o", ms=4.5,
                        markeredgecolor=SURF, markeredgewidth=1.2, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32])
            ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
            ax.set_xlabel("NFE")
            ax.set_title(title, loc="left", fontsize=10)
            ax.grid(True, color=GRID, lw=0.8)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            if m == "wer":
                ax.set_yscale("symlog", linthresh=10)
                ax.set_ylim(0, 200)
            else:
                ax.set_ylim(0.0, 0.9)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
        fig.suptitle(f"E1 NFE sweep — Seed-TTS {s} (subset n={'300' if s != 'hard' else '100'}; "
                     f"mean, bootstrap 95% CI; dashed = teacher/step reduction)", x=0.01, ha="left",
                     fontsize=11, color=INK)
        name = f"E1_{s}{'_appendix' if '--appendix' in sys.argv else ''}.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        print("saved", out / name)


if __name__ == "__main__":
    main()
