"""TV-CFG per-step guidance scales for the IndexTTS-2 / GLM-TTS runners (E14R-2, v10.7): run_cosy.tv_scales copied
verbatim (run_cosy imports the CosyVoice frontend at module level, which the other envs do not have)."""
import numpy as np


def tv_scales(t_span, w, kind):
    """E8'-10 TV-CFG (2509.22007), sqt24/tvcfg guidance_schedulers.py transcribed. Guidance scale g = 1 + w.
    timesteps normalised by their max = sigma_i = 1 - t_i at each step start (sigma_0 = 1); delta_i = sigma_i - sigma_{i+1}
    with a final 0.0 appended (official). 'symmetric' = symmetric(low=1, high=2g-1, normalize=True) (generate.py call);
    'stepup' = stepup(low=g/3, high=g, tau=0.25, normalize=False). '*_fix' (our variant, appendix): last delta = sigma_{N-1}
    (the official final 0.0 drops the last step from the normalisation, which inflates few-step scales)."""
    g = 1.0 + w
    sig = np.array([1.0 - t for t in t_span[:-1]])
    n = len(sig)
    delta = np.append(sig[:-1] - sig[1:], sig[-1] if kind.endswith("_fix") else 0.0)
    base = kind.replace("_fix", "")
    if base == "symmetric":
        low, high, half = 1.0, 2 * g - 1.0, n // 2
        if n % 2 == 0:
            s = np.concatenate([np.linspace(low, high, half, endpoint=False), np.linspace(high, low, half, endpoint=True)])
        else:
            s = np.concatenate([np.linspace(low, high, half + 1, endpoint=True), np.linspace(high, low, half + 1, endpoint=True)[1:]])
        return (s * g / np.sum(s * delta)).tolist()
    if base == "stepup":
        steps = np.linspace(0, 1, n)
        return np.where(steps < 0.25, g / 3, g).tolist()
    raise ValueError(kind)
