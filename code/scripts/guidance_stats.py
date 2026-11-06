"""Shared summary of first-step guidance vectors g1 = w (v_c - v_u) at t=0 (E8'-2 / E8'-8 block D).
Per utterance, generated region (mel C x T):
  E_tot   = ||g||^2 / (C T);   norm = ||g|| / sqrt(C T)  (RMS, used for E1 = w-scaled RMS x Δt1)
  timbre  = share of ||g||^2 in the frame-mean component (per-bin mean over time) -> 'global spectrum' axis
  temporal= 1 - timbre (frame-to-frame varying part) -> 'content / rhythm' axis
  cos_ab  = cosine between branch vectors on the same utterance / noise
"""
import numpy as np


def summarise(utt, G):
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
