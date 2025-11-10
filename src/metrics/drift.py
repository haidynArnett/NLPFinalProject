# src/metrics/drift.py
# Metrics for measuring information drift in agent handoffs
# Includes drift velocity, acceleration, and similarity tracking

from __future__ import annotations
from typing import Iterable
import numpy as np

def velocity(series: Iterable[float]) -> np.ndarray:
    """
    First difference: v[t] = s[t] - s[t-1]
    Input:  s[0..N-1]
    Output: v[1..N-1]
    """
    s = np.asarray(series, dtype=float)
    if s.size < 2:
        return np.zeros(0, dtype=float)
    return np.diff(s, n=1)

def acceleration(series: Iterable[float]) -> np.ndarray:
    """
    Second difference: a[t] = v[t] - v[t-1]
    Input:  s[0..N-1]
    Output: a[2..N-1]
    """
    s = np.asarray(series, dtype=float)
    if s.size < 3:
        return np.zeros(0, dtype=float)
    return np.diff(s, n=2)

def normalize_0_1(series: Iterable[float]) -> np.ndarray:
    """
    Min-max normalize to [0,1]. If constant, returns zeros.
    """
    s = np.asarray(series, dtype=float)
    lo, hi = np.min(s), np.max(s)
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)

def ema(series: Iterable[float], alpha: float = 0.3) -> np.ndarray:
    """
    Exponential Moving Average (smoothing). 0<alpha<=1 (higher = less smoothing).
    """
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return s
    out = np.empty_like(s)
    out[0] = s[0]
    for i in range(1, s.size):
        out[i] = alpha * s[i] + (1 - alpha) * out[i-1]
    return out

from .similarity import compute_cosine_similarity_matrix

# gets similarity to the original at each hop
def sim_to_original(restatements,embed):
    S = compute_cosine_similarity_matrix(restatements, embed)
    similarities= []
    for j in range(1, S.shape[0]):
        value = float(S[0, j])
        similarities.append(value)
    return similarities

# similarity between each hop
def step_similarity(restatements,embed):
    n = len(restatements)
    sims = []
    for i in range(n - 1):
        pair = [restatements[i], restatements[i + 1]]
        S_pair = compute_cosine_similarity_matrix(pair, embed)
        value = float(S_pair[0, 1])
        sims.append(value)
    return sims


def drift_velocity(similarities):
    velocities = []
    for i in range(1, len(similarities)):
        v = similarities[i] - similarities[i - 1]
        velocities.append(float(v))
    return velocities


def drift_acceleration(velocities):
    accelerations = []
    for i in range(1, len(velocities)):
        a = velocities[i] - velocities[i - 1]
        accelerations.append(float(a))
    return accelerations