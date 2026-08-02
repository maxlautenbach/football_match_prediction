"""Kicktipp score decoding helpers (Poisson mode / expected-points)."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def poisson_mode_score(lam_home: float, lam_away: float, goal_cap: int) -> Tuple[int, int]:
    lam_home = float(max(lam_home, 1e-8))
    lam_away = float(max(lam_away, 1e-8))

    ks = np.arange(goal_cap + 1)
    log_fact = np.array([math.lgamma(k + 1) for k in ks])

    logp_h = -lam_home + ks * math.log(lam_home) - log_fact
    logp_a = -lam_away + ks * math.log(lam_away) - log_fact

    grid = logp_h.reshape(-1, 1) + logp_a.reshape(1, -1)
    idx = int(np.argmax(grid))
    h = idx // (goal_cap + 1)
    a = idx % (goal_cap + 1)
    return int(h), int(a)


def build_kicktipp_points_matrix(goal_cap: int) -> np.ndarray:
    k = goal_cap + 1
    n = k * k
    pts = np.zeros((n, n), dtype=np.float32)

    def outcome(h: int, a: int) -> int:
        return 1 if h > a else (0 if h == a else -1)

    for ph in range(k):
        for pa in range(k):
            p_idx = ph * k + pa
            p_diff = ph - pa
            p_out = outcome(ph, pa)
            for th in range(k):
                for ta in range(k):
                    t_idx = th * k + ta
                    if ph == th and pa == ta:
                        pts[p_idx, t_idx] = 5.0
                    elif p_diff == (th - ta):
                        pts[p_idx, t_idx] = 3.0
                    elif p_out == outcome(th, ta):
                        pts[p_idx, t_idx] = 1.0
    return pts


def poisson_probs(lam: float, goal_cap: int) -> np.ndarray:
    lam = float(max(lam, 1e-8))
    ks = np.arange(goal_cap + 1)
    log_fact = np.array([math.lgamma(k + 1) for k in ks])
    logp = -lam + ks * math.log(lam) - log_fact
    m = float(np.max(logp))
    p = np.exp(logp - m)
    p = p / float(np.sum(p))
    return p


def kicktipp_optimal_score_from_joint(
    joint_probs: np.ndarray,
    goal_cap: int,
    points_matrix: np.ndarray,
) -> Tuple[int, int]:
    """Score maximizing expected Kicktipp points under an arbitrary joint distribution.

    `joint_probs` is a flattened (goal_cap+1)^2 grid indexed by home*k + away.
    """
    exp_pts = points_matrix @ np.asarray(joint_probs, dtype=np.float64).reshape(-1)
    idx = int(np.argmax(exp_pts))
    k = goal_cap + 1
    return idx // k, idx % k


def kicktipp_optimal_score(
    lam_home: float,
    lam_away: float,
    goal_cap: int,
    points_matrix: np.ndarray,
) -> Tuple[int, int]:
    p_home = poisson_probs(lam_home, goal_cap)
    p_away = poisson_probs(lam_away, goal_cap)
    p_true = np.outer(p_home, p_away).reshape(-1)
    return kicktipp_optimal_score_from_joint(p_true, goal_cap, points_matrix)
