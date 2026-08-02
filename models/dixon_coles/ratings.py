"""Dixon-Coles maximum-likelihood ratings and score-grid probabilities.

Goal counts are modelled as Poisson with team attack/defence strengths::

    log lambda_home = intercept + home_advantage + attack[home] - defence[away]
    log mu_away     = intercept                  + attack[away] - defence[home]

plus the Dixon-Coles ``tau`` correction, which re-weights the four low-score
cells where independent Poisson demonstrably misfits football results.
Matches are weighted by exponential time decay so recent seasons dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from models.common.kicktipp import poisson_probs

# tau can turn non-positive for extreme lambda/rho combinations during the
# optimizer's line search; floor it and drop its gradient there.
TAU_FLOOR = 1e-8

DEFAULT_RHO_BOUNDS = (-0.35, 0.35)
_STRENGTH_BOUND = 3.0
_INTERCEPT_BOUNDS = (-2.0, 2.0)
_HOME_ADV_BOUNDS = (-1.0, 1.0)


@dataclass(frozen=True)
class DixonColesFit:
    attack: np.ndarray
    defence: np.ndarray
    intercept: float
    home_advantage: float
    rho: float
    log_likelihood: float
    n_matches: int
    weight_sum: float
    converged: bool

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate(
            [
                self.attack,
                self.defence,
                [self.intercept, self.home_advantage, self.rho],
            ]
        )


def _tau_terms(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dixon-Coles low-score correction and its partial derivatives."""
    n = lam.shape[0]
    tau = np.ones(n, dtype=np.float64)
    d_lam = np.zeros(n, dtype=np.float64)
    d_mu = np.zeros(n, dtype=np.float64)
    d_rho = np.zeros(n, dtype=np.float64)

    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    d_lam[m00] = -mu[m00] * rho
    d_mu[m00] = -lam[m00] * rho
    d_rho[m00] = -lam[m00] * mu[m00]

    tau[m01] = 1.0 + lam[m01] * rho
    d_lam[m01] = rho
    d_rho[m01] = lam[m01]

    tau[m10] = 1.0 + mu[m10] * rho
    d_mu[m10] = rho
    d_rho[m10] = mu[m10]

    tau[m11] = 1.0 - rho
    d_rho[m11] = -1.0

    floored = tau < TAU_FLOOR
    if floored.any():
        tau = np.where(floored, TAU_FLOOR, tau)
        d_lam = np.where(floored, 0.0, d_lam)
        d_mu = np.where(floored, 0.0, d_mu)
        d_rho = np.where(floored, 0.0, d_rho)

    return tau, d_lam, d_mu, d_rho


def _objective(
    theta: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    l2: float,
) -> Tuple[float, np.ndarray]:
    """Weighted negative log-likelihood (constants dropped) and its gradient."""
    attack = theta[:n_teams]
    defence = theta[n_teams : 2 * n_teams]
    intercept = theta[2 * n_teams]
    home_adv = theta[2 * n_teams + 1]
    rho = theta[2 * n_teams + 2]

    log_lam = intercept + home_adv + attack[home_idx] - defence[away_idx]
    log_mu = intercept + attack[away_idx] - defence[home_idx]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    tau, dtau_dlam, dtau_dmu, dtau_drho = _tau_terms(home_goals, away_goals, lam, mu, rho)

    ll = weights * (
        home_goals * log_lam - lam + away_goals * log_mu - mu + np.log(tau)
    )
    nll = -float(ll.sum()) + l2 * (float(attack @ attack) + float(defence @ defence))

    # d(log-likelihood) / d(log lambda) and / d(log mu)
    g_home = weights * (home_goals - lam + lam * dtau_dlam / tau)
    g_away = weights * (away_goals - mu + mu * dtau_dmu / tau)

    grad_attack = np.bincount(home_idx, g_home, n_teams) + np.bincount(away_idx, g_away, n_teams)
    grad_defence = -(
        np.bincount(away_idx, g_home, n_teams) + np.bincount(home_idx, g_away, n_teams)
    )
    grad_intercept = float(g_home.sum() + g_away.sum())
    grad_home_adv = float(g_home.sum())
    grad_rho = float((weights * dtau_drho / tau).sum())

    grad = -np.concatenate(
        [grad_attack, grad_defence, [grad_intercept, grad_home_adv, grad_rho]]
    )
    grad[:n_teams] += 2.0 * l2 * attack
    grad[n_teams : 2 * n_teams] += 2.0 * l2 * defence
    return nll, grad


def fit_dixon_coles(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    *,
    l2: float = 0.05,
    rho_bounds: Sequence[float] = DEFAULT_RHO_BOUNDS,
    max_iter: int = 500,
    x0: np.ndarray | None = None,
) -> DixonColesFit:
    """Fit attack/defence strengths, home advantage and rho by weighted MLE."""
    home_idx = np.asarray(home_idx, dtype=np.int64)
    away_idx = np.asarray(away_idx, dtype=np.int64)
    home_goals = np.asarray(home_goals, dtype=np.float64)
    away_goals = np.asarray(away_goals, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if x0 is None:
        mean_goals = float(np.average(np.concatenate([home_goals, away_goals]), weights=np.concatenate([weights, weights])))
        theta0 = np.zeros(2 * n_teams + 3)
        theta0[2 * n_teams] = np.log(max(mean_goals, 0.1))
        theta0[2 * n_teams + 1] = 0.2
    else:
        theta0 = np.array(x0, dtype=np.float64)

    bounds = (
        [(-_STRENGTH_BOUND, _STRENGTH_BOUND)] * (2 * n_teams)
        + [_INTERCEPT_BOUNDS, _HOME_ADV_BOUNDS, (float(rho_bounds[0]), float(rho_bounds[1]))]
    )

    result = minimize(
        _objective,
        theta0,
        args=(home_idx, away_idx, home_goals, away_goals, weights, n_teams, l2),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    theta = result.x
    attack = theta[:n_teams].copy()
    defence = theta[n_teams : 2 * n_teams].copy()
    intercept = float(theta[2 * n_teams])

    # Attack/defence are only identified up to a shift; centre them so ratings
    # stay comparable across checkpoints.
    attack_mean = float(attack.mean())
    defence_mean = float(defence.mean())
    attack -= attack_mean
    defence -= defence_mean
    intercept += attack_mean - defence_mean

    return DixonColesFit(
        attack=attack,
        defence=defence,
        intercept=intercept,
        home_advantage=float(theta[2 * n_teams + 1]),
        rho=float(theta[2 * n_teams + 2]),
        log_likelihood=-float(result.fun),
        n_matches=int(home_idx.shape[0]),
        weight_sum=float(weights.sum()),
        converged=bool(result.success),
    )


def dixon_coles_joint_probs(
    lam: float,
    mu: float,
    rho: float,
    goal_cap: int,
) -> np.ndarray:
    """Flattened (goal_cap+1)^2 score distribution with the tau correction applied."""
    p_home = poisson_probs(lam, goal_cap)
    p_away = poisson_probs(mu, goal_cap)
    grid = np.outer(p_home, p_away)

    if grid.shape[0] >= 2:
        grid[0, 0] *= max(1.0 - lam * mu * rho, TAU_FLOOR)
        grid[0, 1] *= max(1.0 + lam * rho, TAU_FLOOR)
        grid[1, 0] *= max(1.0 + mu * rho, TAU_FLOOR)
        grid[1, 1] *= max(1.0 - rho, TAU_FLOOR)

    total = float(grid.sum())
    if total <= 0.0:
        return np.full(grid.size, 1.0 / grid.size)
    return (grid / total).reshape(-1)
