"""Log-linear blend of CatBoost and Dixon-Coles goal expectations.

Both sub-models produce per-match Poisson means (lambda_home, lambda_away).
The blend combines them geometrically with a single weight w::

    log lambda = (1 - w) * log lambda_catboost + w * log lambda_dixon_coles

and decodes the blended means through the Dixon-Coles low-score correction
(rho from the causal rating checkpoint) into the Kicktipp expected-points
optimal score. The weight is calibrated on training seasons only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from models.catboost_poisson import model as cb_model
from models.common.kicktipp import (
    build_kicktipp_points_matrix,
    kicktipp_optimal_with_diagnostics,
)
from models.dixon_coles import model as dc_model
from models.dixon_coles.ratings import dixon_coles_joint_probs

_LAMBDA_FLOOR = 1e-8


def blend_lambdas(
    lam_cb: np.ndarray,
    lam_dc: np.ndarray,
    weight: float,
) -> np.ndarray:
    lam_cb = np.maximum(np.asarray(lam_cb, dtype=float), _LAMBDA_FLOOR)
    lam_dc = np.maximum(np.asarray(lam_dc, dtype=float), _LAMBDA_FLOOR)
    return np.exp((1.0 - weight) * np.log(lam_cb) + weight * np.log(lam_dc))


def decode_blended_diagnostics(
    cb_home: np.ndarray,
    cb_away: np.ndarray,
    dc_home: np.ndarray,
    dc_away: np.ndarray,
    rho: np.ndarray,
    weight: float,
    goal_cap: int,
    points_matrix: np.ndarray,
) -> List[dict[str, Any]]:
    lam_home = blend_lambdas(cb_home, dc_home, weight)
    lam_away = blend_lambdas(cb_away, dc_away, weight)

    out: List[dict[str, Any]] = []
    for lam, mu, r in zip(lam_home, lam_away, rho):
        joint = dixon_coles_joint_probs(float(lam), float(mu), float(r), goal_cap)
        diag = kicktipp_optimal_with_diagnostics(joint, goal_cap, points_matrix)
        out.append(
            {
                "tip": str(diag["tip"]),
                "expected_points": float(diag["expected_points"]),
                "variance": float(diag["variance"]),
            }
        )
    return out


def decode_blended(
    cb_home: np.ndarray,
    cb_away: np.ndarray,
    dc_home: np.ndarray,
    dc_away: np.ndarray,
    rho: np.ndarray,
    weight: float,
    goal_cap: int,
    points_matrix: np.ndarray,
) -> List[str]:
    return [
        d["tip"]
        for d in decode_blended_diagnostics(
            cb_home,
            cb_away,
            dc_home,
            dc_away,
            rho,
            weight,
            goal_cap,
            points_matrix,
        )
    ]


class PoissonBlendModel:
    def __init__(self, bundle_dir: Path | str):
        art = Path(bundle_dir)
        config = json.loads((art / "blend.json").read_text(encoding="utf-8"))
        self.blend_weight = float(config["blend_weight"])
        self.goal_cap = int(config.get("goal_cap", 7))

        self.catboost = cb_model.load(art / "catboost")
        self.dixon_coles = dc_model.load(art / "dixon_coles")

        self._points_matrix = build_kicktipp_points_matrix(self.goal_cap)
        self.config = config
        self.bundle_dir = art

    def predict(self, X: pd.DataFrame) -> List[str]:
        return [d["tip"] for d in self.predict_with_diagnostics(X)]

    def predict_with_diagnostics(self, X: pd.DataFrame) -> List[dict[str, Any]]:
        cb_home, cb_away = self.catboost.predict_lambdas(X)
        dc_home, dc_away, rho = self.dixon_coles.predict_lambdas(X)
        return decode_blended_diagnostics(
            cb_home,
            cb_away,
            dc_home,
            dc_away,
            rho,
            self.blend_weight,
            self.goal_cap,
            self._points_matrix,
        )


def load(bundle_dir: Path | str) -> PoissonBlendModel:
    return PoissonBlendModel(bundle_dir)
