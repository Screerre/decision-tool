"""Prévision de l'amplitude de la séance : modèle HAR sur le log de la volatilité de Parkinson.

Pourquoi cette brique est centrale : l'amplitude (donc le risque de KO) est bien plus prévisible que
la direction (clustering de volatilité). Le modèle fournit une échelle sigma_hat par séance ; les
résultats historiques sont ensuite exprimés en multiples de cette échelle (voir conditional.py).

Deux modes :
  - pré-ouverture : régresseurs = volatilité de la veille, des 5 et 22 derniers jours ;
  - post-ouverture : ajoute le log du |gap| d'ouverture, connu à 15h30 pour le Nasdaq.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import GAP_FLOOR


def har_design(frame: pd.DataFrame, use_gap: bool) -> pd.DataFrame:
    """Matrice de régresseurs pour la séance t, construite uniquement avec l'information <= t-1
    (plus le gap de t si use_gap, connu à l'ouverture)."""
    s = frame["sig_pk"]
    v = s ** 2
    X = pd.DataFrame(index=frame.index)
    X["const"] = 1.0
    X["l1"] = np.log(s.shift(1))
    X["l5"] = 0.5 * np.log(v.rolling(5, min_periods=3).mean().shift(1))
    X["l22"] = 0.5 * np.log(v.rolling(22, min_periods=11).mean().shift(1))
    if use_gap:
        X["lgap"] = np.log(frame["gap"].abs() + GAP_FLOOR)
    return X


@dataclass
class HARModel:
    use_gap: bool
    names: list
    coef: np.ndarray
    resid_sd: float
    r2: float
    n: int

    def predict_log(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(X[self.names].to_numpy(float) @ self.coef, index=X.index)

    def describe(self) -> str:
        c = ", ".join(f"{n}={v:+.3f}" for n, v in zip(self.names, self.coef))
        mode = "post-ouverture" if self.use_gap else "pré-ouverture"
        return f"HAR {mode}: n={self.n}, R2={self.r2:.3f}, sd résiduelle={self.resid_sd:.3f} | {c}"


def fit_har(frame: pd.DataFrame, use_gap: bool, start=None, end=None) -> HARModel:
    """Ajuste le HAR par MCO sur les séances [start, end). Les barres plates sont exclues."""
    X = har_design(frame, use_gap)
    y = np.log(frame["sig_pk"])
    mask = X.notna().all(axis=1) & y.notna()
    if start is not None:
        mask &= frame.index >= pd.Timestamp(start)
    if end is not None:
        mask &= frame.index < pd.Timestamp(end)
    Xv, yv = X.loc[mask].to_numpy(float), y.loc[mask].to_numpy(float)
    if len(yv) < 250:
        raise ValueError(f"Pas assez de séances exploitables pour ajuster le HAR ({len(yv)}).")
    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    res = yv - Xv @ coef
    r2 = 1 - res.var() / yv.var()
    return HARModel(use_gap, list(X.columns), coef, float(res.std(ddof=len(coef))), float(r2), len(yv))


def sigma_scale(model: HARModel, frame: pd.DataFrame) -> pd.Series:
    """Échelle de volatilité prévue (niveau géométrique, exp de la prévision du log)."""
    return np.exp(model.predict_log(har_design(frame, model.use_gap)))


def ewma_sigma(frame: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """Benchmark RiskMetrics : variance EWMA de la proxy de Parkinson, décalée d'un jour."""
    v = frame["sig_pk"] ** 2
    return np.sqrt(v.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean().shift(1))

