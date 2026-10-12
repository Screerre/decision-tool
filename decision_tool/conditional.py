"""Distribution conditionnelle des résultats de séance.

Méthode : simulation historique filtrée (filtered historical simulation).
  1. Chaque séance passée i a une échelle de volatilité prévue s_i (HAR, information d'avant séance).
  2. Ses résultats (rendement ouverture-clôture, excursions défavorables, amplitude) sont divisés par
     s_i : on obtient des résultats standardisés z_i, comparables entre régimes de volatilité.
  3. Pour la séance du jour, on ré-échelonne ces z_i par la prévision s_0 du jour : on obtient des
     milliers de scénarios réalistes, qui préservent le lien entre trajet (KO) et résultat final.
  4. Une pondération par analogues (noyau gaussien sur le contexte : rendement de la veille, gap,
     régime de volatilité) introduit l'information directionnelle, avec rétrécissement automatique
     vers la distribution inconditionnelle quand les analogues sont trop peu nombreux.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import GAP_FLOOR  # noqa: F401  (ré-export utile pour les tests)
from .volatility import HARModel, har_design

CTX_COLS = ("zprev", "zgap", "volpct")


@dataclass
class Sample:
    dates: pd.DatetimeIndex
    scale: np.ndarray
    z_oc: np.ndarray
    z_long: np.ndarray
    z_short: np.ndarray
    z_range: np.ndarray
    ctx: pd.DataFrame
    wd: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)

    def take(self, mask: np.ndarray) -> "Sample":
        """Sous-échantillon (masque booléen)."""
        return Sample(self.dates[mask], self.scale[mask], self.z_oc[mask], self.z_long[mask],
                      self.z_short[mask], self.z_range[mask], self.ctx.loc[mask], self.wd[mask])


def build_sample(frame: pd.DataFrame, model: HARModel, open_start) -> Sample:
    """Résultats historiques standardisés, sur les séances à ouverture fiable."""
    s = np.exp(model.predict_log(har_design(frame, model.use_gap)))
    d = frame.assign(s=s)
    mask = (d.index >= pd.Timestamp(open_start)) & d["s"].notna() & d["r_oc"].notna() \
        & d["adv_long"].notna() & d["adv_short"].notna() & d["gap"].notna() & d["ret_cc"].shift(1).notna()
    d = d.loc[mask]
    sc = d["s"].to_numpy(float)
    prev_ret = frame["ret_cc"].shift(1).loc[d.index].to_numpy(float)
    ctx = pd.DataFrame({
        "zprev": prev_ret / sc,
        "zgap": d["gap"].to_numpy(float) / sc,
        "volpct": pd.Series(np.log(sc)).rank(pct=True).to_numpy(),
    }, index=d.index)
    return Sample(
        dates=d.index, scale=sc,
        z_oc=d["r_oc"].to_numpy(float) / sc,
        z_long=d["adv_long"].to_numpy(float) / sc,
        z_short=d["adv_short"].to_numpy(float) / sc,
        z_range=((d["High"] - d["Low"]) / d["Open"]).to_numpy(float) / sc,
        ctx=ctx, wd=d["wd"].to_numpy(int),
    )


def today_context(frame_with_session: pd.DataFrame, model: HARModel, sample: Sample) -> tuple[float, dict]:
    """Échelle prévue s_0 et contexte de la séance du jour (dernière ligne du frame)."""
    X = har_design(frame_with_session, model.use_gap).iloc[[-1]]
    if X.isna().any(axis=None):
        raise ValueError("Contexte incomplet pour la séance du jour (historique trop court ou gap manquant).")
    s0 = float(np.exp(model.predict_log(X).iloc[0]))
    row = frame_with_session.iloc[-1]
    ret_prev = frame_with_session["ret_cc"].iloc[-2]
    gap0 = row["gap"]
    ctx = {
        "zprev": float(ret_prev / s0),
        "zgap": float(gap0 / s0) if np.isfinite(gap0) else np.nan,
        "volpct": float((np.log(sample.scale) < np.log(s0)).mean()),
        "wd": int(row["wd"]),
        "ret_prev": float(ret_prev),
        "gap": float(gap0) if np.isfinite(gap0) else np.nan,
    }
    return s0, ctx


def analog_weights(sample: Sample, ctx0: dict, cols=("zprev",), bw_scale: float = 1.0,
                   k_prior: float = 200.0, wd_weight: float | None = None, clip: float = 6.0):
    """Poids des séances passées selon leur ressemblance avec le contexte du jour.

    Noyau gaussien produit, fenêtre de Silverman multipliée par bw_scale. Les poids sont ensuite
    mélangés avec les poids uniformes : lam = ESS / (ESS + k_prior). Moins il y a d'analogues, plus
    on revient vers la distribution inconditionnelle (rétrécissement de type bayésien).

    Renvoie (poids normalisés, ESS du noyau, lam, ESS final).
    """
    n = len(sample)
    if not cols:
        w = np.full(n, 1.0 / n)
        return w, float(n), 0.0, float(n)
    Xh = np.clip(sample.ctx[list(cols)].to_numpy(float), -clip, clip)
    x0 = np.clip(np.array([ctx0[c] for c in cols], float), -clip, clip)
    if not np.all(np.isfinite(x0)):
        raise ValueError(f"Variable de contexte manquante parmi {cols} (gap non renseigné ?).")
    d = Xh.shape[1]
    h = bw_scale * Xh.std(0, ddof=1) * (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
    w = np.exp(-0.5 * (((Xh - x0) / h) ** 2).sum(1))
    if wd_weight is not None:
        w = w * np.where(sample.wd == ctx0["wd"], 1.0, float(wd_weight))
    if w.sum() < 1e-300:
        w = np.ones(n)
    wn = w / w.sum()
    ess_k = float(1.0 / np.sum(wn ** 2))
    lam = ess_k / (ess_k + k_prior)
    wf = lam * wn + (1 - lam) / n
    return wf, ess_k, float(lam), float(1.0 / np.sum(wf ** 2))


# ---------------------------------------------------------------------------------- statistiques pondérées
def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.sum(w))


def wprob(mask: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w[mask]) / np.sum(w))


def wquantile(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    cw = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(np.asarray(qs, float), cw, xs)


def boot_indices(w: np.ndarray, n_boot: int = 1000, m: int | None = None, seed: int = 0) -> np.ndarray:
    """Tirages bootstrap pondérés (indices). Taille m = ESS : l'IC reflète l'information réellement
    disponible, pas le nombre brut de séances. Mêmes tirages pour toutes les statistiques."""
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    m = int(m if m is not None else max(50, round(1.0 / np.sum(p ** 2))))
    return rng.choice(len(w), size=(n_boot, m), p=p).astype(np.int32)

