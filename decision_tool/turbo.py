"""Simulation d'un turbo sur les scénarios de la séance : KO, stop, coûts, taille de position.

Modèle du produit (volontairement simple et explicite) :
  - barrière à 1/levier du cours d'entrée ; si l'excursion défavorable l'atteint, la mise est perdue ;
  - gain en proportion de la mise : levier x rendement du sous-jacent, moins le coût aller-retour ;
  - stop optionnel sur le sous-jacent, placé en deçà de la barrière ; sortie au prix du stop ;
  - sinon sortie à la clôture.
Limites assumées : pas de financement, pas de glissement ni de gap sur le stop, spread résumé par
cost_bp. Un vrai turbo a une barrière et un spread propres : à saisir si tu veux plus de précision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conditional import wmean, wprob, wquantile


@dataclass(frozen=True)
class Spec:
    direction: str            # "long" ou "short"
    leverage: float
    stop: float | None = None  # distance du stop sur le sous-jacent, en fraction (0.004 = 0,4 %)

    @property
    def barrier(self) -> float:
        return 1.0 / self.leverage

    @property
    def label(self) -> str:
        s = "sans stop" if self.stop is None else f"stop {self.stop * 100:.2f}%"
        return f"{self.direction} x{self.leverage:g} ({s})"


def scenario_pnl(spec: Spec, r_oc: np.ndarray, adv_long: np.ndarray, adv_short: np.ndarray,
                 cost_bp: float = 3.0):
    """Résultat en proportion de la mise pour chaque scénario. Renvoie (pnl, ko, stopped)."""
    L = spec.leverage
    adv, r = (adv_long, r_oc) if spec.direction == "long" else (adv_short, -r_oc)
    cost = L * cost_bp / 1e4
    ko = adv >= spec.barrier
    if spec.stop is not None and spec.stop < spec.barrier:
        stopped = (~ko) & (adv >= spec.stop)
        stop_loss = -L * spec.stop - cost
    else:
        stopped = np.zeros_like(ko)
        stop_loss = 0.0
    pnl = np.where(ko, -1.0, np.where(stopped, stop_loss, L * r - cost))
    return np.maximum(pnl, -1.0), ko, stopped


def holm(p: np.ndarray) -> np.ndarray:
    """Correction de Holm-Bonferroni (contrôle du risque de faux positif sur l'ensemble de la grille)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def kelly_fraction(pnl: np.ndarray, w: np.ndarray, f_max: float = 1.0, n_grid: int = 201) -> float:
    """Fraction du capital maximisant la croissance logarithmique espérée, mise = f x capital."""
    grid = np.linspace(0.0, f_max, n_grid)
    wn = w / w.sum()
    g = np.array([np.sum(wn * np.log(np.maximum(1.0 + f * pnl, 1e-12))) for f in grid])
    return float(grid[int(np.argmax(g))])


def evaluate(specs, scen: dict, w: np.ndarray, boot_idx: np.ndarray, cost_bp: float = 3.0,
             risk_eur: float = 10.0, capital: float = 1000.0) -> pd.DataFrame:
    """Évalue chaque produit sur les scénarios pondérés, avec IC bootstrap et correction de Holm."""
    rows, boots = [], []
    for sp in specs:
        pnl, ko, st = scenario_pnl(sp, scen["r_oc"], scen["adv_long"], scen["adv_short"], cost_bp)
        ev = wmean(pnl, w)
        bm = pnl[boot_idx].mean(axis=1)
        boots.append(bm)
        q05, q50, q95 = wquantile(pnl, w, [0.05, 0.5, 0.95])
        tail = pnl <= q05
        cvar = wmean(pnl[tail], w[tail]) if tail.any() else q05
        # taille : on risque risk_eur ; avec stop, la perte au stop vaut mise x (levier x stop + coût)
        if sp.stop is not None and sp.stop < sp.barrier:
            loss_unit = min(1.0, sp.leverage * sp.stop + sp.leverage * cost_bp / 1e4)
        else:
            loss_unit = 1.0
        stake = min(capital, risk_eur / loss_unit)
        rows.append({
            "produit": sp.label, "direction": sp.direction, "leverage": sp.leverage, "stop": sp.stop,
            "EV_pct_mise": ev * 100, "CI90_lo": float(np.percentile(bm, 5)) * 100,
            "CI90_hi": float(np.percentile(bm, 95)) * 100,
            "p_value": (float(np.sum(bm <= 0)) + 1) / (len(bm) + 1),
            "P_KO": wprob(ko, w), "P_stop": wprob(st, w), "P_gain": wprob(pnl > 0, w),
            "q05": q05 * 100, "median": q50 * 100, "q95": q95 * 100, "CVaR5": cvar * 100,
            "kelly_f": kelly_fraction(pnl, w, f_max=1.0), "mise_eur": stake, "EV_eur": stake * ev,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"].to_numpy())
    return out


def ko_table(scen: dict, w: np.ndarray, boot_idx: np.ndarray, leverages) -> pd.DataFrame:
    """Probabilité que la barrière soit touchée, par levier et par sens, avec IC90 bootstrap."""
    rows = []
    for L in leverages:
        b = 1.0 / L
        row = {"levier": L, "barrière_%": b * 100}
        for sens, key in (("long", "adv_long"), ("short", "adv_short")):
            hit = (scen[key] >= b).astype(float)
            bm = hit[boot_idx].mean(axis=1)
            row[f"P_KO_{sens}"] = wprob(hit > 0, w)
            row[f"P_KO_{sens}_lo"] = float(np.percentile(bm, 5))
            row[f"P_KO_{sens}_hi"] = float(np.percentile(bm, 95))
        rows.append(row)
    return pd.DataFrame(rows)

