"""Validation hors échantillon (walk-forward) : ce que l'outil sait vraiment prévoir.

Principe : pour chaque année de test Y, tout (coefficients HAR, distribution des résultats
standardisés, analogues) est estimé uniquement avec les séances antérieures au 1er janvier Y.
Trois questions, trois tests :
  1. La prévision de volatilité bat-elle des benchmarks simples ?       -> oos_volatility
  2. Les probabilités de KO annoncées sont-elles calibrées ?            -> ko_calibration
  3. Le contexte (veille, gap) améliore-t-il la probabilité de hausse ? -> direction_skill
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import conditional as cond
from .data import Quality, clean_incoherent
from .features import build_frame
from .volatility import ewma_sigma, fit_har, har_design, sigma_scale


def _prep(ohlc: pd.DataFrame):
    return build_frame(clean_incoherent(ohlc))


def oos_volatility(frame: pd.DataFrame, q: Quality, first_test_year: int | None = None) -> dict:
    """Carré de la corrélation (prévision, réalisé) sur le log de la volatilité, hors échantillon."""
    y0 = first_test_year or q.open_start.year + 5
    X = har_design(frame, False)
    y = np.log(frame["sig_pk"])
    ew = np.log(ewma_sigma(frame))
    preds = {"HAR": [], "EWMA 0.94": [], "veille seule": [], "moyenne 22 j": [], "y": [], "train_mean": []}
    for Y in range(y0, int(frame.index.year.max()) + 1):
        m = fit_har(frame, False, start=q.range_start, end=pd.Timestamp(year=Y, month=1, day=1))
        test = (frame.index.year == Y) & y.notna() & X.notna().all(axis=1) & ew.notna()
        tm = y[(frame.index < pd.Timestamp(year=Y, month=1, day=1)) & (frame.index >= q.range_start)].mean()
        preds["HAR"].append(m.predict_log(X.loc[test]).to_numpy())
        preds["EWMA 0.94"].append(ew[test].to_numpy())
        preds["veille seule"].append(X.loc[test, "l1"].to_numpy())
        preds["moyenne 22 j"].append(X.loc[test, "l22"].to_numpy())
        preds["y"].append(y[test].to_numpy())
        preds["train_mean"].append(np.full(test.sum(), tm))
    cat = {k: np.concatenate(v) for k, v in preds.items()}
    out = {k: float(np.corrcoef(cat[k], cat["y"])[0, 1] ** 2) for k in ("HAR", "EWMA 0.94", "veille seule", "moyenne 22 j")}
    sse = lambda p: float(np.sum((cat["y"] - p) ** 2))
    out["R2_oos_HAR_vs_moyenne"] = 1 - sse(cat["HAR"]) / sse(cat["train_mean"])
    out["n_test"] = int(len(cat["y"]))
    return out


def ko_calibration(frame: pd.DataFrame, q: Quality, barriers=(0.01, 0.02), first_test_year: int | None = None,
                   n_bins: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probabilités de KO prévues (volatilité filtrée) contre fréquences réalisées, hors échantillon.

    Référence : fréquence historique brute (sans tenir compte du régime de volatilité).
    Renvoie (résumé par sens et barrière, table de fiabilité par classes de probabilité prévue).
    """
    y0 = first_test_year or q.open_start.year + 5
    rows: dict = {(s, b): {"p": [], "p0": [], "y": []} for s in ("long", "short") for b in barriers}
    for Y in range(y0, int(frame.index.year.max()) + 1):
        start = pd.Timestamp(year=Y, month=1, day=1)
        m = fit_har(frame, False, start=q.range_start, end=start)
        s = sigma_scale(m, frame)
        ok = s.notna() & frame["adv_long"].notna() & (frame.index >= q.open_start)
        train, test = ok & (frame.index < start), ok & (frame.index.year == Y)
        for sens, col in (("long", "adv_long"), ("short", "adv_short")):
            z = np.sort((frame.loc[train, col] / s[train]).to_numpy())
            a_tr = frame.loc[train, col].to_numpy()
            s_te, a_te = s[test].to_numpy(), frame.loc[test, col].to_numpy()
            for b in barriers:
                p = 1.0 - np.searchsorted(z, b / s_te, side="left") / len(z)
                rows[(sens, b)]["p"].append(p)
                rows[(sens, b)]["p0"].append(np.full(len(s_te), np.mean(a_tr >= b)))
                rows[(sens, b)]["y"].append((a_te >= b).astype(float))
    summ, rel = [], []
    for (sens, b), d in rows.items():
        p, p0, y = (np.concatenate(d[k]) for k in ("p", "p0", "y"))
        bm, b0 = float(np.mean((p - y) ** 2)), float(np.mean((p0 - y) ** 2))
        summ.append({"sens": sens, "barrière_%": b * 100, "n": len(y), "fréquence_réalisée_%": y.mean() * 100,
                     "prob_prévue_moy_%": p.mean() * 100, "Brier_modèle": bm, "Brier_référence": b0,
                     "gain_Brier_%": (1 - bm / b0) * 100})
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        cls = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        for k in range(len(edges) - 1):
            sel = cls == k
            if sel.any():
                rel.append({"sens": sens, "barrière_%": b * 100, "classe": k + 1, "n": int(sel.sum()),
                            "prévue_%": p[sel].mean() * 100, "réalisée_%": y[sel].mean() * 100})
    return pd.DataFrame(summ), pd.DataFrame(rel)


def _newey_west_t(d: np.ndarray, lag: int = 5) -> float:
    n = len(d)
    dc = d - d.mean()
    v = np.sum(dc ** 2) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    return float(d.mean() / np.sqrt(v / n))


def direction_skill(frame: pd.DataFrame, q: Quality, cols=("zprev",), first_test_year: int | None = None,
                    bw_scale: float = 1.0, k_prior: float = 200.0) -> dict:
    """Le contexte améliore-t-il P(hausse ouverture->clôture) hors échantillon ? Test de Diebold-Mariano."""
    y0 = first_test_year or q.open_start.year + 5
    pc_all, p0_all, y_all = [], [], []
    for Y in range(y0, int(frame.index.year.max()) + 1):
        start = pd.Timestamp(year=Y, month=1, day=1)
        m = fit_har(frame, "zgap" in cols, start=q.open_start if "zgap" in cols else q.range_start, end=start)
        full = cond.build_sample(frame, m, q.open_start)
        tr = full.dates < start
        te = full.dates.year == Y
        if te.sum() == 0:
            continue
        train = full.take(tr)
        up_tr = train.z_oc > 0
        p0 = up_tr.mean()
        ctx_te = full.ctx.loc[te]
        wd_te = full.wd[te]
        for j, (_, row) in enumerate(ctx_te.iterrows()):
            c0 = {**row.to_dict(), "wd": int(wd_te[j])}
            w, *_ = cond.analog_weights(train, c0, cols, bw_scale, k_prior)
            pc_all.append(float(np.sum(w[up_tr])))
        p0_all.append(np.full(te.sum(), p0))
        y_all.append((full.z_oc[te] > 0).astype(float))
    pc, p0, y = np.array(pc_all), np.concatenate(p0_all), np.concatenate(y_all)
    d = (p0 - y) ** 2 - (pc - y) ** 2
    t = _newey_west_t(d)
    return {"n_test": len(y), "Brier_contexte": float(np.mean((pc - y) ** 2)), "Brier_référence": float(np.mean((p0 - y) ** 2)),
            "gain_Brier_%": float((1 - np.mean((pc - y) ** 2) / np.mean((p0 - y) ** 2)) * 100),
            "t_DM": t, "p_value": float(2 * stats.norm.sf(abs(t))), "taux_hausse_%": float(y.mean() * 100),
            "écart-type_prob_contexte": float(pc.std())}


def run_all(ohlc: pd.DataFrame, q: Quality) -> dict:
    frame = _prep(ohlc)
    ko_sum, ko_rel = ko_calibration(frame, q)
    return {
        "volatilité": oos_volatility(frame, q),
        "ko_résumé": ko_sum, "ko_fiabilité": ko_rel,
        "direction_pré": direction_skill(frame, q, ("zprev",)),
        "direction_post": direction_skill(frame, q, ("zprev", "zgap")),
    }

