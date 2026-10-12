"""Orchestration : du contexte de la séance à un verdict chiffré et honnête."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import conditional as cond
from .data import Quality, clean_incoherent, quality_report
from .features import add_session, build_frame
from .turbo import Spec, evaluate, ko_table
from .volatility import HARModel, fit_har


@dataclass
class Settings:
    leverages: tuple = (10, 20, 30, 50, 100)
    stops: tuple = (0.003, 0.005)          # distances de stop testées en plus de "sans stop"
    cost_bp: float = 3.0                   # coût aller-retour en pb du sous-jacent
    risk_eur: float = 10.0                 # perte maximale acceptée par trade
    capital: float = 1000.0                # enveloppe de trading
    alpha: float = 0.05                    # seuil après correction de Holm
    n_boot: int = 1000
    seed: int = 0
    use_vol_context: bool = False          # ajoute le régime de volatilité au noyau d'analogues
    use_weekday: bool = False              # favorise les mêmes jours de semaine (effet non significatif)
    wd_weight: float = 0.3
    bw_scale: float = 1.0
    k_prior: float = 200.0


@dataclass
class Report:
    mode: str
    session_date: pd.Timestamp
    warnings: list
    quality: Quality
    model: HARModel
    context: dict
    s0: float
    n_sample: int
    ess_kernel: float
    shrink_lambda: float
    ess_final: float
    ctx_cols: tuple
    direction: dict
    amplitude: dict
    ko: pd.DataFrame
    grid: pd.DataFrame
    verdict: dict
    settings: Settings = field(default_factory=Settings)
    scen: dict = field(default_factory=dict)
    weights: np.ndarray | None = None


def _busdays_missing(last, session) -> int:
    return int(np.busday_count(np.datetime64(last.date()), np.datetime64(session.date()))) - 1


def analyse(ohlc: pd.DataFrame, session_date, open_price: float | None = None,
            settings: Settings | None = None) -> Report:
    """Analyse la séance `session_date` pour l'historique `ohlc` (OHLC journalier brut)."""
    st = settings or Settings()
    session_date = pd.Timestamp(session_date)
    warnings: list = []
    q = quality_report(ohlc)
    frame = build_frame(clean_incoherent(ohlc))
    miss = _busdays_missing(frame.index[-1], session_date)
    if miss > 0:
        warnings.append(f"{miss} séance(s) entre la dernière ligne du fichier ({frame.index[-1].date()}) et "
                        f"le {session_date.date()} : ajoute-les, sinon la 'veille' utilisée est périmée.")

    use_gap = open_price is not None
    mode = "post-ouverture" if use_gap else "pré-ouverture"
    model = fit_har(frame, use_gap=use_gap, start=q.open_start if use_gap else q.range_start)
    sample = cond.build_sample(frame, model, q.open_start)
    frame2 = add_session(frame, session_date, open_price)
    s0, ctx0 = cond.today_context(frame2, model, sample)

    cols = ["zprev"] + (["zgap"] if use_gap else []) + (["volpct"] if st.use_vol_context else [])
    w, ess_k, lam, ess_f = cond.analog_weights(
        sample, ctx0, tuple(cols), st.bw_scale, st.k_prior, st.wd_weight if st.use_weekday else None)
    n = len(sample)
    w_unif = np.full(n, 1.0 / n)
    idx = cond.boot_indices(w, st.n_boot, seed=st.seed)

    scen = {"r_oc": sample.z_oc * s0, "adv_long": sample.z_long * s0, "adv_short": sample.z_short * s0}
    rng_ = sample.z_range * s0

    up = scen["r_oc"] > 0
    bp_up = up[idx].mean(axis=1)
    bm_mean = scen["r_oc"][idx].mean(axis=1)
    direction = {
        "p_up": cond.wprob(up, w), "p_up_lo": float(np.percentile(bp_up, 5)), "p_up_hi": float(np.percentile(bp_up, 95)),
        "p_up_uncond": cond.wprob(up, w_unif),
        "mean": cond.wmean(scen["r_oc"], w), "mean_lo": float(np.percentile(bm_mean, 5)),
        "mean_hi": float(np.percentile(bm_mean, 95)), "mean_uncond": cond.wmean(scen["r_oc"], w_unif),
        "quantiles": dict(zip((5, 25, 50, 75, 95), cond.wquantile(scen["r_oc"], w, [.05, .25, .5, .75, .95]))),
        "p_gt_1": cond.wprob(scen["r_oc"] > 0.01, w), "p_lt_m1": cond.wprob(scen["r_oc"] < -0.01, w),
    }
    direction["distinguable"] = not (direction["p_up_lo"] <= direction["p_up_uncond"] <= direction["p_up_hi"])
    amplitude = {
        "range_median": float(cond.wquantile(rng_, w, [0.5])[0]), "range_q90": float(cond.wquantile(rng_, w, [0.9])[0]),
        "p_range_ge_1": cond.wprob(rng_ >= 0.01, w), "p_range_ge_2": cond.wprob(rng_ >= 0.02, w),
        "vol_percentile": ctx0["volpct"],
    }

    specs = []
    for sens in ("long", "short"):
        for L in st.leverages:
            specs.append(Spec(sens, float(L), None))
            for s in st.stops:
                if s < 1.0 / L:
                    specs.append(Spec(sens, float(L), float(s)))
    grid = evaluate(specs, scen, w, idx, st.cost_bp, st.risk_eur, st.capital)
    ko = ko_table(scen, w, idx, st.leverages)

    sig = grid[(grid["p_holm"] < st.alpha) & (grid["EV_pct_mise"] > 0)]
    if len(sig):
        best = sig.sort_values("CI90_lo", ascending=False).iloc[0]
        verdict = {"trade": True, "produit": best["produit"], "row": best.to_dict(),
                   "reason": (f"Seul(s) {len(sig)} produit(s) sur {len(grid)} ont une espérance positive "
                              f"significative après correction de Holm (alpha={st.alpha}).")}
    else:
        top = grid.sort_values("EV_pct_mise", ascending=False).iloc[0]
        verdict = {"trade": False, "produit": None, "row": top.to_dict(),
                   "reason": (f"Aucun des {len(grid)} produits testés n'a une espérance positive distinguable de "
                              f"zéro après correction de Holm (alpha={st.alpha}). Le moins mauvais est "
                              f"{top['produit']} avec {top['EV_pct_mise']:+.1f}% de la mise en moyenne.")}
    if ess_f < 100:
        warnings.append(f"Peu d'analogues (ESS final={ess_f:.0f}) : les résultats se rapprochent de l'inconditionnel.")

    return Report(mode, session_date, warnings, q, model, ctx0, s0, n, ess_k, lam, ess_f, tuple(cols),
                  direction, amplitude, ko, grid, verdict, st, scen, w)

