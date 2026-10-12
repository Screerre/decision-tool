"""Mise en forme des résultats (texte et tableaux d'affichage)."""
from __future__ import annotations

import pandas as pd

from .decision import Report

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
DISCLAIMER = ("Outil d'aide à l'analyse fondé sur des statistiques historiques, pas un conseil en investissement. "
              "Le passé ne garantit pas l'avenir ; un turbo peut perdre 100 % de la mise.")


def regime_label(pct: float) -> str:
    return "calme" if pct < 1 / 3 else ("moyen" if pct < 2 / 3 else "agité")


def display_grid(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid.copy()
    out = pd.DataFrame({
        "Produit": g["produit"],
        "Espérance (% de la mise)": g["EV_pct_mise"].round(2),
        "IC90 bas": g["CI90_lo"].round(2), "IC90 haut": g["CI90_hi"].round(2),
        "p ajusté (Holm)": g["p_holm"].round(3),
        "P(KO) %": (g["P_KO"] * 100).round(1), "P(stop) %": (g["P_stop"] * 100).round(1),
        "P(gain) %": (g["P_gain"] * 100).round(1),
        "Pire 5% (% mise)": g["CVaR5"].round(1),
        "Mise pour le risque fixé (€)": g["mise_eur"].round(0), "Espérance (€)": g["EV_eur"].round(3),
    })
    return out


def display_ko(ko: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Levier": ko["levier"], "Barrière %": ko["barrière_%"].round(2),
        "KO long %": (ko["P_KO_long"] * 100).round(1),
        "IC90 long": [f"{a * 100:.1f} à {b * 100:.1f}" for a, b in zip(ko["P_KO_long_lo"], ko["P_KO_long_hi"])],
        "KO short %": (ko["P_KO_short"] * 100).round(1),
        "IC90 short": [f"{a * 100:.1f} à {b * 100:.1f}" for a, b in zip(ko["P_KO_short_lo"], ko["P_KO_short_hi"])],
    })


def format_report(r: Report, top: int = 5) -> str:
    c, d, a = r.context, r.direction, r.amplitude
    L = []
    L.append(f"ANALYSE DE LA SÉANCE DU {r.session_date.date()} ({JOURS[r.session_date.dayofweek]}) | mode {r.mode}")
    L.append("=" * 78)
    L.append("Données : " + r.quality.summary())
    L.append(f"Échantillon d'entraînement : {r.n_sample} séances à ouverture fiable | analogues sur "
             f"{', '.join(r.ctx_cols)} | ESS noyau {r.ess_kernel:.0f}, rétrécissement lambda {r.shrink_lambda:.2f}, "
             f"ESS final {r.ess_final:.0f}")
    L.append(r.model.describe())
    L.append("")
    L.append("CONTEXTE")
    L.append(f"  Veille : {c['ret_prev'] * 100:+.2f}% ({c['zprev']:+.2f} fois la volatilité prévue du jour)")
    if c["gap"] == c["gap"]:
        L.append(f"  Gap d'ouverture : {c['gap'] * 100:+.2f}% ({c['zgap']:+.2f} fois la volatilité prévue)")
    else:
        L.append("  Gap d'ouverture : inconnu (analyse avant l'ouverture)")
    L.append(f"  Régime de volatilité : {regime_label(a['vol_percentile'])} (percentile {a['vol_percentile'] * 100:.0f} "
             f"de l'historique)")
    L.append("")
    L.append("AMPLITUDE ATTENDUE (haut moins bas, en % de l'ouverture)")
    L.append(f"  Médiane {a['range_median'] * 100:.2f}% | 90% des séances sous {a['range_q90'] * 100:.2f}% | "
             f"P(>= 1%) = {a['p_range_ge_1'] * 100:.0f}% | P(>= 2%) = {a['p_range_ge_2'] * 100:.1f}%")
    L.append("")
    L.append("DIRECTION (ouverture vers clôture)")
    L.append(f"  P(hausse) = {d['p_up'] * 100:.1f}% (IC90 {d['p_up_lo'] * 100:.1f} à {d['p_up_hi'] * 100:.1f}) "
             f"contre {d['p_up_uncond'] * 100:.1f}% sans contexte : "
             f"{'écart notable' if d['distinguable'] else 'écart non distinguable du bruit'}")
    L.append(f"  Mouvement moyen {d['mean'] * 100:+.3f}% (IC90 {d['mean_lo'] * 100:+.3f} à {d['mean_hi'] * 100:+.3f})")
    q = d["quantiles"]
    L.append(f"  Quantiles 5/25/50/75/95% : " + " / ".join(f"{q[k] * 100:+.2f}%" for k in (5, 25, 50, 75, 95)))
    L.append(f"  P(hausse > 1%) = {d['p_gt_1'] * 100:.1f}% | P(baisse > 1%) = {d['p_lt_m1'] * 100:.1f}%")
    L.append("")
    L.append("RISQUE DE KNOCK-OUT depuis l'ouverture (IC90)")
    for _, k in r.ko.iterrows():
        L.append(f"  Levier {k['levier']:>5g} (barrière {k['barrière_%']:.2f}%) : long {k['P_KO_long'] * 100:5.1f}% "
                 f"[{k['P_KO_long_lo'] * 100:.1f}-{k['P_KO_long_hi'] * 100:.1f}] | short {k['P_KO_short'] * 100:5.1f}% "
                 f"[{k['P_KO_short_lo'] * 100:.1f}-{k['P_KO_short_hi'] * 100:.1f}]")
    L.append("")
    s = r.settings
    L.append(f"PRODUITS (mise pour risquer {s.risk_eur:g} EUR, coût {s.cost_bp:g} pb, {len(r.grid)} produits testés)")
    g = r.grid.sort_values("EV_pct_mise", ascending=False).head(top)
    for _, x in g.iterrows():
        L.append(f"  {x['produit']:<28} esp. {x['EV_pct_mise']:+6.2f}% de la mise [IC90 {x['CI90_lo']:+.2f} ; {x['CI90_hi']:+.2f}] "
                 f"p Holm {x['p_holm']:.2f} | KO {x['P_KO'] * 100:.1f}% stop {x['P_stop'] * 100:.0f}% | mise {x['mise_eur']:.0f} EUR")
    L.append("")
    v = r.verdict
    L.append("VERDICT : " + ("UN PRODUIT SE DÉMARQUE" if v["trade"] else "NE PAS TRADER sur cet avantage statistique"))
    L.append("  " + v["reason"])
    if v["trade"]:
        x = v["row"]
        L.append(f"  Candidat : {x['produit']} | espérance {x['EV_pct_mise']:+.2f}% de la mise "
                 f"[IC90 {x['CI90_lo']:+.2f} ; {x['CI90_hi']:+.2f}] | Kelly {x['kelly_f'] * 100:.0f}% du capital en mise "
                 f"(prendre au plus le quart)")
    for w in r.warnings:
        L.append("  ATTENTION : " + w)
    L.append("")
    L.append(DISCLAIMER)
    return "\n".join(L)


def format_validation(v: dict) -> str:
    vol, dp, dq = v["volatilité"], v["direction_pré"], v["direction_post"]
    L = ["VALIDATION HORS ÉCHANTILLON (walk-forward, ré-estimation annuelle)", "=" * 78]
    L.append(f"1. Prévision de volatilité ({vol['n_test']} séances de test) : carré de la corrélation avec le réalisé")
    L.append(f"   HAR {vol['HAR']:.3f} | EWMA 0.94 {vol['EWMA 0.94']:.3f} | moyenne 22 j {vol['moyenne 22 j']:.3f} | "
             f"veille seule {vol['veille seule']:.3f} | R2 HAR vs moyenne {vol['R2_oos_HAR_vs_moyenne']:.3f}")
    L.append("2. Calibration des probabilités de KO (modèle contre fréquence brute, score de Brier)")
    for _, x in v["ko_résumé"].iterrows():
        L.append(f"   {x['sens']:<5} barrière {x['barrière_%']:.0f}% : prévue {x['prob_prévue_moy_%']:.1f}% "
                 f"réalisée {x['fréquence_réalisée_%']:.1f}% | gain de Brier {x['gain_Brier_%']:+.1f}%")
    L.append("   Fiabilité par classe de probabilité prévue (prévue / réalisée, en %) :")
    for (sens, b), grp in v["ko_fiabilité"].groupby(["sens", "barrière_%"]):
        L.append(f"   {sens:<5} {b:.0f}% : " + " | ".join(f"{p:.1f}/{r:.1f}" for p, r in zip(grp["prévue_%"], grp["réalisée_%"])))
    L.append("3. Le contexte améliore-t-il P(hausse) ? (Diebold-Mariano, écart de Brier)")
    L.append(f"   Veille seule : gain {dp['gain_Brier_%']:+.2f}% (t = {dp['t_DM']:+.2f}, p = {dp['p_value']:.2f})")
    L.append(f"   Veille + gap : gain {dq['gain_Brier_%']:+.2f}% (t = {dq['t_DM']:+.2f}, p = {dq['p_value']:.2f})")
    L.append("   Lecture : un gain proche de 0 avec p élevé signifie qu'aucune information directionnelle fiable "
             "n'est détectée.")
    return "\n".join(L)

