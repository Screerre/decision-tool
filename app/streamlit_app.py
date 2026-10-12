"""Application Streamlit : saisis le contexte de la séance, lis le verdict statistique.

Lancement : streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from decision_tool.data import append_rows, load_ohlc, quality_report
from decision_tool.decision import Settings, analyse
from decision_tool.report import (DISCLAIMER, JOURS, display_grid, display_ko, format_report,
                                  format_validation, regime_label)
from decision_tool.validation import run_all

DATA_DIR = ROOT / "data"

st.set_page_config(page_title="Aide à la décision : turbos", layout="wide")


@st.cache_data(show_spinner=False)
def _load_path(path: str, mtime: float) -> pd.DataFrame:
    return load_ohlc(path)


@st.cache_data(show_spinner=False)
def _load_upload(content: bytes, name: str) -> pd.DataFrame:
    import io
    buf = io.BytesIO(content)
    buf.name = name
    return load_ohlc(buf, name=Path(name).stem)


@st.cache_data(show_spinner="Validation walk-forward en cours (quelques secondes)...")
def _validate(df: pd.DataFrame) -> dict:
    return run_all(df, quality_report(df))


# ------------------------------------------------------------------------------------------ barre latérale
st.sidebar.header("1. Données")
files = sorted([p for p in DATA_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")])
choice = st.sidebar.selectbox("Sous-jacent (fichier du dossier data/)", [p.name for p in files] or ["(aucun)"])
upload = st.sidebar.file_uploader("ou charger un fichier", type=["csv", "xlsx"])
extra = st.sidebar.text_area(
    "Séances à ajouter (une par ligne : AAAA-MM-JJ,Open,High,Low,Close)", value="", height=80,
    help="Utile si ton fichier n'est pas à jour. Les séances ajoutées servent de 'veille'.")

df = None
try:
    if upload is not None:
        df = _load_upload(upload.getvalue(), upload.name)
    elif files:
        p = DATA_DIR / choice
        df = _load_path(str(p), p.stat().st_mtime)
    if df is not None and extra.strip():
        df = append_rows(df, extra)
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Lecture impossible : {e}")

if df is None:
    st.title("Aide à la décision statistique : turbos intraday")
    st.info("Dépose un fichier OHLC journalier (CSV Stooq ou Excel) dans le dossier data/ ou charge-le dans la barre latérale.")
    st.stop()

next_day = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=1)[0]
st.sidebar.header("2. Séance à analyser")
session_date = st.sidebar.date_input("Date de la séance", value=next_day.date())
open_in = st.sidebar.number_input("Ouverture du jour (0 = pas encore connue)", min_value=0.0, value=0.0, step=1.0,
                                  help="Une fois l'ouverture connue, le modèle intègre le gap et gagne en précision.")

st.sidebar.header("3. Produits et risque")
levs = st.sidebar.multiselect("Leviers testés", [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200],
                              default=[10, 20, 30, 50, 100])
stops = st.sidebar.multiselect("Stops testés (distance sur le sous-jacent, en fraction)",
                               [0.002, 0.003, 0.004, 0.005, 0.0075, 0.01], default=[0.003, 0.005])
cost_bp = st.sidebar.number_input("Coût aller-retour (pb du sous-jacent)", min_value=0.0, value=3.0, step=0.5)
risk_eur = st.sidebar.number_input("Perte maximale par trade (EUR)", min_value=1.0, value=10.0, step=1.0)
capital = st.sidebar.number_input("Enveloppe de trading (EUR)", min_value=100.0, value=1000.0, step=100.0)
alpha = st.sidebar.selectbox("Seuil de significativité (après correction de Holm)", [0.01, 0.05, 0.10], index=1)

with st.sidebar.expander("Réglages avancés"):
    use_vol = st.checkbox("Analogues sur le régime de volatilité", value=False)
    use_wd = st.checkbox("Favoriser le même jour de semaine", value=False,
                         help="Effet non significatif dans nos tests : à utiliser en connaissance de cause.")
    n_boot = st.slider("Tirages bootstrap", 200, 3000, 1000, step=100)
    bw = st.slider("Largeur du noyau (x Silverman)", 0.5, 2.0, 1.0, step=0.1)

go = st.sidebar.button("Lancer l'analyse", type="primary")

# ------------------------------------------------------------------------------------------ page principale
st.title("Aide à la décision statistique : turbos intraday")
st.caption(DISCLAIMER)
q = quality_report(df)
with st.expander(f"Données : {df.attrs.get('name', '')} | {q.summary()}"):
    st.dataframe(q.by_year.tail(15))
    st.write("Les ouvertures ne sont fiables qu'à partir de la date indiquée : avant, elles sont recopiées de la clôture "
             "de la veille et n'alimentent pas les calculs d'ouverture, de gap et de KO.")

if go:
    try:
        settings = Settings(leverages=tuple(float(x) for x in levs), stops=tuple(stops), cost_bp=cost_bp,
                            risk_eur=risk_eur, capital=capital, alpha=alpha, n_boot=n_boot, use_vol_context=use_vol,
                            use_weekday=use_wd, bw_scale=bw)
        with st.spinner("Calcul en cours..."):
            st.session_state["rep"] = analyse(df, pd.Timestamp(session_date), open_in if open_in > 0 else None, settings)
    except Exception as e:  # noqa: BLE001
        st.session_state.pop("rep", None)
        st.error(str(e))

rep = st.session_state.get("rep")
if rep is None:
    st.info("Règle le contexte dans la barre latérale puis clique sur « Lancer l'analyse ».")
    st.stop()

for w in rep.warnings:
    st.warning(w)

v = rep.verdict
if v["trade"]:
    st.success(f"Un produit se démarque : {v['produit']}. {v['reason']}")
else:
    st.warning("Verdict : ne pas trader sur un avantage statistique. " + v["reason"])

d, a, c = rep.direction, rep.amplitude, rep.context
k50 = rep.ko.iloc[(rep.ko["levier"] - 50).abs().argmin()]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Régime de volatilité", regime_label(a["vol_percentile"]), f"percentile {a['vol_percentile'] * 100:.0f}", delta_color="off")
m2.metric("Amplitude médiane attendue", f"{a['range_median'] * 100:.2f} %", f"90 % sous {a['range_q90'] * 100:.2f} %", delta_color="off")
m3.metric("P(hausse), ouverture vers clôture", f"{d['p_up'] * 100:.1f} %", f"sans contexte {d['p_up_uncond'] * 100:.1f} %", delta_color="off")
m4.metric(f"KO levier {k50['levier']:g}", f"long {k50['P_KO_long'] * 100:.1f} %", f"short {k50['P_KO_short'] * 100:.1f} %", delta_color="off")

tabs = st.tabs(["Distribution", "Risque de KO", "Produits", "Validation", "Rapport complet"])

with tabs[0]:
    r = rep.scen["r_oc"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(r, bins=80, weights=rep.weights, range=(-4, 4), color="#4c78a8")
    ax.axvline(0, color="k", lw=1)
    qs = d["quantiles"]
    for key in (5, 50, 95):
        ax.axvline(qs[key] * 100, color="#e45756", ls="--", lw=1)
    ax.set_xlabel("Variation ouverture vers clôture (%)")
    ax.set_ylabel("Probabilité")
    ax.set_title("Scénarios pondérés pour la séance (pointillés : quantiles 5, 50, 95 %)")
    st.pyplot(fig)
    st.write(f"P(hausse) = {d['p_up'] * 100:.1f} % (IC90 {d['p_up_lo'] * 100:.1f} à {d['p_up_hi'] * 100:.1f}). "
             f"{'Écart notable avec' if d['distinguable'] else 'Non distinguable de'} la fréquence sans contexte "
             f"({d['p_up_uncond'] * 100:.1f} %).")
    st.write(f"Contexte : veille {c['ret_prev'] * 100:+.2f} %"
             + (f", gap {c['gap'] * 100:+.2f} %" if c["gap"] == c["gap"] else ", gap inconnu")
             + f" | analogues sur {', '.join(rep.ctx_cols)} | ESS final {rep.ess_final:.0f} sur {rep.n_sample} séances.")

with tabs[1]:
    ko = rep.ko
    chart = pd.DataFrame({"KO long %": ko["P_KO_long"] * 100, "KO short %": ko["P_KO_short"] * 100}, index=[f"x{x:g}" for x in ko["levier"]])
    st.bar_chart(chart)
    st.dataframe(display_ko(ko), hide_index=True)
    st.caption("Probabilité que le prix touche la barrière (1/levier) depuis l'ouverture, tirée de l'historique standardisé "
               "par la volatilité prévue du jour. IC90 par bootstrap.")

with tabs[2]:
    st.dataframe(display_grid(rep.grid.sort_values("EV_pct_mise", ascending=False)), hide_index=True)
    st.caption("Espérance en % de la mise, nette de coûts. 'p ajusté' tient compte du nombre de produits testés : "
               "un produit n'est retenu que si p ajusté est sous le seuil ET l'espérance est positive.")

with tabs[3]:
    if st.button("Lancer la validation walk-forward"):
        st.session_state["val"] = _validate(df)
    val = st.session_state.get("val")
    if val is None:
        st.info("La validation ré-estime tout année par année, uniquement avec le passé, pour mesurer ce que l'outil sait vraiment prévoir.")
    else:
        st.code(format_validation(val))
        rel = val["ko_fiabilité"]
        fig, ax = plt.subplots(figsize=(5, 4))
        for (sens, b), grp in rel.groupby(["sens", "barrière_%"]):
            ax.plot(grp["prévue_%"], grp["réalisée_%"], marker="o", label=f"{sens} {b:.0f}%")
        mx = max(rel["prévue_%"].max(), rel["réalisée_%"].max())
        ax.plot([0, mx], [0, mx], "k--", lw=1)
        ax.set_xlabel("KO prévu (%)")
        ax.set_ylabel("KO réalisé (%)")
        ax.set_title("Fiabilité hors échantillon (diagonale = parfait)")
        ax.legend()
        st.pyplot(fig)

with tabs[4]:
    txt = format_report(rep)
    st.code(txt)
    st.download_button("Télécharger le rapport", txt, file_name=f"rapport_{rep.session_date.date()}.txt")

