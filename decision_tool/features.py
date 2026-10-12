"""Construction des variables journalières (rendements, gap, excursions, proxy de volatilité)."""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_FLOOR = 5e-4  # plancher de 5 pb pour le log du |gap|


def build_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables dérivées. Toutes les grandeurs sont arithmétiques (fractions, pas en %).

    ret_cc    : variation clôture à clôture
    gap       : ouverture / clôture veille - 1
    r_oc      : variation ouverture -> clôture (le trade d'un day-trader)
    adv_long  : excursion défavorable d'un long pris à l'ouverture : (Open - Low) / Open
    adv_short : excursion défavorable d'un short pris à l'ouverture : (High - Open) / Open
    sig_pk    : écart-type journalier de Parkinson, ln(High/Low) / (2 sqrt(ln 2))
    """
    d = ohlc[["Open", "High", "Low", "Close"]].copy()
    d["prev_close"] = d["Close"].shift(1)
    d["ret_cc"] = d["Close"] / d["prev_close"] - 1
    d["gap"] = d["Open"] / d["prev_close"] - 1
    d["r_oc"] = d["Close"] / d["Open"] - 1
    d["adv_long"] = (d["Open"] - d["Low"]) / d["Open"]
    d["adv_short"] = (d["High"] - d["Open"]) / d["Open"]
    lhl = np.log(d["High"] / d["Low"])
    d["flat"] = lhl < 1e-6
    d["sig_pk"] = (lhl / (2 * np.sqrt(np.log(2)))).where(~d["flat"])
    d["wd"] = d.index.dayofweek
    d.attrs.update(ohlc.attrs)
    return d


def add_session(frame: pd.DataFrame, date, open_price: float | None = None) -> pd.DataFrame:
    """Ajoute une ligne vide pour la séance à analyser (High/Low/Close inconnus).

    Si l'ouverture est connue, le gap est calculé : le modèle post-ouverture devient utilisable.
    """
    date = pd.Timestamp(date)
    if date <= frame.index[-1]:
        raise ValueError(f"La date {date.date()} doit être postérieure à la dernière séance du fichier "
                         f"({frame.index[-1].date()}). Ajoute les séances manquantes avant d'analyser.")
    row = {c: np.nan for c in frame.columns}
    row["flat"] = False
    row["prev_close"] = frame["Close"].iloc[-1]
    row["wd"] = date.dayofweek
    if open_price is not None:
        row["Open"] = float(open_price)
        row["gap"] = float(open_price) / row["prev_close"] - 1
    out = pd.concat([frame, pd.DataFrame([row], index=pd.DatetimeIndex([date], name=frame.index.name))])
    out["flat"] = out["flat"].astype(bool)
    out.attrs.update(frame.attrs)
    return out