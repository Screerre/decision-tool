"""Chargement et nettoyage des historiques OHLC journaliers.

Formats acceptés : CSV (Stooq, Yahoo, courtier) et Excel (.xlsx), avec ou sans ligne de titre
avant l'en-tête. Colonnes reconnues : Date, Open, High, Low, Close (et Volume si présent).

Deux contrôles de qualité sont faits automatiquement, car ils invalident silencieusement les
statistiques sur les historiques d'indices :
  1. barres plates (High == Low) : l'amplitude n'est pas réelle ;
  2. ouverture recopiée de la clôture de la veille : le gap et le trajet depuis l'ouverture
     ne sont pas réels.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SYNONYMS = {
    "date": "Date",
    "open": "Open", "ouv": "Open", "ouverture": "Open",
    "high": "High", "haut": "High", "plus haut": "High",
    "low": "Low", "bas": "Low", "plus bas": "Low",
    "close": "Close", "clôture": "Close", "cloture": "Close", "dernier": "Close",
    "volume": "Volume", "vol.": "Volume",
}


def _norm(name) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _to_num(s: pd.Series) -> pd.Series:
    """Convertit une colonne en nombres (gère espaces insécables et virgule décimale)."""
    if s.dtype.kind in "fiu":
        return s.astype(float)
    t = s.astype(str).str.replace(r"[\s  ]", "", regex=True)
    has_comma_decimal = t.str.contains(r"^-?\d+,\d+$", regex=True).mean() > 0.5
    if has_comma_decimal:
        t = t.str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce")


def _find_header_row(raw: pd.DataFrame) -> int:
    for i in range(min(20, len(raw))):
        vals = {_SYNONYMS.get(_norm(v)) for v in raw.iloc[i].tolist()}
        if {"Date", "Open", "Close"} <= vals:
            return i
    raise ValueError(
        "En-tête introuvable : le fichier doit contenir les colonnes Date, Open, High, Low, Close."
    )


def _read_csv_raw(source) -> pd.DataFrame:
    """Lit un CSV en repérant l'en-tête (des lignes de titre peuvent le précéder) et le séparateur."""
    if hasattr(source, "read"):
        content = source.read()
    else:
        content = Path(source).read_bytes()
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    lines = text.splitlines()
    hdr = None
    for i, ln in enumerate(lines[:20]):
        low = ln.lower()
        if "date" in low and "open" in low or ("date" in low and "ouv" in low):
            hdr = i
            break
    if hdr is None:
        raise ValueError("En-tête introuvable : le fichier doit contenir les colonnes Date, Open, High, Low, Close.")
    head = lines[hdr]
    sep = max([";", ",", "\t"], key=head.count)
    return pd.read_csv(io.StringIO("\n".join(lines[hdr:])), header=None, sep=sep, dtype=str)


def load_ohlc(source, name: str | None = None) -> pd.DataFrame:
    """Charge un fichier (chemin ou buffer) et renvoie un DataFrame indexé par Date.

    Colonnes de sortie : Open, High, Low, Close, (Volume). Trié, sans doublons, sans lignes vides.
    """
    suffix = Path(getattr(source, "name", str(source))).suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(source, header=None)
    else:
        raw = _read_csv_raw(source)
    h = _find_header_row(raw)
    cols = [_SYNONYMS.get(_norm(v), str(v)) for v in raw.iloc[h].tolist()]
    df = raw.iloc[h + 1:].copy()
    df.columns = cols
    missing = {"Date", "Open", "High", "Low", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    for c in keep[1:]:
        df[c] = _to_num(df[c])
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    df = df.drop_duplicates("Date", keep="last").sort_values("Date").set_index("Date")
    df.index.name = "Date"
    df.attrs["name"] = name or Path(getattr(source, "name", str(source))).stem
    return df


def append_rows(df: pd.DataFrame, text: str) -> pd.DataFrame:
    """Ajoute des séances collées à la main : une ligne par séance 'Date,Open,High,Low,Close'."""
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in re.split(r"[;,\t]", line) if p.strip()]
        if len(parts) < 5 or not re.match(r"\d", parts[0]):
            continue
        rows.append([parts[0]] + [float(p.replace(" ", "")) for p in parts[1:5]])
    if not rows:
        return df
    add = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    add["Date"] = pd.to_datetime(add["Date"])
    add = add.set_index("Date")
    out = pd.concat([df, add]).loc[lambda x: ~x.index.duplicated(keep="last")].sort_index()
    out.attrs.update(df.attrs)
    return out


@dataclass
class Quality:
    """Résultat du contrôle de qualité d'un historique."""
    n: int
    first: pd.Timestamp
    last: pd.Timestamp
    range_start: pd.Timestamp      # à partir de cette date, High/Low sont exploitables
    open_start: pd.Timestamp       # à partir de cette date, l'ouverture est une vraie ouverture
    n_incoherent: int              # lignes où High < max(O,C) ou Low > min(O,C)
    by_year: pd.DataFrame

    def summary(self) -> str:
        return (f"{self.n} séances du {self.first.date()} au {self.last.date()} | amplitudes fiables dès "
                f"{self.range_start.date()} | ouvertures fiables dès {self.open_start.date()} | "
                f"{self.n_incoherent} ligne(s) OHLC incohérente(s)")


def _first_good_year(good: pd.Series, n: pd.Series, min_n: int = 20) -> int:
    """Plus petite année Y telle que toutes les années >= Y (avec assez de séances) sont bonnes."""
    ok = good.where(n >= min_n, True)
    start = None
    for y in reversed(list(ok.index)):
        if ok[y]:
            start = y
        else:
            break
    return int(start) if start is not None else int(ok.index[-1]) + 1


def quality_report(df: pd.DataFrame, flat_max: float = 0.05, open_copy_max: float = 0.10) -> Quality:
    d = df.copy()
    prev = d["Close"].shift(1)
    d["flat"] = (d["High"] - d["Low"]).abs() < 1e-12 * d["Close"]
    d["open_is_prev"] = (d["Open"] / prev - 1).abs() < 1e-6
    d["incoh"] = (d["High"] < d[["Open", "Close"]].max(axis=1) * (1 - 1e-9)) | \
                 (d["Low"] > d[["Open", "Close"]].min(axis=1) * (1 + 1e-9))
    g = d.groupby(d.index.year)
    by_year = pd.DataFrame({"n": g.size(), "flat_share": g["flat"].mean(),
                            "open_equals_prev_close_share": g["open_is_prev"].mean()})
    y_range = _first_good_year(by_year["flat_share"] < flat_max, by_year["n"])
    y_open = _first_good_year(by_year["open_equals_prev_close_share"] < open_copy_max, by_year["n"])
    y_open = max(y_open, y_range)
    last_year = int(by_year.index.max())
    return Quality(
        n=len(d), first=d.index[0], last=d.index[-1],
        range_start=pd.Timestamp(year=min(y_range, last_year + 1), month=1, day=1),
        open_start=pd.Timestamp(year=min(y_open, last_year + 1), month=1, day=1),
        n_incoherent=int(d["incoh"].sum()), by_year=by_year.round(3),
    )


def clean_incoherent(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les lignes OHLC incohérentes (High < max(O,C) ou Low > min(O,C))."""
    bad = (df["High"] < df[["Open", "Close"]].max(axis=1) * (1 - 1e-9)) | \
          (df["Low"] > df[["Open", "Close"]].min(axis=1) * (1 + 1e-9))
    out = df.loc[~bad].copy()
    out.attrs.update(df.attrs)
    return out

