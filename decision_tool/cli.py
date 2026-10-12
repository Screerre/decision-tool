"""Ligne de commande : python -m decision_tool.cli --file data/nasdaq100.xlsx --date 2026-10-12"""
from __future__ import annotations

import argparse

from .data import append_rows, load_ohlc, quality_report
from .decision import Settings, analyse
from .report import format_report, format_validation
from .validation import run_all


def _stops(txt: str) -> tuple:
    return tuple(float(x) for x in txt.split(",") if x.strip() and x.strip().lower() != "none")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Aide à la décision statistique pour le trading intraday de turbos.")
    p.add_argument("--file", required=True, help="CSV ou xlsx OHLC journalier (Date, Open, High, Low, Close)")
    p.add_argument("--date", required=True, help="Date de la séance à analyser (AAAA-MM-JJ)")
    p.add_argument("--open", type=float, default=None, help="Ouverture du jour si connue (active le mode post-ouverture)")
    p.add_argument("--add", default="", help="Séances à ajouter : 'AAAA-MM-JJ,O,H,L,C;AAAA-MM-JJ,O,H,L,C'")
    p.add_argument("--leverages", default="10,20,30,50,100")
    p.add_argument("--stops", default="0.003,0.005", help="distances de stop en fraction, séparées par des virgules")
    p.add_argument("--cost-bp", type=float, default=3.0)
    p.add_argument("--risk-eur", type=float, default=10.0)
    p.add_argument("--capital", type=float, default=1000.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--validate", action="store_true", help="Lance aussi la validation walk-forward")
    a = p.parse_args(argv)

    df = load_ohlc(a.file)
    if a.add:
        df = append_rows(df, a.add.replace(";", "\n"))
    st = Settings(leverages=tuple(float(x) for x in a.leverages.split(",")), stops=_stops(a.stops),
                  cost_bp=a.cost_bp, risk_eur=a.risk_eur, capital=a.capital, alpha=a.alpha, n_boot=a.boot)
    print(format_report(analyse(df, a.date, a.open, st)))
    if a.validate:
        print()
        print(format_validation(run_all(df, quality_report(df))))


if __name__ == "__main__":
    main()
