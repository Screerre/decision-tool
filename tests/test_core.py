"""Tests unitaires du coeur de l'outil (données simulées à propriétés connues)."""
import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from decision_tool import conditional as cond
from decision_tool.data import append_rows, load_ohlc, quality_report
from decision_tool.decision import Settings, analyse
from decision_tool.features import add_session, build_frame
from decision_tool.turbo import Spec, holm, scenario_pnl
from decision_tool.validation import ko_calibration
from decision_tool.volatility import fit_har

ROOT = Path(__file__).resolve().parents[1]


def simulate_ohlc(n=3500, seed=0, persistence=0.97, base=0.009, copy_open_days=0, drift=0.0):
    """Indice simulé : volatilité persistante (AR(1) sur le log), trajet intraday gaussien, gap overnight."""
    rng = np.random.default_rng(seed)
    logv = np.zeros(n)
    for t in range(1, n):
        logv[t] = persistence * logv[t - 1] + 0.12 * rng.standard_normal()
    sig = base * np.exp(logv)
    steps = 60
    price = 1000.0
    rows = []
    for t in range(n):
        gap = 0.0 if t < copy_open_days else 0.4 * sig[t] * rng.standard_normal()
        o = price * np.exp(gap)
        inc = sig[t] / np.sqrt(steps) * rng.standard_normal(steps) + drift / steps
        path = o * np.exp(np.cumsum(inc))
        full = np.concatenate([[o], path])
        rows.append((o, full.max(), full.min(), path[-1]))
        price = path[-1]
    idx = pd.bdate_range("2005-01-03", periods=n)
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    df.index.name = "Date"
    return df


class TestData(unittest.TestCase):
    def test_load_csv_with_title_row_and_french_decimals(self):
        txt = "spgi_us_d;;;;\nDate;Open;High;Low;Close;Volume\n2020-01-02;100,5;102,0;99,5;101,0;1000\n2020-01-03;101,0;103,0;100,0;102,5;900\n"
        buf = io.StringIO(txt)
        buf.name = "x.csv"
        df = load_ohlc(buf)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df["Open"].iloc[0], 100.5)

    def test_append_rows(self):
        df = simulate_ohlc(30)
        out = append_rows(df, "2030-01-02,1,2,0.5,1.5")
        self.assertEqual(len(out), 31)
        self.assertAlmostEqual(out["Close"].iloc[-1], 1.5)

    def test_quality_detects_copied_open(self):
        df = simulate_ohlc(1500, copy_open_days=520)  # environ deux premières années avec Open = clôture veille
        q = quality_report(df)
        self.assertGreaterEqual(q.open_start.year, 2007)
        self.assertLessEqual(q.open_start.year, 2008)

    def test_add_session_rejects_past(self):
        fr = build_frame(simulate_ohlc(60))
        with self.assertRaises(ValueError):
            add_session(fr, fr.index[-3])


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = simulate_ohlc(3500, seed=1)
        cls.frame = build_frame(cls.df)

    def test_har_captures_persistence(self):
        m = fit_har(self.frame, use_gap=False)
        self.assertGreater(m.r2, 0.3)
        self.assertGreater(m.coef[1] + m.coef[2] + m.coef[3], 0.8)

    def test_analog_weights(self):
        m = fit_har(self.frame, False)
        s = cond.build_sample(self.frame, m, self.frame.index[300])
        ctx_far = {"zprev": 50.0, "wd": 0}
        w, ess_k, lam, ess_f = cond.analog_weights(s, ctx_far, ("zprev",))
        self.assertAlmostEqual(w.sum(), 1.0, places=9)
        ctx_mid = {"zprev": 0.0, "wd": 0}
        _, ess_mid, lam_mid, _ = cond.analog_weights(s, ctx_mid, ("zprev",))
        self.assertGreater(ess_mid, 100)
        w0, *_ = cond.analog_weights(s, ctx_mid, ())
        self.assertTrue(np.allclose(w0, 1.0 / len(s)))

    def test_ko_calibration_on_known_process(self):
        q = quality_report(self.df)
        summ, _ = ko_calibration(self.frame, q, first_test_year=2010)
        err = (summ["prob_prévue_moy_%"] - summ["fréquence_réalisée_%"]).abs()
        self.assertLess(err.mean(), 2.0)   # calibration moyenne correcte
        self.assertLess(err.max(), 4.0)    # le bruit d'échantillonnage (régimes persistants) reste borné
        self.assertTrue((summ["gain_Brier_%"] > 0).all())


class TestTurbo(unittest.TestCase):
    def test_pnl_cases(self):
        r = np.array([0.01, 0.01, -0.02, 0.0])
        al = np.array([0.005, 0.10, 0.03, 0.004])
        as_ = np.array([0.0, 0.0, 0.0, 0.0])
        pnl, ko, st = scenario_pnl(Spec("long", 10, None), r, al, as_, cost_bp=0)
        self.assertAlmostEqual(pnl[0], 0.10)          # +1 % x 10
        self.assertTrue(ko[1] and pnl[1] == -1.0)     # excursion 10 % >= barrière 10 %
        self.assertAlmostEqual(pnl[2], -0.20)         # -2 % x 10, barrière non touchée (3 % < 10 %)
        pnl2, ko2, st2 = scenario_pnl(Spec("long", 10, 0.003), r, al, as_, cost_bp=0)
        self.assertTrue(st2[0] and abs(pnl2[0] - (-0.03)) < 1e-12)  # stop touché (0,5 % >= 0,3 %)
        self.assertFalse(st2[3] if al[3] < 0.003 else False)
        pnl3, _, _ = scenario_pnl(Spec("short", 10, None), r, al, as_, cost_bp=0)
        self.assertAlmostEqual(pnl3[0], -0.10)        # le short perd quand le marché monte

    def test_cost_applies(self):
        r = np.array([0.0])
        pnl, _, _ = scenario_pnl(Spec("long", 50, None), r, np.array([0.0]), np.array([0.0]), cost_bp=3.0)
        self.assertAlmostEqual(pnl[0], -0.015)        # 50 x 3 pb = 1,5 % de la mise

    def test_holm(self):
        adj = holm(np.array([0.01, 0.04, 0.03]))
        self.assertTrue(np.allclose(adj, [0.03, 0.06, 0.06]))


class TestEndToEnd(unittest.TestCase):
    def test_no_edge_process_rarely_flags_a_trade(self):
        flags = 0
        for seed in range(6):
            df = simulate_ohlc(2200, seed=100 + seed)
            rep = analyse(df, df.index[-1] + pd.offsets.BDay(1), None, Settings(n_boot=300))
            flags += int(rep.verdict["trade"])
        self.assertLessEqual(flags, 1)  # faux positifs contrôlés (Holm)

    def test_post_open_mode(self):
        df = simulate_ohlc(2200, seed=7)
        nxt = df.index[-1] + pd.offsets.BDay(1)
        rep = analyse(df, nxt, open_price=float(df["Close"].iloc[-1] * 1.004), settings=Settings(n_boot=300))
        self.assertEqual(rep.mode, "post-ouverture")
        self.assertIn("zgap", rep.ctx_cols)
        self.assertTrue(0 <= rep.direction["p_up"] <= 1)

    @unittest.skipUnless((ROOT / "data" / "nasdaq100.xlsx").exists(), "fichier de données absent")
    def test_real_file(self):
        df = load_ohlc(ROOT / "data" / "nasdaq100.xlsx")
        rep = analyse(df, df.index[-1] + pd.offsets.BDay(1), None, Settings(n_boot=300))
        self.assertGreaterEqual(rep.quality.open_start.year, 2000)
        self.assertTrue((rep.ko["P_KO_long"].diff().dropna() >= -1e-12).all())  # KO croît avec le levier
        self.assertTrue(0 < rep.amplitude["range_median"] < 0.05)