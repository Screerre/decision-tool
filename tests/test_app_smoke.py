"""Test de fumée de l'application : exécute le script avec un faux module streamlit.

Il détecte les erreurs Python (noms, types, clés) mais pas les erreurs d'usage de l'API Streamlit :
pour cela, lance réellement `streamlit run app/streamlit_app.py`.
"""
import runpy
import sys
import types
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


class _Stop(Exception):
    pass


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return lambda *a, **k: None


def _make_streamlit():
    st = types.ModuleType("streamlit")
    st.calls = []

    def cache_data(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f

    def selectbox(label, options, index=0, **k):
        return options[index]

    def multiselect(label, options, default=None, **k):
        return default or []

    def number_input(label, min_value=None, value=0.0, **k):
        return value

    def date_input(label, value=None, **k):
        return value

    def slider(label, mn, mx, value, **k):
        return value

    def checkbox(label, value=False, **k):
        return value

    def text_area(label, value="", **k):
        return value

    def stop():
        raise _Stop()

    def noop(*a, **k):
        return None

    st.cache_data = cache_data
    st.sidebar = types.SimpleNamespace(
        header=noop, selectbox=selectbox, file_uploader=lambda *a, **k: None, text_area=text_area,
        date_input=date_input, number_input=number_input, multiselect=multiselect, button=lambda *a, **k: True,
        expander=lambda *a, **k: _Ctx(), error=noop, slider=slider, checkbox=checkbox)
    for name in ("set_page_config", "title", "caption", "write", "info", "warning", "success", "error", "metric",
                 "dataframe", "pyplot", "bar_chart", "code", "download_button", "header"):
        setattr(st, name, noop)
    st.selectbox, st.multiselect, st.number_input, st.date_input = selectbox, multiselect, number_input, date_input
    st.slider, st.checkbox, st.text_area = slider, checkbox, text_area
    st.button = lambda *a, **k: True
    st.stop = stop
    st.expander = lambda *a, **k: _Ctx()
    st.spinner = lambda *a, **k: _Ctx()
    st.columns = lambda n, **k: [_Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    st.tabs = lambda labels: [_Ctx() for _ in labels]
    st.session_state = {}
    return st


@unittest.skipUnless((ROOT / "data" / "nasdaq100.xlsx").exists(), "fichier de données absent")
class TestAppSmoke(unittest.TestCase):
    def test_app_runs_end_to_end(self):
        stub = _make_streamlit()
        saved = sys.modules.get("streamlit")
        sys.modules["streamlit"] = stub
        try:
            runpy.run_path(str(ROOT / "app" / "streamlit_app.py"), run_name="__main__")
        except _Stop:
            self.fail("L'application s'est arrêtée avant d'afficher les résultats.")
        finally:
            if saved is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = saved
        self.assertIn("rep", stub.session_state)
        self.assertIn("val", stub.session_state)


if __name__ == "__main__":
    unittest.main()