from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()


def test_full_dashboard_renders_and_profile_updates():
    at = app()
    assert not at.exception
    assert len(at.tabs) == 6
    assert at.metric[0].value == "395"
    old = [m.value for m in at.markdown if 'class="score"' in m.value][0]
    profile_key = next(n.key for n in at.number_input if n.key.startswith("predict_G2"))
    at.number_input(key=profile_key).set_value(18).run()
    assert not at.exception
    new = [m.value for m in at.markdown if 'class="score"' in m.value][0]
    assert new != old


def test_settings_require_training_and_simple_mode_works():
    at = app()
    at.selectbox(key="preset").select("Simple regression").run()
    assert not at.exception
    assert any("settings have changed" in item.value for item in at.info)
    at.button(key="train").click().run()
    assert not at.exception
    assert len(at.select_slider) == 1
    assert not any(n.key.startswith("predict_") for n in at.number_input)
    feature_key = next(m.key for m in at.multiselect)
    at.multiselect(key=feature_key).set_value([]).run()
    at.button(key="train").click().run()
    assert not at.exception
    assert any("at least one" in e.value for e in at.error)


def test_subject_switch_and_upload_empty_state():
    at = app()
    at.selectbox(key="source").select("UCI · Portuguese").run()
    at.button(key="train").click().run()
    assert not at.exception
    assert at.metric[0].value == "649"
    at.selectbox(key="source").select("Upload a CSV").run()
    assert not at.exception
    assert any("Upload a CSV" in info.value for info in at.info)
