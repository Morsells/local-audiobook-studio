from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def app_text():
    return (ROOT / "app.py").read_text(encoding="utf-8")


def test_app_parses():
    ast.parse(app_text())


def test_header_is_versionless():
    text = app_text()
    assert 'st.title("Local Audiobook Studio")' in text
    assert 'page_title="Local Audiobook Studio"' in text
    assert 'Local Audiobook Studio V3.7.6' not in text


def test_sidebar_is_clean():
    text = app_text()
    start = text.index("with st.sidebar:")
    end = text.index('st.subheader("Open a book")', start)
    sidebar = text[start:end]
    assert "Piper German" not in sidebar
    assert "Kikiri German" not in sidebar
    assert "Chatterbox V3" in sidebar
    assert "Kokoro" in sidebar
    assert "Optional engine" in sidebar


def test_visible_tts_choices():
    text = app_text()
    start = text.index("visible_tts_engines = [")
    end = text.index("]", start)
    selector = text[start:end]
    assert "TTSEngine.CHATTERBOX" in selector
    assert "TTSEngine.KOKORO" in selector
    assert "TTSEngine.QWEN3" in selector
    assert "TTSEngine.PIPER" not in selector
    assert "TTSEngine.KIKIRI" not in selector


def test_project_status_has_no_kikiri_card():
    text = app_text()
    project = text.split("with tab_project:", 1)[1]
    assert "**Kikiri German sidecar**" not in project
    assert "**Qwen3-TTS · optional**" in project


def test_toolbar_minimal():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'toolbarMode = "minimal"' in cfg
