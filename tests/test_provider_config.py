"""Config/URL plumbing tests (no network). Synthetic values only."""
import os

from paszel.ai import OpenAICompatProvider, load_dotenv_file


def test_openai_compat_url_has_no_double_slash(monkeypatch):
    monkeypatch.setenv("PASZEL_LLM_API_KEY", "k"); monkeypatch.setenv("PASZEL_LLM_MODEL", "m")
    monkeypatch.setenv("PASZEL_LLM_BASE_URL", "https://example.test/v1beta/openai/")
    seen = {}
    p = OpenAICompatProvider()
    p._post = lambda url, headers, body: seen.update(url=url, auth=headers) or {"choices": [{"message": {"content": "{}"}}]}
    assert p.complete("s", "u") == "{}"
    assert seen["url"] == "https://example.test/v1beta/openai/chat/completions"
    assert seen["auth"]["authorization"] == "Bearer k"


def test_dotenv_loader_respects_existing_env_and_comments(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# c\nSYNTH_A=1 # inline\nSYNTH_B=\"two\"\nSYNTH_C=keep\n")
    for k in ("SYNTH_A", "SYNTH_B"): monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYNTH_C", "already")
    assert sorted(load_dotenv_file(str(f))) == ["SYNTH_A", "SYNTH_B"]
    assert os.environ["SYNTH_A"] == "1" and os.environ["SYNTH_B"] == "two" and os.environ["SYNTH_C"] == "already"
    for k in ("SYNTH_A", "SYNTH_B"): os.environ.pop(k, None)


def test_dotenv_loader_tolerates_bom_utf16_crlf_export(tmp_path, monkeypatch):  # SYNTHETIC values
    for k in ("SYNTH_X", "SYNTH_Y"): monkeypatch.delenv(k, raising=False)
    a = tmp_path / "a.env"; a.write_bytes("\ufeffexport SYNTH_X=1\r\nSYNTH_Y=2\r\n".encode("utf-8"))
    assert sorted(load_dotenv_file(str(a))) == ["SYNTH_X", "SYNTH_Y"] and os.environ["SYNTH_X"] == "1"
    for k in ("SYNTH_X", "SYNTH_Y"): os.environ.pop(k, None)
    b = tmp_path / "b.env"; b.write_bytes("SYNTH_X=7\n".encode("utf-16"))
    assert load_dotenv_file(str(b)) == ["SYNTH_X"] and os.environ["SYNTH_X"] == "7"
    os.environ.pop("SYNTH_X", None)
    assert load_dotenv_file(str(tmp_path / "missing")) == []


def test_unconfigured_app_explains_itself(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    for k in ("PASZEL_LLM_PROVIDER", "PASZEL_LLM_MODEL", "PASZEL_LLM_API_KEY"): monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PASZEL_STATE", str(tmp_path / "s.json"))
    app = Path(__file__).parent.parent / "app.py"
    if (app.parent / ".env").exists(): return  # a developer .env would configure it; nothing to assert
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert any("NOT CONFIGURED" in c.value for c in at.sidebar.caption)
    assert any("Model" == t.label for t in at.sidebar.text_input)
