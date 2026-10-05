import io

import pytest

rich_console = pytest.importorskip("rich.console")

from myagy.terminal import ui


def _capture(monkeypatch, width=60):
    buf = io.StringIO()
    monkeypatch.setattr(ui, "console", rich_console.Console(file=buf, force_terminal=False, width=width))
    return buf


def test_markdown_rendered_when_enabled(monkeypatch):
    buf = _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True)
    for tok in ["# Title\n\n", "- item one\n", "- item two\n"]:
        r.feed(tok)
    r.stop()
    out = buf.getvalue()
    assert "Title" in out and "item one" in out
    assert "# Title" not in out  # heading marker consumed by the renderer


def test_raw_mode_streams_tokens(monkeypatch, capsys):
    _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=False)
    r.feed("hello ")
    r.feed("world")
    r.stop()
    assert capsys.readouterr().out == "hello world\n"


def test_disabled_renderer_prints_nothing(monkeypatch, capsys):
    buf = _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True, enabled=False)
    r.feed("secret")
    r.stop()
    assert capsys.readouterr().out == "" and buf.getvalue() == ""


def test_stop_is_idempotent(monkeypatch):
    _capture(monkeypatch)
    r = ui.StreamRenderer(markdown=True)
    r.feed("x")
    r.stop()
    r.stop()
