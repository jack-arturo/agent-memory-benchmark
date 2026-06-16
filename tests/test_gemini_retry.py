"""Regression: Gemini's _generate_raw must retry transient connection errors,
not just 429/503. Full-split benchmark runs (hours) hit ReadError / connection
reset, which the 429/503-only predicate previously re-raised and crashed the run.
"""
from memory_bench.llm import gemini


class _FakeModels:
    def __init__(self, errors):
        self.errors = errors
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return type("Resp", (), {"text": "ok", "candidates": []})()


def _make_llm(models):
    llm = gemini.GeminiLLM.__new__(gemini.GeminiLLM)
    llm._model = "gemini-2.5-flash-lite"
    llm._client = type("Client", (), {"models": models})()
    return llm


def test_connection_reset_is_retried(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda *_: None)
    models = _FakeModels(["[Errno 54] Connection reset by peer (ReadError)"])
    llm = _make_llm(models)
    resp = llm._generate_raw("hi")
    assert models.calls == 2  # first raised connection error, retried, second succeeded
    assert resp.text == "ok"


def test_non_retryable_error_still_raises(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda *_: None)
    models = _FakeModels(["ValueError: bad request 400"])
    llm = _make_llm(models)
    try:
        llm._generate_raw("hi")
        raised = False
    except Exception:
        raised = True
    assert raised  # 400-class errors are not retried
    assert models.calls == 1
