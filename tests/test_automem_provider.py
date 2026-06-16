import io, json
from contextlib import contextmanager
from memory_bench.memory.automem import (
    _chunk_text, _extract_content, _free_port, AutoMemMemoryProvider,
)
from memory_bench.models import Document
import memory_bench.memory.automem as m


def test_chunk_long_text_under_limit():
    text = " ".join(f"sentence {i}." for i in range(600))
    assert len(text) > 1800
    chunks = _chunk_text(text, 1800)
    assert len(chunks) > 1
    assert all(len(c) <= 1800 for c in chunks)

def test_chunk_short_text_single():
    assert _chunk_text("hello world", 1800) == ["hello world"]

def test_extract_content_from_nested_memory():
    assert _extract_content({"id": "m1", "memory": {"content": "answer", "summary": "s"}}) == "answer"

def test_extract_content_falls_back_to_summary():
    assert _extract_content({"id": "m1", "memory": {"content": "", "summary": "fallback"}}) == "fallback"

def test_extract_content_empty_when_missing():
    assert _extract_content({"id": "m1"}) == ""

def test_free_port_returns_open_port():
    p = _free_port()
    assert 1024 < p < 65536


class _FakeHTTP:
    def __init__(self, responses):
        self.responses = responses; self.calls = []
    @contextmanager
    def urlopen(self, req, timeout=None):
        body = req.data.decode() if req.data else None
        self.calls.append((req.get_method(), req.full_url, dict(req.headers), body))
        payload = self.responses.pop(0) if self.responses else {}
        yield io.BytesIO(json.dumps(payload).encode())


def test_ingest_chunks_and_posts(monkeypatch):
    fake = _FakeHTTP([{"id": str(i)} for i in range(40)])
    monkeypatch.setattr(m.urllib.request, "urlopen", fake.urlopen)
    p = AutoMemMemoryProvider()
    p._endpoint = "http://x:8001"; p._token = "t"; p._run_tag = "ambrun-test"; p._enrich_settle_s = 0
    long_doc = Document(id="d", content=" ".join(f"s{i}." for i in range(600)), user_id="u1")
    p.ingest([long_doc])
    stores = [c for c in fake.calls if c[1].endswith("/memory")]
    assert len(stores) > 1
    for _, _, _, body in stores:
        payload = json.loads(body)
        assert len(payload["content"]) <= 1800
        assert "ambrun-test" in payload["tags"]

def test_retrieve_extracts_nested_content(monkeypatch):
    fake = _FakeHTTP([{"results": [{"id": "m9", "memory": {"content": "answer"}}]}])
    monkeypatch.setattr(m.urllib.request, "urlopen", fake.urlopen)
    p = AutoMemMemoryProvider()
    p._endpoint = "http://x:8001"; p._token = "t"; p._run_tag = "ambrun-test"
    docs, _ = p.retrieve("q", k=5, user_id="u1")
    assert docs[0].content == "answer"
    assert "expand_relations=true" in fake.calls[0][1]
