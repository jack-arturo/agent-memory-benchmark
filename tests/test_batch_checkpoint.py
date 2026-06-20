import os
import json

os.environ.setdefault("GEMINI_API_KEY", "test-dummy")

from memory_bench import runner as runner_mod
from memory_bench.runner import EvalRunner
from memory_bench.dataset.base import Dataset
from memory_bench.memory.base import MemoryProvider
from memory_bench.modes.base import ResponseMode
from memory_bench.models import Query, Document, AnswerResult


class _FakeLLM:
    model_id = "fake-judge"


class _FakeJudge:
    def __init__(self, llm=None):
        self._llm = _FakeLLM()

    def score(self, *a, **k):
        from memory_bench.models import JudgeResult
        return JudgeResult(correct=True, reason="fake")


class _StubDataset(Dataset):
    name = "stubds"
    description = "stub"
    splits = ["s"]
    task_type = "mcq"          # mcq => exact-letter scoring, no judge network call
    isolation_unit = None      # => BATCH mode (the path under test)

    def __init__(self, n):
        self._n = n

    def load_queries(self, split, category=None, limit=None):
        qs = [Query(id=f"q{i}", query=f"q{i}", gold_ids=[], gold_answers=["A"]) for i in range(self._n)]
        return qs[:limit] if limit else qs

    def load_documents(self, split, category=None, limit=None, ids=None, user_ids=None):
        return [Document(id="d0", content="ctx", user_id=None)]

    def supports_oracle(self):
        return False


class _StubMemory(MemoryProvider):
    name = "stubmem"
    description = "stub"
    kind = "local"
    concurrency = 4

    def ingest(self, documents):
        pass

    def retrieve(self, query, k=10, user_id=None, query_timestamp=None):
        return ([Document(id="d0", content="ctx")], None)


class _StubMode(ResponseMode):
    name = "stub"
    description = "stub"

    def __init__(self, seen):
        self._seen = seen

    @property
    def llm_id(self):
        return "stub-answerer"

    def answer(self, query, memory, task_type="open", user_id=None):
        self._seen.append(query)                       # record which queries were answered
        return AnswerResult(answer="A", reasoning="", context="ctx", retrieve_time_ms=1.0)

    def answer_from_context(self, query, context, task_type="open"):
        return AnswerResult(answer="A", reasoning="", context=context, retrieve_time_ms=0.0)


def _make_runner(tmp_path, monkeypatch, save_calls):
    monkeypatch.setattr(runner_mod, "GeminiJudge", _FakeJudge)
    r = EvalRunner(output_dir=tmp_path / "outputs")
    orig_save = r._save

    def spy(summary):
        save_calls.append(summary.total_queries)
        return orig_save(summary)

    monkeypatch.setattr(r, "_save", spy)
    return r


def test_batch_mode_saves_incrementally(tmp_path, monkeypatch):
    save_calls = []
    r = _make_runner(tmp_path, monkeypatch, save_calls)
    seen = []
    r.run(dataset=_StubDataset(25), split="s", memory=_StubMemory(),
          mode=_StubMode(seen), run_name="t1")
    # Incremental saves fired before the final save (25 queries, SAVE_EVERY=10 => 10, 20, final)
    assert any(0 < n < 25 for n in save_calls), f"no partial save observed: {save_calls}"
    assert len(seen) == 25


def test_batch_mode_resume_skips_done(tmp_path, monkeypatch):
    # Pre-write a partial output file with q0..q9 already answered.
    # Mode dir is the stub mode's name ("stub"), matching _output_path(... mode.name).
    out = tmp_path / "outputs" / "stubds" / "t2" / "stub"
    out.mkdir(parents=True)
    prior = {"dataset": "stubds", "split": "s", "category": None,
             "memory_provider": "stubmem", "run_name": "t2", "mode": "rag",
             "oracle": False, "total_queries": 10, "correct": 10, "accuracy": 1.0,
             "ingestion_time_ms": 0.0, "ingested_docs": 1,
             "results": [{"query_id": f"q{i}", "query": f"q{i}", "answer": "A",
                          "reasoning": "", "context": "ctx", "context_tokens": 1,
                          "retrieve_time_ms": 0.0, "gold_answers": ["A"],
                          "correct": True, "judge_reason": "", "score": None,
                          "meta": {}, "raw_response": None, "category_axes": {}}
                         for i in range(10)]}
    (out / "s.json").write_text(json.dumps(prior))

    save_calls = []
    r = _make_runner(tmp_path, monkeypatch, save_calls)
    seen = []
    r.run(dataset=_StubDataset(25), split="s", memory=_StubMemory(),
          mode=_StubMode(seen), run_name="t2", skip_ingested=True)
    # Only q10..q24 should be answered this run; q0..q9 skipped.
    assert sorted(seen) == [f"q{i}" for i in range(10, 25)], f"resume answered wrong set: {sorted(seen)}"
    # Final file contains all 25 (merge preserved the prior 10).
    final = json.loads((out / "s.json").read_text())
    assert final["total_queries"] == 25
