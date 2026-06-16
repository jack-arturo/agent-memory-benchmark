# Reproducing the AutoMem results

AutoMem is a graph + vector memory service (FalkorDB + Qdrant behind a Flask API).
The `automem` provider is **self-spinning**: it brings the whole stack up via Docker,
runs the benchmark, and tears it down. You need **Docker** and a `GEMINI_API_KEY`
(the same shared answer/judge key every provider uses) — **no embedding API keys**.

## One command

```bash
GEMINI_API_KEY=... \
OMB_ANSWER_LLM=gemini OMB_ANSWER_MODEL=gemini-3.1-pro-preview \
OMB_JUDGE_LLM=gemini OMB_JUDGE_MODEL=gemini-2.5-flash-lite \
uv run omb run --memory automem --dataset locomo --split locomo10
```

The provider's `initialize()` runs `docker compose up` on
`src/memory_bench/memory/automem_compose.yml` (AutoMem image + FalkorDB + Qdrant),
waits for `/health`, and `cleanup()` runs `docker compose down -v`. Ports are chosen
per-run, so concurrent/repeat runs don't collide. A crash also tears the stack down
(registered via `atexit`).

## Pinned configuration (matches the committed `outputs/`)

| Knob | Value |
|---|---|
| AutoMem image | `ghcr.io/verygoodplugins/automem:amb-v1` (override with `AUTOMEM_IMAGE`) |
| Embeddings | FastEmbed local, `BAAI/bge-base-en-v1.5`, 768-dim (`EMBEDDING_PROVIDER=local`, no API key) |
| Answer LLM | `gemini-3.1-pro-preview` (matches the board's runs) |
| Judge LLM | `gemini-2.5-flash-lite` (matches the board's runs) |
| Mode | `rag` |
| Datasets | `locomo/locomo10`, `longmemeval/s`, `personamem/32k` (full splits) |

## How the provider uses AutoMem

- **Ingest** → `POST /memory`, one memory per document. Documents longer than AutoMem's
  2000-char limit are chunked at `AUTOMEM_MAX_CHARS` (default 1800) on sentence/paragraph
  boundaries; timestamps are backdated to the source. After ingest the provider waits for
  AutoMem's enrichment queue to settle so the graph it queries is fully built.
- **Retrieve** → `GET /recall`, scoped to the run's tags, with graph relation expansion on
  (`expand_relations` + `expand_respect_tags`). Content is read from
  `result["memory"]["content"]`.

## Tuning knobs (env)

| Env | Default | Purpose |
|---|---|---|
| `AUTOMEM_IMAGE` | `ghcr.io/verygoodplugins/automem:amb-v1` | AutoMem image tag |
| `AUTOMEM_MAX_CHARS` | `1800` | chunk size (under AutoMem's 2000 hard limit) |
| `AUTOMEM_RECALL_K` | (harness `k`) | override retrieval depth |
| `AUTOMEM_ENRICH_SETTLE_SECONDS` | `120` | max wait for enrichment to drain after ingest |

## Tests

```bash
uv run --with pytest pytest tests/test_automem_provider.py tests/test_gemini_retry.py
```
(The provider's HTTP contract and helpers are unit-tested without Docker.)
