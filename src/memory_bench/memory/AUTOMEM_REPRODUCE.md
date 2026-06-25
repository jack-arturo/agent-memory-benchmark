# Reproducing the AutoMem results

AutoMem is a graph + vector memory service (FalkorDB + Qdrant behind a Flask API).
Both the AutoMem stack **and** the benchmark harness are containerized: you need only
**Docker** and a **GEMINI_API_KEY** (the same shared answer/judge key every provider on
the board uses) — **no host Python/uv toolchain and no embedding API keys**.

## One command

From the repo root, with a `GEMINI_API_KEY` in your shell or a `.env` file:

```bash
make repro-smoke      # builds the runner, spins AutoMem, scores 5 queries, tears down
```

That verifies the whole path end-to-end. For a full split, pick the dataset/split:

```bash
make repro DATASET=locomo      SPLIT=locomo10
make repro DATASET=longmemeval SPLIT=s
make repro DATASET=personamem  SPLIT=32k
make repro DATASET=beam        SPLIT=100k     # also 500k / 1m / 10m
```

Results land in `outputs/<dataset>/automem-sub/rag/<split>.json` (bind-mounted to the
host). `make repro-test` runs the provider/retry unit tests inside the image.

> Before the public image is pushed, override the service image:
> `make repro-smoke AUTOMEM_IMAGE=ghcr.io/verygoodplugins/automem:amb-local`.

### How the dockerized harness reaches AutoMem

The `runner` container (the omb harness + uv, built from `Dockerfile`) mounts the host
Docker socket. The self-spinning AutoMem provider runs `docker compose` from inside it to
bring up AutoMem + FalkorDB + Qdrant as **sibling** containers on the host daemon; their
ports are published on the host, so the runner reaches them via `host.docker.internal`
(`AUTOMEM_HOST`, set in `docker-compose.repro.yml`) on three pinned ports. A crash still
tears the stack down (registered via `atexit`).

### Host-toolchain alternative

If you already have `uv` + Python 3.11, run the harness directly — the provider still
self-spins the AutoMem stack via Docker:

```bash
GEMINI_API_KEY=... \
OMB_ANSWER_LLM=gemini OMB_ANSWER_MODEL=gemini-3.1-pro-preview \
OMB_JUDGE_LLM=gemini OMB_JUDGE_MODEL=gemini-2.5-flash-lite \
uv run omb run --memory automem --dataset locomo --split locomo10 --name automem-sub
```

`scripts/orchestrate.py` runs the entire submission matrix back-to-back on this host path
(serial — one AutoMem stack saturates a many-core host — and resume-aware via
`--skip-ingested`).

## The full submission

`--name automem-sub` keeps the committed results separate. BEAM ingests a large haystack
per conversation unit (the 10M split is ~1 GB and pegs CPU for ~20 h); runs are
checkpointed per unit and resume with `--skip-ingested`. The committed splits:
**Core-3** `locomo/locomo10`, `longmemeval/s`, `personamem/32k`; **BEAM**
`beam/{100k,500k,1m,10m}` (the 100k split run x3 as a reproducibility check).

## Pinned configuration (matches the committed `outputs/`)

| Knob | Value |
|---|---|
| AutoMem image | `ghcr.io/verygoodplugins/automem:amb-v1` (override with `AUTOMEM_IMAGE`) |
| FalkorDB | `falkordb/falkordb:v4.18.3` (pinned; graph module 41803) |
| Qdrant | `qdrant/qdrant:v1.11.3` (pinned) |
| Embeddings | FastEmbed local, `BAAI/bge-base-en-v1.5`, 768-dim (`EMBEDDING_PROVIDER=local`, no API key) |
| Answer LLM | `gemini-3.1-pro-preview` (matches the board's runs) |
| Judge LLM | `gemini-2.5-flash-lite` (matches the board's runs) |
| Mode | `rag` |

## How the provider uses AutoMem

- **Ingest** → `POST /memory`, one memory per document. Documents longer than AutoMem's
  2000-char limit are chunked at `AUTOMEM_MAX_CHARS` (default 1800) on sentence/paragraph
  boundaries; timestamps are backdated to the source. After ingest the provider waits for
  AutoMem's enrichment queue to settle so the graph it queries is fully built.
- **Retrieve** → `GET /recall`, scoped to the run's tags, with graph relation expansion on
  (`expand_relations` + `expand_respect_tags`). Content is read from
  `result["memory"]["content"]`.

## Reported metrics & latency methodology

Each run writes per-query `retrieve_time_ms`, `context_tokens`, and `correct`/`score`
to `outputs/{dataset}/{name}/rag/{split}.json`. When summarizing:

- **Accuracy** — over ALL queries (deterministic given inputs; unaffected by host load).
- **Recall latency — P50 (median), not mean.** Median because (a) P50 is the peer-standard
  reporting unit, and (b) recall latency is wall-clock around `memory.retrieve()` on local
  hardware, so it reflects host load — the median keeps a minority of load-inflated samples
  from dominating. Latency is **environment-relative** (Apple M-series, FastEmbed
  `bge-base-en-v1.5` 768-dim in-process, single-query/RAG mode) and **not comparable across
  hardware/deployments**; it is not a cross-system axis on the board.
- **Context tokens** — median retrieved-context size fed to the answerer.

## Tuning knobs (env)

| Env | Default | Purpose |
|---|---|---|
| `AUTOMEM_IMAGE` | `ghcr.io/verygoodplugins/automem:amb-v1` | AutoMem image tag |
| `AUTOMEM_HOST` | `localhost` | host the harness reaches the stack on (the repro compose sets `host.docker.internal`) |
| `AUTOMEM_API_PORT` / `AUTOMEM_FALKOR_PORT` / `AUTOMEM_QDRANT_PORT` | free port | published stack ports (the repro compose pins them) |
| `AUTOMEM_MAX_CHARS` | `1800` | chunk size (under AutoMem's 2000 hard limit) |
| `AUTOMEM_RECALL_K` | (harness `k`) | override retrieval depth |
| `AUTOMEM_ENRICH_SETTLE_SECONDS` | `120` | max wait for enrichment to drain after ingest |

## Tests

```bash
make repro-test
# or on the host toolchain:
uv run --with pytest pytest tests/test_automem_provider.py tests/test_gemini_retry.py
```
(The provider's HTTP contract and helpers are unit-tested without Docker.)
