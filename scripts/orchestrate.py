#!/usr/bin/env python3
"""CPU-aware orchestrator for the full AutoMem AMB submission matrix.

Runs every split in the submission back-to-back on the host toolchain
(`uv run omb …`), one AutoMem stack at a time — a single stack saturates a
many-core host, so serial at full CPU is optimal (raise MAXC only on a host
that can fit two stacks). Ordered fast→slow with the multi-day BEAM-10m tail
last, so the headline splits bank first. Crash-safe: a non-zero exit is
re-queued and resumes via `--skip-ingested` (per-unit checkpoints), up to
MAX_ATTEMPTS.

For a single split with no host toolchain, prefer the Docker-only path:
`make repro DATASET=<ds> SPLIT=<split>`. This script is the full-matrix runner
for a host that already has `uv` + the repo.

Env overrides: AUTOMEM_IMAGE, OMB_ANSWER_LLM/MODEL, OMB_JUDGE_LLM/MODEL, MAXC.
Requires GEMINI_API_KEY in the environment (or the repo's .env, which omb loads).
"""
import os
import subprocess
import time
from pathlib import Path

FORK = Path(__file__).resolve().parent.parent
ENV = {**os.environ,
       "AUTOMEM_IMAGE": os.environ.get("AUTOMEM_IMAGE", "ghcr.io/verygoodplugins/automem:amb-v1"),
       "OMB_ANSWER_LLM": os.environ.get("OMB_ANSWER_LLM", "gemini"),
       "OMB_ANSWER_MODEL": os.environ.get("OMB_ANSWER_MODEL", "gemini-3.1-pro-preview"),
       "OMB_JUDGE_LLM": os.environ.get("OMB_JUDGE_LLM", "gemini"),
       "OMB_JUDGE_MODEL": os.environ.get("OMB_JUDGE_MODEL", "gemini-2.5-flash-lite")}
MAXC = int(os.environ.get("MAXC", "1"))  # one AutoMem stack saturates a many-core host

# (dataset, split, run_name, description) — SERIAL order, fast→slow, long pole LAST.
# beam-100k runs x3 as a reproducibility check; everything else is a single
# full-split run (large-n binomial CI is already tighter than run-to-run noise).
JOBS = [
    ("locomo", "locomo10", "automem-sub", "core-3 locomo"),
    ("personamem", "32k", "automem-sub", "core-3 personamem"),
    ("beam", "100k", "automem-sub-rep1", "beam-100k reproducibility rep1"),
    ("beam", "100k", "automem-sub-rep2", "beam-100k reproducibility rep2"),
    ("beam", "100k", "automem-sub-rep3", "beam-100k reproducibility rep3"),
    ("beam", "500k", "automem-sub", "beam-500k"),
    ("beam", "1m", "automem-sub", "beam-1m"),
    ("longmemeval", "s", "automem-sub", "core-3 longmemeval (slow per-question ingest)"),
    ("beam", "10m", "automem-sub", "beam-10m extreme tail (~1-2 days)"),
]


def outpath(ds, run, split):
    return FORK / "outputs" / ds / run / "rag" / f"{split}.json"


def launch(job):
    ds, split, run, desc = job
    log = f"/tmp/amb-{ds}-{split}-{run}.log"
    cmd = ["uv", "run", "omb", "run", "--memory", "automem", "--mode", "rag",
           "--dataset", ds, "--split", split, "--name", run, "--description", desc]
    if outpath(ds, run, split).exists():
        cmd.append("--skip-ingested")  # resume: re-ingest nothing, score remaining units
    lf = open(log, "a")
    lf.write(f"\n==== launch {ds}/{split} name={run} {time.strftime('%H:%M:%S')} "
             f"cmd={' '.join(cmd)} ====\n")
    lf.flush()
    p = subprocess.Popen(cmd, cwd=str(FORK), env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    return {"job": job, "p": p, "log": log, "lf": lf, "t0": time.time()}


def main():
    MAX_ATTEMPTS = 3
    attempts = {}
    queue = list(JOBS)
    running = []
    done = []
    while queue or running:
        while queue and len(running) < MAXC:
            j = queue.pop(0)
            running.append(launch(j))
            print(f"[{time.strftime('%H:%M:%S')}] START {j[0]}/{j[1]} name={j[2]} "
                  f"(running={len(running)}, queued={len(queue)})", flush=True)
            time.sleep(12)  # stagger docker spins
        still = []
        for r in running:
            rc = r["p"].poll()
            if rc is None:
                still.append(r)
                continue
            r["lf"].close()
            dt = int(time.time() - r["t0"])
            j = r["job"]
            key = (j[0], j[1], j[2])
            attempts[key] = attempts.get(key, 0) + 1
            if rc != 0 and attempts[key] < MAX_ATTEMPTS:
                # Crash backstop: re-queue and resume (launch() adds --skip-ingested
                # since partial output exists). Each attempt banks more saved units.
                print(f"[{time.strftime('%H:%M:%S')}] RETRY {j[0]}/{j[1]} name={j[2]} "
                      f"exit={rc} in {dt}s (attempt {attempts[key]}/{MAX_ATTEMPTS}) — re-queue resume", flush=True)
                queue.insert(0, j)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] DONE  {j[0]}/{j[1]} name={j[2]} "
                      f"exit={rc} in {dt}s (attempt {attempts[key]})", flush=True)
                done.append((j, rc))
        running = still
        time.sleep(15)
    print(f"[{time.strftime('%H:%M:%S')}] ALL_RUNS_COMPLETE n={len(done)}", flush=True)
    for j, rc in done:
        flag = "" if rc == 0 else "  <-- NONZERO (gave up after retries)"
        print(f"  {j[0]}/{j[1]} name={j[2]} exit={rc}{flag}", flush=True)


if __name__ == "__main__":
    main()
