# MMR — Multi-Granularity Memory Representation Pipeline

This directory implements the **MMR retrieval → REL localization → generation → evaluation** pipeline for long-term memory QA (MemLoc). It unifies four benchmarks — **LongMemEval-S, LongMemEval-M, LoCoMo, Long-MT-Bench+** — into a single question-sample schema (see `data_utils.py`) and processes them end-to-end.

## Requirements / Environment

- Python 3 with `openai`, `backoff`, `torch`, `transformers`, `sentence_transformers` (or equivalent), `tqdm`.
- OpenAI-compatible LLM endpoint configured via environment variables:

  | Variable | Default | Used by |
  |---|---|---|
  | `OPENAI_API_KEY` | `EMPTY` | all LLM scripts (`llm_client.py`) |
  | `OPENAI_BASE_URL` | — | all LLM scripts |
  | `EXTRACT_MODEL` | `gpt-4o-mini` | steps 1–2 (extraction) |
  | `GENERATOR_MODEL` | `gpt-4o-mini` | step 7 (generation) |
  | `JUDGE_MODEL` | `gpt-4o` | step 8 (verification) |
  | `LOCATOR_MODEL` | `judge` | step 6 (REL localization; set empty to skip) |
  | `LOCATOR_BASE_URL` | falls back to `OPENAI_BASE_URL`, then `http://localhost:8000/v1` | step 6 locator endpoint (vLLM instance separate from GPT steps) |
  | `WORKERS` | `64` | concurrent LLM calls per extraction step |
  | `ENCODE_BATCH` | `64` | BGE-M3 texts per encode call (step 3) |

- BGE-M3 encoder weights for step 3 (default `checkpoints/bge-m3`).

## Pipeline Overview

Each step reads the raw dataset (or a previous step's output) and writes one artifact into the `outputs/` directory. Artifacts are namespaced by dataset, e.g. `outputs/events_longmemeval_s.json`.

```
                ┌─────────────── raw dataset json (unified by data_utils.py) ───────────────┐
                │                                                                            │
                ▼                                                                            ▼
        [1] event_summary.py                                                      [2] keyword_summary.py
        (event/time granularities, LLM)                                           (summary/keyword granularities, LLM)
                │  events_{ds}.json                                                        │  memory_units_{ds}.json
                │                                                                            │
                ▼                                                                            ▼
        [3] Memory_Construct.py  (BGE-M3 embeddings + memory graph with semantic/temporal edges)
                │  memory_embeddings_{ds}.pt   memory_graphs_{ds}.pt   memory_units_full_{ds}.json
                ▼
        [4] Dynamic_Router.py  (MMR: inner-memory routing + cross-memory propagation)
                │  retrieval_{ds}.jsonl
                ▼
        [5] evaluate_retrieval.py ───► retrieval metrics (skipped for longmtbench: no GT)
                │
                ▼
        [6] locator_inference.py  (optional, needs LOCATOR_MODEL)
                │  localization_{ds}.jsonl
                ▼
        [7] generation.py  (multi-granular reasoning prompt, cue-guidance)
                │  generation_{ds}.jsonl
                ▼
        [8] generation_judge.py  (LLM-as-a-judge answer verification)
                   generation_{ds}_judge.jsonl
```

## Step-by-Step Inputs / Outputs

### [1] `event_summary.py` — Event extraction & timeline construction
- **Input:** raw dataset json (`--data_path`), `--dataset`, extraction LLM (`--model`).
- **Output:** `outputs/events_{dataset}.json` — per question, the question text plus a list of events, each with `sessid`, `Time`, and `Event_fine`. Builds the **event** and **time** memory granularities (loaded later via `data_utils.load_event_memory` / `load_session_time_memory`).

### [2] `keyword_summary.py` — Summarization & keyword extraction
- **Input:** raw dataset json, `--dataset`, extraction LLM.
- **Output:** `outputs/memory_units_{dataset}.json` — a content-addressed cache keyed by `md5(session_text)`; each entry holds `{"summary": ..., "keywords": "kw1; kw2; ..."}`. Builds the **summary** and **keyword** memory granularities. Supports resume (already-digested sessions are skipped).

### [3] `Memory_Construct.py` — Multi-granularity memory construction
- **Input:** raw dataset json, `events_{dataset}.json` (step 1), `memory_units_{dataset}.json` (step 2), BGE-M3 encoder path.
- **Output:**
  - `memory_embeddings_{dataset}.pt` — embeddings for the six granularity vectors (`session`, `turn`, `summary`, `keyword`, `event`, `time`) per session.
  - `memory_graphs_{dataset}.pt` — the memory graph with semantic and temporal edges; node id of granularity `g` in session `i` is `i * 6 + g`.
  - `memory_units_full_{dataset}.json` — per-session texts of all granularities, consumed by steps 6–7.

### [4] `Dynamic_Router.py` — MMR retrieval
- **Input:** raw dataset json, `memory_embeddings_{dataset}.pt`, `memory_graphs_{dataset}.pt`.
- **Output:** `outputs/retrieval_{dataset}.jsonl` — per question, the ranked retrieval list (session-level candidates) from inner-memory routing plus cross-memory propagation.

### [5] `evaluate_retrieval.py` — Retrieval evaluation
- **Input:** raw dataset json, `retrieval_{dataset}.jsonl`.
- **Output:** retrieval metrics printed / saved for top-k hits (`--ks 3,5,10`). Long-MT-Bench+ has no retrieval ground truth and is skipped internally.

### [6] `locator_inference.py` — REL evidence localization *(optional)*
- **Input:** raw dataset json, `retrieval_{dataset}.jsonl`, `memory_units_full_{dataset}.json`, served locator model.
- **Output:** `outputs/localization_{dataset}.jsonl` — localized evidence spans within the retrieved sessions. Skipped when `LOCATOR_MODEL` is empty.

### [7] `generation.py` — Cue-guidance answer generation
- **Input:** raw dataset json, `retrieval_{dataset}.jsonl` (and `localization_{dataset}.jsonl` if step 6 ran), `memory_units_full_{dataset}.json`, generator LLM.
- **Output:** `outputs/generation_{dataset}.jsonl` — final answers produced with the **Multi-Granular Reasoning** prompt (`--topk 3`).

### [8] `generation_judge.py` — LLM-as-a-judge verification
- **Input:** `generation_{dataset}.jsonl`, judge LLM.
- **Output:** `outputs/generation_{dataset}_judge.jsonl` — judged generations with verification scores using the **Answer Verification** prompt.

## Shared Modules (imported, not run directly)

- `data_utils.py` — unified dataset loading / normalization (`load_dataset`, `session_key`, event & time memory loaders, retrieval GT helpers).
- `llm_client.py` — OpenAI-compatible chat client with exponential-backoff retry.
- `prompts.py` — the paper's prompt templates (`render()`).

## Running

The whole pipeline is orchestrated by `run_pipeline.sh`:

```bash
bash run_pipeline.sh <dataset> <data_path> [bge_m3_path] [output_tag]
# e.g.
bash run_pipeline.sh longmemeval_s data_example/LongMemEval_S_example.json
```

Steps 1–2 always run but resume internally (processed questions /
content-addressed session caches), so re-runs skip completed work; step 3
caches its artifacts (`[ ! -f ... ]` guard), and steps 4+ always re-run.

### Processing longmemeval_m in shards

The full LongMemEval-M set is split into ten files (`data/longmemeval_m_part{1..10}_all.json`).
Process each shard individually with a distinct `output_tag` so the artifacts of one shard
never overwrite another's (the loader already accepts both the original haystack format and
this reformatted per-conversation layout):

```bash
for i in $(seq 1 10); do
    bash run_pipeline.sh longmemeval_m data/longmemeval_m_part${i}_all.json \
        checkpoints/bge-m3 part${i}
done
```

Results land in `outputs/part{1..10}/events_longmemeval_m.json`, etc.
