# 🧠 MemLoc — Long-Term Conversational Memory QA (EMNLP 2026)

> **Where to Look and What to Use: Retrieve–Localize–Generate for Long-Term Conversational Memory Question Answering**

This repository provides the code, configuration, and data examples for our
EMNLP 2026 paper. It implements a unified **Retrieve–Localize–Generate** (RLG)
pipeline for long-term conversational memory question answering.

---

## 📰 News 🔥

🎉 **August 2026** — Our paper **"Where to Look and What to Use: Retrieve–Localize–Generate for Long-Term Conversational Memory Question Answering"** has been **accepted by EMNLP 2026**!

- ✅ **Models are now open-source** on ModelScope:
  - [**MemLoc-8B**](https://modelscope.cn/models/soliton110/MemLoc-8B) 🐘 — full-capacity locator trained with SFT + SHPO
  - [**MemLoc-4B**](https://modelscope.cn/models/soliton110/MemLoc-4B) 🐋 — lightweight locator for resource-constrained environments
- ✅ **Code released**: retrieval pipeline (`MMR/`), locator training & inference (`REL/`), and one unified deployment script.
- 🔔 Stay tuned for more materials to support full reproducibility!

---

## 🚀 Overview

MemLoc tackles long-term memory QA over multi-session conversations through
three stages (Fig. 2 of the paper):

1. 🔍 **MMR — Memory Retrieval with Multi-granularity Routing** (§3.3).
   Every session is organized into **six memory granularities**
   (session / turn / summary / keyword / event / time). An entropy-based
   router (§3.3.2, Eq. 5–8) adaptively weights the most discriminative
   granularity per query, and relevance is propagated over a cross-memory
   graph — semantic edges selected by a **two-component GMM** plus temporal
   edges (Eq. 9) — via **Personalized PageRank** (§3.3.5, Eq. 10–12).

2. 🎯 **REL — Reasoning-based Evidence Locator** (§3.4). The locator (Qwen3
   family, trained with **SFT cold-start + Self-reflective Hint Policy
   Optimization (SHPO)**) performs two-level localization:
   - **Inner-memory extraction** (Eq. 13–14): decomposes each retrieved
     memory unit into atomic evidence segments and selects the
     query-relevant ones (`eids`), producing purified evidence blocks.
   - **Cross-memory reranking** (Eq. 15–17): selects the minimal evidence
     chain (`sids`) across the compact evidence pool and reranks the
     original top-K memories, yielding the final subset **M^K′(q)**.

3. ✍️ **Cue-Guidance Generation** (§3.4, Eq. 18–19). The generator consumes
   exactly **M^K′(q)** together with the evidence IDs as lightweight,
   interpretable cues — anchoring it to critical evidence and mitigating
   the lost-in-the-middle effect.

All prompt templates of the paper (Appendix J) are centralized in
[`MMR/prompts.py`](MMR/prompts.py) and shared by every pipeline / deployment
script.

---

## 🛠️ Installation

```bash
pip install torch numpy scikit-learn igraph tqdm backoff openai FlagEmbedding
# BGE-M3 checkpoint (dense retriever)
huggingface-cli download BAAI/bge-m3 --local-dir checkpoints/bge-m3
```

LLM calls (summary / keyword / event extraction, localization, generation,
judging) go through **any OpenAI-compatible endpoint**:

```bash
export OPENAI_API_KEY=...          # "EMPTY" for local vLLM servers
export OPENAI_BASE_URL=...         # e.g. http://localhost:8000/v1
```

> 💡 The extraction steps fan out over `WORKERS` (default `64`) concurrent
> threads; `ENCODE_BATCH` (default `64`) controls the BGE-M3 batch size.

---

## 📚 Datasets

Four benchmarks are supported out of the box through the unified adapter
[`MMR/data_utils.py`](MMR/data_utils.py) (`--dataset` accepts):
`longmemeval_s`, `longmemeval_m`, `locomo`, `longmtbench` (Long-MT-Bench+).
One example conversation of each dataset ships in
[`MMR/data_example/`](MMR/data_example/):

| Dataset        | Conversations | Avg. sessions | Avg. queries | Type      | Retrieval GT |
|----------------|---------------|---------------|--------------|-----------|--------------|
| LongMemEval-S  | 500           | 50.2          | 1.0          | User-AI   | ✅           |
| LongMemEval-M  | 500           | 501.9         | 1.0          | User-AI   | ✅           |
| LoCoMo         | 10            | 27.2          | 198.6        | User-User | ✅           |
| Long-MT-Bench+ | 11            | 4.9           | 26.2         | User-AI   | ❌           |

---

## 🔍 MMR: Memory Retrieval with Multi-granularity Routing

The **`MMR/`** directory implements the full retrieval pipeline. The
simplest entry point is:

```bash
cd MMR
bash run_pipeline.sh longmemeval_s data_example/LongMemEval_S_example.json checkpoints/bge-m3
```

which runs the following stages (each script can also be invoked standalone):

| Script | Stage | Prompt template (Appendix J) |
|--------|-------|------------------------------|
| `event_summary.py` | Event / temporal memory extraction | Event Extraction and Timeline Construction |
| `keyword_summary.py` | Summary / keyword memory extraction | Summarization and Keyword Extraction |
| `Memory_Construct.py` | BGE-M3 six-granularity embeddings + memory graph (GMM semantic edges + Eq. 9 temporal edges) | — |
| `Dynamic_Router.py` | Inner-memory routing (Eq. 5–8) + PPR propagation (Eq. 10–12); single-granularity ablations via `--granularity` | — |
| `evaluate_retrieval.py` | Recall@k / NDCG@k (Table 7) | — |
| `locator_inference.py` | REL two-level localization over top-K retrievals (Eq. 13–17) | Evidence Filtering and Sentence Selection |
| `generation.py` | Cue-guidance generation over the **locator-selected subset M^K′(q)** (Eq. 17–19) | Multi-Granular Reasoning and Answer Generation |
| `generation_judge.py` | LLM-as-a-judge evaluation (4o-J) | Answer Verification |
| `prompts.py` | All paper prompt templates | (all seven) |
| `data_utils.py` | Unified four-dataset adapter | — |
| `llm_client.py` | Shared OpenAI-compatible client | — |

💡 Steps 1–2 run **in parallel** and resume internally (content-addressed
session caches), so interrupted runs pick up where they left off:

```bash
# 1) event / time memories        -> outputs/events_{dataset}.json
python event_summary.py --dataset longmemeval_s \
    --data_path data_example/LongMemEval_S_example.json \
    --output_path outputs/events_longmemeval_s.json

# 2) summary / keyword memories   -> outputs/memory_units_{dataset}.json
python keyword_summary.py --dataset longmemeval_s \
    --data_path data_example/LongMemEval_S_example.json \
    --output_path outputs/memory_units_longmemeval_s.json

# 3) embeddings + memory graph    -> outputs/memory_embeddings_{dataset}.pt
python Memory_Construct.py --dataset longmemeval_s \
    --data_path data_example/LongMemEval_S_example.json \
    --event_path outputs/events_longmemeval_s.json \
    --encoder_path checkpoints/bge-m3 --output_dir outputs
```

Example: single-granularity ablation (Table 8):

```bash
python Dynamic_Router.py --dataset longmemeval_s \
    --embedding_path outputs/memory_embeddings_longmemeval_s.pt \
    --granularity event \
    --output_path outputs/retrieval_longmemeval_s_event.jsonl
```

For LongMemEval-M (ten shards), process each shard with a distinct
`output_tag`:

```bash
for i in $(seq 1 10); do
    bash run_pipeline.sh longmemeval_m data/longmemeval_m_part${i}_all.json \
        checkpoints/bge-m3 part${i}
done
```

---

## 🎯 REL: Reasoning-based Evidence Locator

The **`REL/`** directory contains the locator **training**, **inference**,
and **deployment** code.

### 🏋️ Training the Locator

The locator is trained in two stages (Fig. 2, §3.5). We provide ready-to-use
configurations for two popular training frameworks — **just drop our config
files into the corresponding framework folder and run**:

#### Stage 1 — SFT Cold-Start (LlamaFactory) 🤖

Supervised fine-tuning on multi-hop QA (HotpotQA / MuSiQue / 2WikiMultihopQA;
see `SFT_Data_example.json` for the data format).

1. Clone [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) and install it.
2. Copy our training config:
   ```bash
   cp REL/LlamaFactory-main/examples/train_full/qwen3_locator_sft.yaml <LlamaFactory>/examples/train_full/
   ```
3. Register the dataset (e.g. in `data/dataset_info.json` of LlamaFactory) and launch:
   ```bash
   cd LlamaFactory
   llamafactory-cli train examples/train_full/qwen3_locator_sft.yaml
   ```

#### Stage 2 — SHPO RL (EasyR1) ⚡

Reinforcement learning with **Self-reflective Hint Policy Optimization**
(§3.5.2) on conversational QA (TimeDialogue / HaluMem; see
`RL_Data_example.json`).

1. Clone [EasyR1](https://github.com/hiyouga/EasyR1) and install it (`pip install -e .`).
2. Copy our example folder:
   ```bash
   cp -r REL/EasyR1-main/examples <EasyR1>/examples
   ```
   This provides the RL config (`config_locator_rl.yaml`), the launch script
   (`qwen3_locator.sh`), the structured-output format
   (`format_prompt/answer.jinja`), and the reward function
   (`reward_function/answer.py`, implementing Eq. 21–23: format / ID-match /
   answer-match rewards).
3. Launch training:
   ```bash
   cd EasyR1
   bash examples/qwen3_locator.sh
   ```

#### 🔁 SHPO: Self-Reflective Hint Iteration (Algorithm 1)

SHPO lets the policy act as its own teacher: for every **mixed-outcome**
query (≥1 correct and ≥1 incorrect rollout), a corrective hint is distilled
**without leaking the ground-truth answer**, then injected into the prompt of
the next training stage (Eq. 25: `x^(t+1) = x^(t) ⊕ h^(t)`).

```bash
# 1) distill hints from mixed-outcome rollouts (self-hint by default:
#    the deployed locator itself is the teacher, resolved from
#    LOCATOR_MODEL / LOCATOR_BASE_URL / LOCATOR_API_KEY)
python hint_shpo.py --input_path rollouts.json \
    --output_path rollouts_with_hints.json
#    ... or use an external teacher:
python hint_shpo.py --input_path rollouts.json \
    --output_path rollouts_with_hints.json --model gpt-4o-mini

# 2) append each hint to its query prompt (Eq. 25)
python inject_hints.py --input_path rollouts_with_hints.json \
    --output_path data/rl_data_stage2.json --prompt_key input
```

Then point `config_locator_rl.yaml`'s `data.train_files` at the updated file
and **simply re-run Stage 2** (`bash examples/qwen3_locator.sh`) to start the
next SHPO stage. Iterate (typically 3 stages × 60 GRPO steps):
rollout → `hint_shpo.py` → `inject_hints.py` → training.

### 🧪 Inference

`MMR/locator_inference.py` runs the trained locator (served via vLLM behind
an OpenAI-compatible endpoint) over the top-K retrieval results of MMR:

```bash
python MMR/locator_inference.py --dataset longmemeval_s \
    --data_path MMR/data_example/LongMemEval_S_example.json \
    --retrieval_path MMR/outputs/retrieval_longmemeval_s.jsonl \
    --units_path MMR/outputs/memory_units_full_longmemeval_s.json \
    --model served_locator_model \
    --output_path MMR/outputs/localization_longmemeval_s.jsonl
```

It performs **inner-memory extraction** (Eq. 13–14) followed by
**cross-memory reranking** (Eq. 15–17) using the *Prompt Template for
Evidence Filtering and Sentence Selection*. The output
(`reranked_session_keys` + `eids`) feeds `MMR/generation.py` through
`--localization_path`; the generator consumes the **locator-selected
subset M^K′(q)** (Eq. 17–19).

### 🚀 Deployment (localization + answer generation, one script)

`rel_generate.py` is the unified deployment entry point used for the paper's
experiments: it runs the deployed SHPO locator for the two-stage evidence
localization, reranks the retrieved sessions, and generates the final answer
with cue guidance (Eq. 18–19). It supports all four datasets through
`--dataset` and the shared data adapter.

```bash
# locator endpoint (the deployed SHPO model, e.g. vLLM)
export LOCATOR_BASE_URL=http://localhost:8000/v1   # default
export LOCATOR_API_KEY=EMPTY
export LOCATOR_MODEL=judge                          # default
# generator endpoint (any OpenAI-compatible service)
export OPENAI_API_KEY=...      export OPENAI_BASE_URL=...

python rel_generate.py \
    --dataset longmemeval_s \
    --data_path ../MMR/data_example/LongMemEval_S_example.json \
    --retrieval_path ../MMR/outputs/retrieval_longmemeval_s.jsonl \
    --event_path ../MMR/outputs/events_longmemeval_s.json \
    --output_dir ../MMR/outputs \
    --topk 10 --generator_model gpt-4o-mini
```

LongMemEval-M is processed in parts; use `{part}` placeholders and `--parts`:

```bash
python rel_generate.py --dataset longmemeval_m \
    --data_path "data/longmemeval_m_part{part}_all.json" \
    --retrieval_path "logs/part{part}/retrieval.jsonl" \
    --event_path "logs/part{part}/events.json" \
    --parts 1-10 --output_dir outputs
```

Each output record keeps the full deployment log (`filter_log` with the
stage-1/stage-2 prompts and locator outputs, `original_retrieval_rank`,
`reranked_retrieval_rank`, `global_id_to_original`,
`first_stage_selected_global_ids`, `final_context_for_answer`). Rerunning
with the same `--output_dir` resumes from the saved questions.

---

## 📊 Training Data Examples

* **`SFT_Data_example.json`** — 12 supervised samples (4 each for 2/3/4-hop
  reasoning) in the LlamaFactory format (`instruction` / `input` /
  `retrieve_id` / `answer` / `output`), where `output` follows the
  `<reason>...</reason><id>...</id><answer>...</answer>` structure.
* **`RL_Data_example.json`** — 10 representative samples for SHPO rollouts.

---

## 🤗 Pre-trained Locator Models

| Model | Size | Link |
|-------|------|------|
| **MemLoc-8B** 🐘 | 8B | https://modelscope.cn/models/soliton110/MemLoc-8B |
| **MemLoc-4B** 🐋 | 4B | https://modelscope.cn/models/soliton110/MemLoc-4B |

Serve the model with vLLM (OpenAI-compatible) and point `LOCATOR_BASE_URL`
/ `LOCATOR_MODEL` at it to run localization and generation.

---

## 📝 Notes

- The full datasets and complete training corpora are not redistributed in
  this repository; download the benchmarks from their official sources and
  convert them with `data_utils.load_dataset`.
- We commit to releasing further materials (full data, checkpoints, logs)
  to support full reproducibility.

---

## 🙏 Acknowledgments

This work builds upon several outstanding open-source projects:
[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory),
[EasyR1](https://github.com/hiyouga/EasyR1), [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding),
[MemGAS](https://github.com/quqxui/MemGAS),
and the underlying LLM backbones (Qwen3, BGE-M3). We sincerely thank the
authors for their contributions to the community!

---

> 📬 **Contact**: For questions or collaboration, please reach out via
> GitHub Issues or email (linxinkui@iie.ac.cn).
