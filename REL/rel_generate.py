# -*- coding: utf-8 -*-
"""MemLoc deployment: REL evidence localization + cue-guidance generation.

This script unifies the four per-dataset deployment scripts used for the
paper's experiments (``0108_Memory_new/4o_0108_v26_*.py``) into a single
dataset-agnostic implementation.  It uses the *deployed* SHPO locator
(served via vLLM behind an OpenAI-compatible endpoint) to localize
evidence and then generates the final answer with cue guidance:

**Stage 1 — Inner-Memory Extraction (Eq. 13-14).**  For every session in
the top-K retrieval results, the session dialogue plus its event summary
is decomposed into sentences; the locator (``<reason>/<id>/<answer>``
structured output, Prompt Template: Evidence Filtering and Sentence
Selection) selects the query-relevant sentences (local IDs).

**Stage 2 — Cross-Memory Reranking (Eq. 15-17).**  The selected blocks of
all K sessions are renumbered; the locator selects the minimal evidence
chain (``sids``), which reranks the original retrieval indices
(selected first, unselected after).

**Stage 3 — Cue-Guidance Generation (Eq. 18-19).**  The reranked top-3
sessions are laid out with global IDs ``[1]..[N]``; the global IDs of the
stage-1 selected sentences form the reference mapping (``[51,138,...]``)
and the generator (Prompt Template: Multi-Granular Reasoning and Answer
Generation) produces the final answer.

Endpoints are configured via the environment (never hard-coded):

* Locator:     ``LOCATOR_BASE_URL`` (default ``http://localhost:8000/v1``),
               ``LOCATOR_API_KEY`` (default ``EMPTY``),
               ``LOCATOR_MODEL`` (default ``judge``)
* Generator:   ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` + ``--generator_model``

Supported datasets: ``longmemeval_s`` / ``longmemeval_m`` / ``locomo`` /
``longmtbench`` (see ``../MMR/data_utils.py``).  LongMemEval-M is split
into parts; pass ``{part}`` placeholders in the paths together with
``--parts 1-10`` to process all parts in one run.

Example:
    # LongMemEval-S / LoCoMo / Long-MT-Bench+
    python rel_generate.py \
        --dataset longmemeval_s \
        --data_path ../MMR/data_example/LongMemEval_S_example.json \
        --retrieval_path ../MMR/outputs/retrieval_longmemeval_s.jsonl \
        --event_path ../MMR/outputs/events_longmemeval_s.json \
        --output_dir ../MMR/outputs

    # LongMemEval-M (10 parts)
    python rel_generate.py --dataset longmemeval_m \
        --data_path "data/longmemeval_m_part{part}_all.json" \
        --retrieval_path "logs/part{part}/retrieval.jsonl" \
        --event_path "events/part{part}/events.json" \
        --parts 1-10 --output_dir outputs
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# share the data adapter and LLM client with the MMR module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MMR"))

from data_utils import load_dataset, load_event_memory, session_key  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from prompts import (  # noqa: E402
    build_evidence_filtering_prompt,
    build_generation_prompt,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Thin aliases so call sites stay readable (Appendix J templates).
build_locator_prompt = build_evidence_filtering_prompt


# =====================================================================
# ID parsing helpers
# =====================================================================
def extract_ids(response: str):
    """Numeric IDs inside <id>...</id> (skipping anything after <answer>)."""
    if not response:
        return []
    response_clean = response.split("<answer>")[0]
    id_match = re.search(r"<id>(.*?)</id>", response_clean, re.DOTALL)
    if not id_match:
        return []
    content = id_match.group(1).replace("[", "").replace("]", "").replace(" ", "")
    tokens = re.split(r"[,;，；、]+", content)
    return sorted({int(t) for t in tokens if t.isdigit()})


def extract_by_ids(sentences: List[str], id_str: str) -> List[str]:
    """Sentences whose 1-based local IDs appear in the ``<id>`` tag.

    Parses strictly via :func:`extract_ids` so digits inside the
    ``<reason>`` reasoning text never leak into the selection.
    """
    if not id_str.strip():
        return []
    ids = set()
    for n in extract_ids(id_str):
        idx = n - 1
        if 0 <= idx < len(sentences):
            ids.add(idx)
    return [sentences[i] for i in sorted(ids)]


# =====================================================================
# Session text construction (same layout as the deployment scripts)
# =====================================================================
def build_session_lines(messages, session_time: str) -> List[str]:
    """Render the dialogue turns of one session as ``<time> [role]: text``."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        text = str(msg.get("content", "")).strip()
        if not text:
            continue
        lines.append(f"{session_time} [{role}]: {text}")
    return lines


def build_session_text(messages, session_time: str, event_text: str) -> str:
    """Session block fed to the locator:
    ``### Session Date: ...\\n<dialogue>\\nSession Event:\\n<event>``"""
    content = "\n".join(build_session_lines(messages, session_time))
    content = content or "[No session content]"
    return (
        f"### Session Date: {session_time}\n{content}\nSession Event:\n{event_text}"
    )


def split_text_into_sentences(text: str) -> List[str]:
    """One sentence per line; drop the structural header/footer lines."""
    results = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("### ", "Session Event:", "Session Summary:", "Session Keywords:")):
            continue
        results.append(stripped)
    return results


# =====================================================================
# Two-stage localization + generation for one question
# =====================================================================
def process_question(
    locator: LLMClient,
    generator: LLMClient,
    sample: Dict,
    retrieval_item: Dict,
    event_memory: Dict[str, str],
    args,
) -> Optional[Dict]:
    question_id = sample["question_id"]
    question = sample["question"]
    question_date = sample.get("question_date", "")
    sessions_ids = sample["session_ids"]
    sessions_times = sample["session_dates"]
    sessions = sample["sessions"]

    top_k_indices = retrieval_item.get("retrieval_rank", [])[: args.topk]
    original_topk = list(top_k_indices)
    date_prefix = f"{question_date} " if question_date else ""

    # ---- Stage 1: inner-memory extraction (per session -> block) ----
    session_blocks: List[Tuple[str, str]] = []
    stage1_logs = []
    first_stage_results = {}

    for idx in top_k_indices:
        if idx >= len(sessions_ids):
            continue
        session_id = sessions_ids[idx]
        session_time = sessions_times[idx] if idx < len(sessions_times) else ""
        messages = sessions[idx] if idx < len(sessions) else []
        if not messages:
            continue

        key = session_key(question_id, session_id)
        event_text = event_memory.get(key, "[No event]")
        session_text = build_session_text(messages, session_time, event_text)
        sentences = split_text_into_sentences(session_text)
        if not sentences:
            continue

        documents = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sentences))
        local_prompt = build_locator_prompt(
            f"{date_prefix}{question}".strip(), documents
        )
        local_ids = locator.chat(local_prompt)

        selected_local_ids = [
            n for n in extract_ids(local_ids) if 1 <= n <= len(sentences)
        ]
        selected = extract_by_ids(sentences, local_ids)
        block_text = "\n".join(selected) if selected else ""
        if block_text.strip():
            session_blocks.append((session_id, block_text))

        first_stage_results[session_id] = {
            "local_ids": local_ids,
            "selected_local_ids": selected_local_ids,
            "selected_sentences": selected,
            "all_sentences": sentences,
            "original_index": idx,
        }
        stage1_logs.append(
            {
                "session_id": session_id,
                "prompt": local_prompt,
                "output": local_ids,
                "block": block_text,
                "original_sentences_count": len(sentences),
                "selected_sentences_count": len(selected),
                "selected_local_ids": selected_local_ids,
            }
        )

    # ---- Stage 2: cross-memory reranking (blocks -> sids) ----
    if not session_blocks:
        reranked_indices = top_k_indices
        filter_log = {"stage1": stage1_logs, "stage2": {}}
    else:
        documents = "\n\n".join(
            f"[{i + 1}] {block}" for i, (_, block) in enumerate(session_blocks)
        )
        global_prompt = build_locator_prompt(
            f"{date_prefix}{question}".strip(), documents
        )
        global_ids = locator.chat(global_prompt)

        selected_block_indices = {
            n - 1 for n in extract_ids(global_ids) if 1 <= n <= len(session_blocks)
        }

        selected_original = [
            top_k_indices[i] for i in sorted(selected_block_indices)
        ]
        unselected_original = [
            idx
            for i, idx in enumerate(top_k_indices)
            if i not in selected_block_indices
        ]
        reranked_indices = selected_original + unselected_original

        filter_log = {
            "stage1": stage1_logs,
            "stage2": {
                "prompt": global_prompt,
                "output": global_ids,
                "selected_block_count": len(selected_block_indices),
            },
        }

    # ---- Stage 3: global-ID layout of reranked top-3 + generation ----
    top3_reranked_indices = reranked_indices[:3]

    session_sentences_map = {}
    for original_idx in top3_reranked_indices:
        if original_idx >= len(sessions_ids):
            continue
        session_id = sessions_ids[original_idx]
        if session_id in first_stage_results:
            session_sentences_map[session_id] = first_stage_results[session_id][
                "all_sentences"
            ]

    all_sentences_for_answer = []
    global_id_to_original = {}
    current_global_id = 1
    for session_id, sentences in session_sentences_map.items():
        for sentence_idx, sentence in enumerate(sentences):
            all_sentences_for_answer.append(f"[{current_global_id}] {sentence}")
            global_id_to_original[current_global_id] = {
                "session_id": session_id,
                "original_local_id": sentence_idx + 1,
            }
            current_global_id += 1

    selected_global_ids = []
    for global_id, info in global_id_to_original.items():
        session_id = info["session_id"]
        original_local_id = info["original_local_id"]
        if session_id in first_stage_results:
            if original_local_id in first_stage_results[session_id]["selected_local_ids"]:
                selected_global_ids.append(global_id)
    selected_global_ids.sort()

    global_id_mapping_str = (
        "[" + ",".join(map(str, selected_global_ids)) + "]" if selected_global_ids else "[]"
    )
    final_context_for_answer = (
        "\n".join(all_sentences_for_answer) if all_sentences_for_answer else "[No relevant context]"
    )

    prompt = build_generation_prompt(
        context=final_context_for_answer,
        reference=global_id_mapping_str,
        question=question,
        question_date=question_date,
    )
    final_answer = generator.chat(prompt)

    return {
        "conversation_id": sample["conversation_id"],
        "question_id": question_id,
        "question": question,
        "question_date": question_date,
        "response": final_answer,
        "answer": sample["answer"],
        "answer_prompt": prompt,
        "filter_log": filter_log,
        "original_retrieval_rank": original_topk,
        "reranked_retrieval_rank": reranked_indices,
        "top3_reranked_indices": top3_reranked_indices,
        "global_id_to_original": global_id_to_original,
        "first_stage_selected_global_ids": selected_global_ids,
        "final_context_for_answer": final_context_for_answer,
    }


# =====================================================================
# I/O helpers
# =====================================================================
def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_or_create_output(output_path: str):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# =====================================================================
# Main
# =====================================================================
def run(args, locator: LLMClient, generator: LLMClient, tag: str):
    data_path = args.data_path.format(part=args.current_part)
    retrieval_path = args.retrieval_path.format(part=args.current_part)
    event_path = args.event_path.format(part=args.current_part)

    samples = load_dataset(data_path, args.dataset)
    logger.info(
        "Loaded %d question samples from %s (%s)",
        len(samples),
        data_path,
        args.dataset,
    )

    retrieval = {}
    if os.path.exists(retrieval_path):
        retrieval = {item["question_id"]: item for item in load_jsonl(retrieval_path)}
        logger.info("Loaded %d retrieval results from %s", len(retrieval), retrieval_path)
    else:
        logger.warning("Retrieval file not found: %s", retrieval_path)

    event_memory = {}
    if os.path.exists(event_path):
        event_memory, _ = load_event_memory(event_path)
        logger.info("Loaded %d event summaries from %s", len(event_memory), event_path)

    output_path = os.path.join(
        args.output_dir, f"{tag}_topk{args.topk}.json"
    )
    if args.current_part is not None:
        output_path = os.path.join(
            args.output_dir, f"{tag}_part{args.current_part}_topk{args.topk}.json"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_or_create_output(output_path)
    processed = {r.get("question_id") for r in results}
    if processed:
        logger.info("Resuming: %d questions already processed", len(processed))

    for sample in tqdm(samples, desc=f"Localize+Generate ({tag})", unit="question"):
        question_id = sample["question_id"]
        if question_id in processed:
            continue

        retrieval_item = retrieval.get(question_id)
        if retrieval_item is None:
            logger.warning("No retrieval result for %s, skipping", question_id)
            continue

        record = process_question(
            locator, generator, sample, retrieval_item, event_memory, args
        )
        if record is None:
            continue

        results.append(record)
        # incremental save so a crash does not lose completed questions
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # final save: guarantees the output file exists even when every
    # question was already processed (or the input was empty)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d results to %s", len(results), output_path)
    return output_path


def build_tag(args) -> str:
    return (
        f"rel_generate_{args.dataset}-{args.retriever}-{args.method}"
        f"_filter-{args.generator_model}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="MemLoc deployment: REL localization + cue-guidance generation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="longmemeval_s | longmemeval_m | locomo | longmtbench",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="raw dataset json; may contain {part} for LongMemEval-M",
    )
    parser.add_argument(
        "--retrieval_path",
        type=str,
        required=True,
        help="retrieval output of MMR/Dynamic_Router.py; may contain {part}",
    )
    parser.add_argument(
        "--event_path",
        type=str,
        default="",
        help="event summaries from MMR/event_summary.py; may contain {part}",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--parts",
        type=str,
        default="",
        help="parts to process when paths contain {part}, e.g. '1-10' or '1,3,5'",
    )
    parser.add_argument("--topk", type=int, default=10,
                        help="K: number of retrieved sessions to localize (paper: 10)")
    parser.add_argument("--retriever", type=str, default="bgem3")
    parser.add_argument("--method", type=str, default="memloc")
    parser.add_argument(
        "--generator_model",
        type=str,
        default="gpt-4o-mini",
        help="generator model (paper: GPT-4o mini)",
    )
    parser.add_argument(
        "--generator_temperature", type=float, default=0.0
    )
    parser.add_argument(
        "--generator_max_tokens", type=int, default=4000
    )
    args = parser.parse_args()

    # ---- clients ----
    # locator: deployed SHPO model behind an OpenAI-compatible endpoint
    from openai import OpenAI as _OpenAI

    locator = LLMClient(
        model=os.environ.get("LOCATOR_MODEL", "judge"),
        client=_OpenAI(
            base_url=os.environ.get("LOCATOR_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LOCATOR_API_KEY", "EMPTY"),
        ),
        temperature=0.0,
        max_tokens=int(os.environ.get("LOCATOR_MAX_TOKENS", "25000")),
        timeout=120,
    )
    # generator: any OpenAI-compatible endpoint (Azure / OpenAI / vLLM)
    generator = LLMClient(
        model=args.generator_model,
        temperature=args.generator_temperature,
        max_tokens=args.generator_max_tokens,
    )

    tag = build_tag(args)

    if "{part}" in args.data_path or "{part}" in args.retrieval_path:
        if not args.parts:
            raise ValueError(
                "Paths contain {part}; pass --parts, e.g. --parts 1-10"
            )
        if "-" in args.parts:
            start, end = map(int, args.parts.split("-"))
            part_list = list(range(start, end + 1))
        elif "," in args.parts:
            part_list = [int(x) for x in args.parts.split(",")]
        else:
            part_list = [int(args.parts)]

        for part_id in part_list:
            args.current_part = part_id
            try:
                run(args, locator, generator, tag)
            except FileNotFoundError as exc:
                logger.warning("Skipping part %d: %s", part_id, exc)
                continue
    else:
        args.current_part = None
        run(args, locator, generator, tag)


if __name__ == "__main__":
    main()
