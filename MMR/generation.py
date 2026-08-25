# -*- coding: utf-8 -*-
"""Cue-Guidance Generation (§3.4, Eq. 19 of the paper).

Generates the final answer for every question using the *Prompt Template
for Multi-Granular Reasoning and Answer Generation* (Appendix J).  The
context is built from the sessions retrieved by ``Dynamic_Router.py``
(and, optionally, refined by ``locator_inference.py``):

* Retrieved sessions are numbered with global IDs ``[1] .. [K]`` in
  reranked order.
* When a localization file is supplied, every retrieved session is
  decomposed into atomic segments with local IDs and annotated with its
  selected evidence IDs (``Evidences IDs: [..]``), following Fig. 2.
* Summary / keyword / event / time of each retrieved session are exposed
  as "Reference Information".

Example:
    python generation.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --retrieval_path outputs/retrieval_longmemeval_s.jsonl \
        --units_path outputs/memory_units_full_longmemeval_s.json \
        --localization_path outputs/localization_longmemeval_s.jsonl \
        --topk 3 --model gpt-4o-mini \
        --output_path outputs/generation_longmemeval_s.jsonl
"""

import argparse
import json
import logging
import os
from tqdm import tqdm

from data_utils import load_dataset, session_key, split_session_key
from llm_client import LLMClient
from prompts import build_generation_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_json(path: str, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_session_text_with_local_ids(units: dict, session_key_: str) -> str:
    """Render one session as atomic segments with local IDs ``[1] .. [L]``."""
    unit = units.get(session_key_, {})
    session_text = unit.get("session_text", "")
    if not session_text:
        return session_text

    segments = [line for line in session_text.split("\n") if line.strip()]
    return "\n".join(
        f"[{i + 1}] {segment}" for i, segment in enumerate(segments)
    )


def build_context(
    ordered_session_keys,
    units: dict,
    localization: dict,
) -> str:
    """Conversation History and Context with global session IDs."""
    blocks = []
    for global_id, key in enumerate(ordered_session_keys, start=1):
        unit = units.get(key, {})
        localization_entry = localization.get("eids", {}).get(key)

        if localization_entry is not None:
            body = build_session_text_with_local_ids(units, key)
            evidence_ids = "".join(f"[{eid}]" for eid in localization_entry)
            blocks.append(f"[{global_id}] {body}\nEvidences IDs: {evidence_ids}")
        else:
            blocks.append(f"[{global_id}] {unit.get('session_text', '')}")

    return "\n\n".join(blocks)


def build_reference(ordered_session_keys, units: dict) -> str:
    """Optional reference information (Global ID mapping)."""
    lines = []
    for global_id, key in enumerate(ordered_session_keys, start=1):
        unit = units.get(key)
        if not unit:
            continue
        lines.append(
            f"[{global_id}] summary: {unit.get('summary', '')} | "
            f"keywords: {unit.get('keywords', '')} | "
            f"event: {unit.get('event_text', '')} | "
            f"time: {unit.get('time_text', '')}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Cue-guidance answer generation"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="longmemeval_s | longmemeval_m | locomo | longmtbench",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--retrieval_path", type=str, required=True)
    parser.add_argument(
        "--units_path", type=str, default="",
        help="memory_units_full_{dataset}.json from Memory_Construct.py",
    )
    parser.add_argument(
        "--localization_path", type=str, default="",
        help="localization results from locator_inference.py (optional)",
    )
    parser.add_argument(
        "--topk", type=int, default=3,
        help="number of retrieved sessions fed to the generator",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument(
        "--output_path", type=str, default="outputs/generation_results.jsonl"
    )
    args = parser.parse_args()

    samples = {s["question_id"]: s for s in load_dataset(args.data_path, args.dataset)}
    retrieval = {
        item["question_id"]: item for item in load_jsonl(args.retrieval_path)
    }
    units = load_json(args.units_path, default={})
    localizations = {
        item["question_id"]: item
        for item in load_jsonl(args.localization_path)
    } if args.localization_path else {}

    client = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    processed = set()
    if os.path.exists(args.output_path):
        processed = {item["question_id"] for item in load_jsonl(args.output_path)}
        logger.info("Resuming: %d questions already generated", len(processed))

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    results = []
    output_mode = "a" if processed else "w"
    with open(args.output_path, output_mode, encoding="utf-8") as out_file:
        for question_id, sample in tqdm(
            samples.items(), desc="Generating", unit="question"
        ):
            if question_id in processed:
                continue

            retrieval_item = retrieval.get(question_id)
            if retrieval_item is None:
                logger.warning("No retrieval result for %s, skipping", question_id)
                continue

            corpus_id = retrieval_item["corpus_id"]
            rank = retrieval_item["retrieval_rank"][: args.topk]

            # Cue-guidance generation (paper Eq. 17-19): the generator input
            # is M^K'(q) -- the locator-SELECTED subset, i.e. the prefix of
            # reranked_session_keys of length len(sids) -- NOT the raw
            # retrieval top-k reordered in place.  Bounded by --topk.
            localization = localizations.get(question_id)
            if localization:
                reranked_keys = localization.get("reranked_session_keys", [])
                n_selected = len(localization.get("sids") or [])
                ordered_keys = reranked_keys[:n_selected][: args.topk]
                if not ordered_keys:
                    ordered_keys = [corpus_id[i] for i in rank]
            else:
                ordered_keys = [corpus_id[i] for i in rank]

            context = build_context(ordered_keys, units, localization or {})
            reference = build_reference(ordered_keys, units)

            prompt = build_generation_prompt(
                context=context,
                reference=reference or "None",
                question=sample["question"],
                question_date=sample.get("question_date", ""),
            )
            response = client.chat(prompt)

            record = {
                "question_id": question_id,
                "question": sample["question"],
                "answer": sample["answer"],
                "response": response,
                "context_session_keys": ordered_keys,
            }
            results.append(record)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

    logger.info(
        "Generated %d new answers (%d total) -> %s",
        len(results),
        len(results) + len(processed),
        args.output_path,
    )


if __name__ == "__main__":
    main()
