# -*- coding: utf-8 -*-
"""REL: Reasoning-based Evidence Locator — inference (§3.4 of the paper).

Given the top-K sessions retrieved by ``Dynamic_Router.py``, the trained
locator (e.g. the SFT+SHPO Qwen3-8B served by vLLM behind an
OpenAI-compatible endpoint) performs the two-level localization of §3.4
using the *Prompt Template for Evidence Filtering and Sentence Selection*
(Appendix J):

Environment:
    ``LOCATOR_BASE_URL``  locator endpoint; when unset, falls back to
                          ``OPENAI_BASE_URL``, then to
                          ``http://localhost:8000/v1`` (local vLLM).
    ``OPENAI_API_KEY``    API key (``"EMPTY"`` for local vLLM servers).

1. **Inner-Memory Extraction (Eq. 13–14)**: every retrieved memory unit
   M_i is decomposed into atomic evidence segments with local IDs
   ``[1] .. [L_i]``; the locator selects the query-relevant segments,
   yielding the evidence IDs ``eids_i`` and a purified evidence block
   ``B_i``.

2. **Cross-Memory Reranking (Eq. 15–17)**: the evidence blocks are
   renumbered into a compact cross-memory representation ``M_in``; the
   locator identifies the minimal evidence chain ``sids`` and reranks
   the top-K candidate memory units accordingly.

Output (JSONL, consumed by ``generation.py``):

    {
      "question_id": ...,
      "reranked_session_keys": [...],      # sids order first
      "sids": [global memory ids],
      "eids":  {session_key: [local ids]},
      "evidence_blocks": {session_key: "..."}
    }

Example:
    python locator_inference.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --retrieval_path outputs/retrieval_longmemeval_s.jsonl \
        --units_path outputs/memory_units_full_longmemeval_s.json \
        --model served_locator_model \
        --output_path outputs/localization_longmemeval_s.jsonl
"""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from llm_client import LLMClient, create_client
from prompts import build_evidence_filtering_prompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ===============================
# Structured output parsing
# ===============================
def extract_ids(response: str):
    """Parse the numeric IDs inside <id>...</id> (falls back to the
    <answer> tag, mirroring the RL reward function)."""
    if not response:
        return []

    id_match = re.search(r"<id>(.*?)</id>", response, re.DOTALL)
    if id_match:
        content = id_match.group(1)
    else:
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not answer_match:
            return []
        content = answer_match.group(1)

    content = content.replace("[", "").replace("]", "").replace(" ", "")
    tokens = re.split(r"[,;，；、]+", content)
    return sorted({int(t) for t in tokens if t.isdigit()})


def extract_tag(response: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
    return match.group(1).strip() if match else ""


# ===============================
# I/O helpers
# ===============================
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


# ===============================
# Two-level localization
# ===============================
def split_segments(session_text: str):
    """Atomic evidence segments of one memory unit (one utterance per
    segment), with 1-based local IDs."""
    segments = [line for line in session_text.split("\n") if line.strip()]
    return segments


def inner_memory_extraction(
    client: LLMClient,
    question: str,
    session_key_: str,
    session_text: str,
):
    """Stage 1 (Eq. 13-14): select query-relevant segments within M_i."""
    segments = split_segments(session_text)
    if not segments:
        return [], "", ""

    documents = "\n".join(
        f"[{i + 1}] {segment}" for i, segment in enumerate(segments)
    )
    prompt = build_evidence_filtering_prompt(question, documents)
    response = client.chat(prompt)

    selected = [i for i in extract_ids(response) if 1 <= i <= len(segments)]
    # paper Eq. 14: B_i = {m_i,t | t in eids_i} -- ONLY the selected evidence
    # segments.  A unit with no selected IDs yields an empty block and is
    # left out of the cross-memory pool M_in (Eq. 15) below.
    block = "\n".join(f"[{i}] {segments[i - 1]}" for i in selected)
    return selected, block, extract_tag(response, "reason")


def cross_memory_reranking(
    client: LLMClient,
    question: str,
    blocks,  # list of (session_key, block_text)
):
    """Stage 2 (Eq. 15-17): pick the minimal evidence chain across units."""
    documents = "\n".join(
        f"[{i + 1}] {block}" for i, (_, block) in enumerate(blocks)
    )
    prompt = build_evidence_filtering_prompt(question, documents)
    response = client.chat(prompt)

    selected = [
        i for i in extract_ids(response) if 1 <= i <= len(blocks)
    ]
    if not selected:  # graceful fallback: keep the retrieval order
        selected = list(range(1, len(blocks) + 1))

    return selected, extract_tag(response, "reason")


def main():
    parser = argparse.ArgumentParser(
        description="REL two-level evidence localization"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="longmemeval_s | longmemeval_m | locomo | longmtbench")
    parser.add_argument("--data_path", type=str, required=True,
                        help="raw dataset json (for the questions)")
    parser.add_argument("--retrieval_path", type=str, required=True,
                        help="retrieval output of Dynamic_Router.py")
    parser.add_argument("--units_path", type=str, default="",
                        help="memory_units_full_{dataset}.json with session texts")
    parser.add_argument("--topk", type=int, default=10,
                        help="K: number of retrieved memories to localize (paper: 10)")
    parser.add_argument("--model", type=str, required=True,
                        help="served locator model name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--workers", type=int, default=64,
        help="concurrent stage-1 LLM calls",
    )
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    from data_utils import load_dataset  # noqa: E402

    samples = {s["question_id"]: s for s in load_dataset(args.data_path, args.dataset)}
    retrieval = {item["question_id"]: item for item in load_jsonl(args.retrieval_path)}
    units = load_json(args.units_path, default={})
    if not units:
        logger.warning(
            "No memory units loaded from %s; sessions will be empty and "
            "localization degenerates.  Run Memory_Construct.py first.",
            args.units_path,
        )

    # the locator runs on its own endpoint so it can be served by a
    # separate vLLM instance from the GPT extraction/generation steps;
    # falls back to OPENAI_BASE_URL and finally to the local vLLM default
    locator_base_url = (
        os.environ.get("LOCATOR_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://localhost:8000/v1"
    )
    client = LLMClient(
        model=args.model,
        client=create_client(base_url=locator_base_url),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    logger.info("Locator endpoint: %s (model: %s)", locator_base_url, args.model)

    processed = set()
    if os.path.exists(args.output_path):
        processed = {item["question_id"] for item in load_jsonl(args.output_path)}
        logger.info("Resuming: %d questions already localized", len(processed))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    output_mode = "a" if processed else "w"
    with open(args.output_path, output_mode, encoding="utf-8") as out_file:
        for question_id, sample in tqdm(
            samples.items(), desc="Localizing", unit="question"
        ):
            if question_id in processed:
                continue

            item = retrieval.get(question_id)
            if item is None:
                logger.warning("No retrieval result for %s, skipping", question_id)
                continue

            corpus_id = item["corpus_id"]
            rank = item["retrieval_rank"][: args.topk]
            candidate_keys = [corpus_id[i] for i in rank]

            # ---- Stage 1: inner-memory extraction (units are independent
            #      -> extracted concurrently; executor.map keeps order) ----
            def _extract(key):
                session_text = units.get(key, {}).get("session_text", "")
                return inner_memory_extraction(
                    client, sample["question"], key, session_text
                )

            if args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    stage1 = list(executor.map(_extract, candidate_keys))
            else:
                stage1 = [_extract(key) for key in candidate_keys]

            eids = {}
            blocks = []
            reasons = []
            for key, (selected, block, reason) in zip(candidate_keys, stage1):
                eids[key] = selected
                reasons.append(reason)
                # paper Eq. 15: M_in integrates the evidence blocks of the
                # units that HAVE output, concatenated in original candidate
                # order; empty-evidence units stay out of the cross-memory pool
                if block:
                    blocks.append((key, block))

            # ---- Stage 2: cross-memory reranking over the compact pool ----
            if blocks:
                sids, rerank_reason = cross_memory_reranking(
                    client, sample["question"], blocks
                )
            else:
                sids, rerank_reason = [], ""

            # reranked order: locator-selected units first, then the rest
            selected_keys = [blocks[i - 1][0] for i in sids]
            remaining_keys = [
                key for key in candidate_keys if key not in set(selected_keys)
            ]
            reranked_session_keys = selected_keys + remaining_keys

            record = {
                "question_id": question_id,
                "sids": sids,
                "reranked_session_keys": reranked_session_keys,
                "eids": eids,
                "evidence_blocks": {key: block for key, block in blocks},
                "reason": rerank_reason or " ".join(filter(None, reasons)),
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

    logger.info("Localization results saved to %s", args.output_path)


if __name__ == "__main__":
    main()
