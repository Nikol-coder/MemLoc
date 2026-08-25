# -*- coding: utf-8 -*-
"""Keyword / Summary extraction: build the summary and keyword granularities.

This script implements the summary and keyword levels of the
Multi-Granularity Memory Representation (§3.3.1 of the paper) using the
**Prompt Template for Summarization and Keyword Extraction** (Appendix J).

It runs *before* embedding: for every unique session it asks the LLM for
a concise summary and the most relevant keywords, and stores them in a
content-addressed cache:

    {
        "<md5(session_text)>": {"summary": "...", "keywords": "kw1; kw2; ..."},
        ...
    }

The cache format matches exactly the file that ``Memory_Construct.py``
reads for its ``embed`` stage (``memory_units_{dataset}.json``), so the
two scripts are fully decoupled:

    event_summary.py      -> events_{dataset}.json          (event / time)
    keyword_summary.py    -> memory_units_{dataset}.json    (summary / keywords)
    Memory_Construct.py   -> embeddings + graphs            (embed, graph)

Supported datasets: ``longmemeval_s`` / ``longmemeval_m`` / ``locomo`` /
``longmtbench`` (see ``data_utils.py``).

Example:
    python keyword_summary.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --output_path outputs/memory_units_longmemeval_s.json \
        --model gpt-4o-mini
"""

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data_utils import format_session_text, load_dataset
from llm_client import LLMClient
from prompts import SUMMARIZE_AND_KEYWORDS, render

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_memory_json(raw: str):
    """Parse the JSON object returned by the summarization prompt.

    Expected structure (strictly, per the paper's template):

        {
            "memory":
            {
                "summary": "<A concise summary of the conversation>",
                "keywords": "<Keyword 1>; <Keyword 2>; <Keyword 3>; ..."
            }
        }

    Returns ``(summary, keywords)``; when the response is not valid JSON
    the raw text is used as the summary so no session is lost.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            memory = obj.get("memory", obj)
            summary = str(memory.get("summary", "")).strip()
            keywords = str(memory.get("keywords", "")).strip()
            if summary:
                return summary, keywords
        except json.JSONDecodeError:
            pass
    # fallback: treat the raw text as the summary
    return text, ""


def load_processed_units(output_path: str):
    """Digests already present in the output file (resume support)."""
    if not os.path.exists(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract summary / keyword memories with the paper's prompt"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="longmemeval_s | longmemeval_m | locomo | longmtbench")
    parser.add_argument("--data_path", type=str, required=True,
                        help="path to the raw dataset json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="where to store the summary / keyword cache (json)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument(
        "--workers", type=int, default=64,
        help="concurrent LLM calls",
    )
    args = parser.parse_args()

    samples = load_dataset(args.data_path, args.dataset)
    logger.info("Loaded %d question samples from %s", len(samples), args.data_path)

    units = load_processed_units(args.output_path)
    if units:
        logger.info("Resuming: %d sessions already summarized", len(units))

    # unique sessions only (content-addressed by session text); LoCoMo and
    # Long-MT-Bench+ share identical sessions across questions
    unique_sessions = {}
    for sample in samples:
        for messages in sample["sessions"]:
            text = format_session_text(messages)
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            unique_sessions[digest] = text

    pending = {
        digest: text
        for digest, text in unique_sessions.items()
        if digest not in units
    }
    logger.info(
        "%d unique sessions (%d already summarized, %d to go)",
        len(unique_sessions),
        len(unique_sessions) - len(pending),
        len(pending),
    )

    client = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    def summarize_one(text: str):
        prompt = render(SUMMARIZE_AND_KEYWORDS, Conversation=text)
        response = client.chat(prompt)
        summary, keywords = parse_memory_json(response)
        return {"summary": summary, "keywords": keywords}

    def save_units():
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(units, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # sessions are independent -> summarize them concurrently
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_digest = {
            executor.submit(summarize_one, text): digest
            for digest, text in pending.items()
        }
        for future in tqdm(
            as_completed(future_to_digest),
            total=len(future_to_digest),
            desc="Summarizing sessions",
            unit="session",
        ):
            digest = future_to_digest[future]
            try:
                units[digest] = future.result()
            except Exception as exc:  # noqa: BLE001 - keep the run alive
                logger.error("Summary failed for %s: %s", digest, exc)
                units[digest] = {"summary": "", "keywords": ""}
            done += 1
            # incremental save so a crash does not lose completed sessions
            if done % 20 == 0:
                save_units()

    save_units()

    logger.info(
        "Saved %d summary / keyword units to %s", len(units), args.output_path
    )


if __name__ == "__main__":
    main()
