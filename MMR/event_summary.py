# -*- coding: utf-8 -*-
"""Event Summary: extract event-level and temporal memories for every session.

This script implements the event / temporal granularity of the
Multi-Granularity Memory Representation (§3.3.1 of the paper) using the
**Prompt Template for Event Extraction and Timeline Construction**
(Appendix J).

For every session of every question sample it asks the LLM to extract key
events and a standardized timeline, and stores one summary per session:

    {
        "question_id": ...,
        "events": [
            {"Time": <session date>, "sessid": <session id>,
             "number": <session index>, "Event_fine": <LLM extraction>}
        ],
        "question": ...,
        "question_date": ...,
        "answer": ...
    }

The output feeds ``Memory_Construct.py`` (event / time node embeddings).

Supported datasets: ``longmemeval_s`` / ``longmemeval_m`` / ``locomo`` /
``longmtbench`` (see ``data_utils.py``).

Example:
    python event_summary.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --output_path outputs/events_longmemeval_s.json \
        --model gpt-4o-mini
"""

import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data_utils import load_dataset, format_session_text, format_session_date
from llm_client import LLMClient
from prompts import EVENT_EXTRACTION, render

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def session_cache_key(session_date: str, session_text: str) -> str:
    """Content-based key so identical sessions shared by many questions
    (e.g. LoCoMo) are summarized only once."""
    digest = hashlib.md5(
        f"{session_date}\n{session_text}".encode("utf-8")
    ).hexdigest()
    return f"event::{digest}"


def summarize_session(client: LLMClient, session_date: str, session_text: str, cache=None):
    """Summarize one session with the paper's event extraction prompt."""
    key = session_cache_key(session_date, session_text)
    if cache is not None and key in cache:
        return cache[key]

    prompt = render(
        EVENT_EXTRACTION,
        ConversationDate=session_date or "Unknown",
        Conversation=session_text,
    )
    description = client.chat(prompt)
    if not description:
        description = "event error"

    if cache is not None:
        cache[key] = description
    return description


def load_processed_questions(output_path: str):
    """Question ids already present in the output file (resume support)."""
    if not os.path.exists(output_path):
        return set(), []
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["question_id"] for item in data}, data


def load_session_cache(cache_path: str):
    """Sidecar cache mapping (date, session text) hashes to summaries."""
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session_cache(cache_path: str, cache) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract event-level memories with the paper's prompt"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="longmemeval_s | longmemeval_m | locomo | longmtbench")
    parser.add_argument("--data_path", type=str, required=True,
                        help="path to the raw dataset json")
    parser.add_argument("--output_path", type=str, required=True,
                        help="where to store the event summaries (json)")
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

    processed_qids, output_data = load_processed_questions(args.output_path)
    if processed_qids:
        logger.info("Resuming: %d questions already processed", len(processed_qids))

    cache_path = args.output_path + ".cache.json"
    event_cache = load_session_cache(cache_path)
    logger.info("Session summary cache: %d entries", len(event_cache))

    client = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # group samples by conversation so identical sessions are summarized once
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    for sample in tqdm(samples, desc="Summarizing events", unit="question"):
        question_id = sample["question_id"]
        if question_id in processed_qids:
            continue

        tasks = [
            (
                index,
                session_id,
                format_session_text(messages),
                format_session_date(date),
            )
            for index, (session_id, messages, date) in enumerate(
                zip(sample["session_ids"], sample["sessions"], sample["session_dates"])
            )
        ]

        # sessions are independent -> summarize them concurrently
        summaries = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    summarize_session, client, session_date, session_text, event_cache
                ): index
                for index, _session_id, session_text, session_date in tasks
            }
            completed = 0
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"  sessions[{question_id}]",
                leave=False,
            ):
                summaries[futures[future]] = future.result()
                # checkpoint the sidecar cache so an interrupted run can
                # resume (single-question datasets otherwise save only
                # at the very end); tolerate concurrent cache mutation
                completed += 1
                if completed % 20 == 0:
                    try:
                        save_session_cache(cache_path, event_cache)
                    except RuntimeError:
                        pass  # retried at the next checkpoint

        events = [
            {
                "Time": session_date,
                "sessid": session_id,
                "number": index,
                "Event_fine": summaries[index],
            }
            for index, session_id, _session_text, session_date in tasks
        ]

        output_data.append(
            {
                "question_id": question_id,
                "events": events,
                "question": sample["question"],
                "question_date": sample["question_date"],
                "answer": sample["answer"],
            }
        )

        # incremental save (the original release saved after every question)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        save_session_cache(cache_path, event_cache)

    logger.info("Saved %d event summaries to %s", len(output_data), args.output_path)


if __name__ == "__main__":
    main()
