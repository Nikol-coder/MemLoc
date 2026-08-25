# -*- coding: utf-8 -*-
"""LLM-as-a-judge answer verification.

Judges generated answers against the reference answers using the *Prompt
Template for Answer Verification* (Appendix J of the paper) and reports
the accuracy ("4o-J" in the paper's tables).

Input: a JSON/JSONL file where each record contains at least
``question``, ``answer`` (reference) and ``response`` (model output) —
i.e. the output of ``generation.py``.

Example:
    python generation_judge.py \
        --input_path outputs/generation_longmemeval_s.jsonl \
        --output_path outputs/generation_longmemeval_s_judge.jsonl \
        --model gpt-4o
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

from tqdm import tqdm

from llm_client import LLMClient
from prompts import ANSWER_VERIFICATION, render

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ===============================
# File handling utilities
# ===============================
def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON or JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        logger.warning("File %s is empty", file_path)
        return []

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]


def save_results(data: List[Dict[str, Any]], file_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Results saved to %s", file_path)


# ===============================
# Evaluation logic
# ===============================
def calculate_accuracy(results_path: str) -> float:
    """Accuracy = fraction of samples judged [[yes]] (4o-J score)."""
    judged = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            verdict = sample.get("llm_judge_single", "")
            judged.append(1 if ("[[yes]]" in verdict and "[[no]]" not in verdict) else 0)

    if not judged:
        logger.warning("No results found for accuracy calculation")
        return 0.0

    accuracy = 100.0 * sum(judged) / len(judged)
    logger.info("LLM Judge Single Accuracy (4o-J): %.2f%%", accuracy)
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="LLM-based answer judging")
    parser.add_argument("--input_path", type=str, required=True,
                        help="generation output (json/jsonl)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="judge model (paper uses GPT-4o)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--force", action="store_true",
                        help="re-run even if the output file exists")
    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        logger.info("Results file already exists: %s", args.output_path)
        calculate_accuracy(args.output_path)
        return

    data = load_data(args.input_path)
    logger.info("Loaded %d samples from %s", len(data), args.input_path)

    # resume support: skip already-judged questions
    existing = {}
    if os.path.exists(args.output_path):
        for item in load_data(args.output_path):
            existing[item.get("question_id")] = item

    client = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    eval_results = []
    for sample in tqdm(data, desc="Evaluating", unit="sample"):
        question_id = sample.get("question_id")

        if question_id is not None and question_id in existing:
            eval_results.append(existing[question_id])
            continue

        prompt = render(
            ANSWER_VERIFICATION,
            Question=sample["question"],
            Answer=sample["answer"],
            Response=sample["response"],
        )
        response = client.chat(prompt)

        result_sample = {
            k: v for k, v in sample.items() if k != "context_session_keys"
        }
        result_sample["llm_judge_single"] = response or "ERROR"
        eval_results.append(result_sample)

    save_results(eval_results, args.output_path)
    calculate_accuracy(args.output_path)


if __name__ == "__main__":
    main()
