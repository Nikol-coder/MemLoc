# -*- coding: utf-8 -*-
"""Hint injection into prompts for SHPO (Eq. 25, Algorithm 1 step 14).

After ``hint_shpo.py`` has distilled a corrective hint for every
mixed-outcome query (Algorithm 1: ``h <- T(o+, o-)``), this script
*updates the training data* for the next GRPO stage by appending each
hint to its query prompt:

    x^(t+1) = x^(t)  (+)  h^(t)          (Eq. 25)

* Mixed-outcome queries get the hint appended to their prompt.
* Homogeneous queries (no hint) keep their prompt unchanged.
* Every other field of the record is preserved, so the output can be
  fed straight back into the *same* training pipeline: point
  ``config_locator_rl.yaml``'s ``data.train_files`` at the new file and
  re-run ``bash examples/qwen3_locator.sh``.

Workflow (one SHPO stage):

    1. ``bash examples/qwen3_locator.sh``          # GRPO on current data
    2. export the rollouts of this stage -> ``rollouts.json``
    3. ``python hint_shpo.py -i rollouts.json -o rollouts_with_hints.json``
    4. ``python inject_hints.py -i rollouts_with_hints.json \\
           -o data/rl_data_stage2.json --prompt_key input``
    5. re-run step 1 with the updated ``data/rl_data_stage2.json``

Example:
    python inject_hints.py \
        --input_path rollouts_with_hints.json \
        --output_path data/rl_data_stage2.json \
        --prompt_key input
"""

import argparse
import json
import logging
import os
from collections import Counter
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fields that may hold the query prompt across the different data formats
# used by the training pipeline (verl / LlamaFactory / custom JSON).
PROMPT_CANDIDATES = ("problem", "input", "prompt", "instruction")

# Separator prepended to the hint when gluing it onto the prompt (Eq. 25).
DEFAULT_HINT_PREFIX = "\n\n[Hint] "


# ===============================
# Helpers
# ===============================
def detect_prompt_key(samples: List[Dict]) -> str:
    """Pick the field that holds the prompt from the first record."""
    if not samples:
        return "input"
    first = samples[0]
    for key in PROMPT_CANDIDATES:
        if key in first and isinstance(first.get(key), str):
            return key
    return "input"


def strategy_hint(sample: Dict) -> str:
    """Read the distilled hint from either the ``hint`` object or the
    flat ``hints`` field written by ``hint_shpo.py``."""
    hint = sample.get("hint") or {}
    text = str(hint.get("strategy_hint", "") or "").strip()
    if not text:
        text = str(sample.get("hints", "") or "").strip()
    return text


def get_prompt(record: Dict, prompt_key: str) -> str:
    """Return the prompt of a record; fall back to another known field
    when ``prompt_key`` is absent (e.g. a verl record that only has
    ``problem`` while ``--prompt_key input`` was passed)."""
    value = str(record.get(prompt_key, "") or "").strip()
    if value:
        return value, prompt_key
    for key in PROMPT_CANDIDATES:
        if key == prompt_key:
            continue
        if key in record and str(record.get(key, "") or "").strip():
            return str(record[key]).strip(), key
    return "", prompt_key


def inject_hints(
    samples: List[Dict],
    prompt_key: str,
    hint_prefix: str,
) -> List[Dict]:
    """Append each hint to its prompt (Eq. 25); keep the rest unchanged."""
    updated: List[Dict] = []
    for sample in samples:
        record = dict(sample)  # shallow copy; preserve every original field
        hint = strategy_hint(record)
        prompt, used_key = get_prompt(record, prompt_key)
        if hint and prompt:
            record[used_key] = prompt + hint_prefix + hint
        updated.append(record)
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="SHPO hint injection into prompts (Eq. 25)"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="rollouts_with_hints.json produced by hint_shpo.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="updated training data for the next GRPO stage",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="",
        help="field holding the query prompt (default: auto-detect among "
        "'problem' / 'input' / 'prompt' / 'instruction')",
    )
    parser.add_argument(
        "--hint_prefix",
        type=str,
        default=DEFAULT_HINT_PREFIX,
        help="separator prepended to the hint when appending (Eq. 25)",
    )
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    logger.info("Loaded %d samples from %s", len(samples), args.input_path)

    prompt_key = args.prompt_key or detect_prompt_key(samples)
    logger.info("Prompt field: '%s'", prompt_key)

    updated = inject_hints(samples, prompt_key, args.hint_prefix)
    stats = Counter()
    for record in updated:
        if strategy_hint(record):
            stats["injected"] += 1
        else:
            stats["unchanged"] += 1

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print("\n===== DONE =====")
    print(f"Total: {len(updated)}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Updated data written to {args.output_path}")
    if stats["injected"]:
        print(
            "Point your training config (data.train_files) at the updated "
            "file and re-run the training code to start the next SHPO stage."
        )


if __name__ == "__main__":
    main()
