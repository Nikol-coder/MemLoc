# -*- coding: utf-8 -*-
"""Unified dataset adapters for the four long-term memory QA benchmarks.

Supported datasets (see Appendix B, Table 6 of the paper):

=================  ============  ============  ===========  ===============
Dataset            LongMemEval-S LongMemEval-M LoCoMo       Long-MT-Bench+
=================  ============  ============  ===========  ===============
Conversations      500           500           10           11
Avg. sessions      50.2          501.9         27.2         4.9
Avg. queries       1.0           1.0           198.6        26.2
Conversation type  User-AI       User-AI       User-User    User-AI
Session dates      yes           yes           yes          no
Retrieval GT       yes           yes           yes          no
=================  ============  ============  ===========  ===============

Every dataset is normalized into a list of *question samples*.  A question
sample is a dict with the following schema::

    {
        "question_id":      str,                  # unique id of the question
        "conversation_id":  str,                  # group key of the sessions
        "question":         str,
        "question_date":    str,                  # "" when unavailable
        "answer":           str,
        "answer_session_ids": List[str],          # [] when unavailable
        "session_ids":      List[str],            # haystack session ids
        "session_dates":    List[str],            # "" when unavailable
        "sessions":         List[List[dict]],     # each msg: {"role", "content"}
    }

For LongMemEval-S/M every question owns its own haystack, so
``conversation_id == question_id``.  For LoCoMo and Long-MT-Bench+ all
questions of one conversation share the same sessions; their question ids
are ``f"{conversation_id}_q{idx}"``.

The per-session key used across the pipeline (embeddings, events, graphs)
is ``f"{question_id}#sessid#{session_id}"`` (see :func:`session_key`).
"""

import json
import re
from typing import Dict, List

# =====================================================================
# Dataset registry
# =====================================================================
DATASETS = ("longmemeval_s", "longmemeval_m", "locomo", "longmtbench")

DATASET_ALIASES = {
    "longmemeval_s": "longmemeval_s",
    "longmemeval-s": "longmemeval_s",
    "lme_s": "longmemeval_s",
    "lme-s": "longmemeval_s",
    "longmemeval_m": "longmemeval_m",
    "longmemeval-m": "longmemeval_m",
    "lme_m": "longmemeval_m",
    "lme-m": "longmemeval_m",
    "locomo": "locomo",
    "longmtbench": "longmtbench",
    "long_mtbench": "longmtbench",
    "long-mt-bench+": "longmtbench",
    "longmtbench+": "longmtbench",
}

SESSION_KEY_SEP = "#sessid#"

# The six memory granularities of §3.3.1, in canonical order.  The node id
# of granularity g in session i on the memory graph is ``i * 6 + g``.
GRANULARITIES = ("session", "turn", "summary", "keyword", "event", "time")
NUM_NODES_PER_SESSION = len(GRANULARITIES)

# Field names of the six granularity vectors inside one embedding record
# of ``memory_embeddings_{dataset}.pt`` (kept for backward compatibility
# with the original release, which used plural keys for two of them).
GRANULARITY_EMB_FIELDS = {
    "session": "sessions",
    "turn": "turns",
    "summary": "summary",
    "keyword": "keywords",
    "event": "event",
    "time": "time",
}


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported datasets: {DATASETS}"
        )
    return DATASET_ALIASES[key]


def session_key(question_id: str, session_id: str) -> str:
    """Composite key of one session inside one question's haystack."""
    return f"{question_id}{SESSION_KEY_SEP}{session_id}"


def split_session_key(key: str):
    """Inverse of :func:`session_key`."""
    question_id, _, session_id = key.partition(SESSION_KEY_SEP)
    return question_id, session_id


# =====================================================================
# Message parsing / formatting
# =====================================================================
_LABELED_LINE_RE = re.compile(r"^\[(?P<role>.*?)\]\s*:\s*(?P<content>.*)$")

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "bot": "assistant",
    "assistant": "assistant",
    "ai": "assistant",
    "gpt": "assistant",
}


def normalize_role(role: str) -> str:
    """Map dataset-specific speaker labels to a common role name."""
    role = role.strip()
    return _ROLE_ALIASES.get(role.lower(), role)


def parse_labeled_session(block_or_text) -> List[Dict[str, str]]:
    """Parse a session whose utterances are ``[Role]: content`` lines.

    Handles both a whole session string and a list of block strings
    (LoCoMo / Long-MT-Bench+ format).  Unlabeled continuation lines are
    appended to the previous utterance so that no content is lost.
    """
    if isinstance(block_or_text, list):
        texts = [b if isinstance(b, str) else str(b) for b in block_or_text]
    else:
        texts = [str(block_or_text)]

    messages: List[Dict[str, str]] = []
    for text in texts:
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            match = _LABELED_LINE_RE.match(line)
            if match:
                messages.append(
                    {
                        "role": normalize_role(match.group("role")),
                        "content": match.group("content").strip(),
                    }
                )
            elif messages:
                # continuation of the previous utterance
                messages[-1]["content"] += " " + line
            # else: preamble without any utterance yet -> dropped
    return messages


def normalize_messages(messages) -> List[Dict[str, str]]:
    """Normalize a session that is already a list of {role, content} dicts
    (LongMemEval haystack format)."""
    normalized = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            normalized.append(
                {
                    "role": normalize_role(str(msg.get("role", "user"))),
                    "content": str(msg["content"]).strip(),
                }
            )
        elif isinstance(msg, str):
            # tolerate raw strings mixed in
            normalized.extend(parse_labeled_session(msg))
    return normalized


def format_session_text(messages: List[Dict[str, str]]) -> str:
    """Render a session as ``[ROLE]: content`` lines (one per utterance)."""
    return "\n".join(
        f"[{msg['role'].upper()}]: {msg['content']}" for msg in messages
    )


def format_session_date(date) -> str:
    if date is None:
        return ""
    return str(date).strip()


# =====================================================================
# Per-dataset loaders
# =====================================================================
def _load_longmemeval(path: str) -> List[Dict]:
    """LongMemEval-S / LongMemEval-M.

    Two on-disk layouts are accepted:

    1. Original release — one question per record with its own haystack::

           {question_id, question, question_date, answer,
            answer_session_ids, haystack_session_ids, haystack_dates,
            haystack_sessions}

    2. Reformatted (LoCoMo-style) layout — records grouped by conversation::

           {conversation_id, qa: [{question, question_date, answer,
                                   answer_session_ids}],
            sessions_ids, sessions_dates, sessions}

       In the reformatted files each record typically holds one question
       and ``conversation_id`` equals the original ``question_id``; when a
       record holds several ``qa`` entries their ids are suffixed with
       ``_q{idx}`` so every question stays unique.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for record in data:
        if "sessions_ids" in record and "qa" in record:
            # reformatted (LoCoMo-style) layout
            conversation_id = str(record["conversation_id"])
            session_ids = [str(s) for s in record.get("sessions_ids", [])]
            session_dates = [
                format_session_date(d) for d in record.get("sessions_dates", [])
            ]
            sessions = [
                parse_labeled_session(session)
                for session in record.get("sessions", [])
            ]
            qa_list = record.get("qa", [])

            for idx, qa in enumerate(qa_list):
                # answer_session_ids may contain duplicates (e.g.
                # ["answer_x", "answer_x"]), deduplicate like _load_locomo
                answer_ids = list(
                    dict.fromkeys(qa.get("answer_session_ids", []) or [])
                )
                # a single-question record keeps the original question id
                # (== conversation_id); multi-question records are suffixed
                question_id = (
                    conversation_id
                    if len(qa_list) == 1
                    else f"{conversation_id}_q{idx}"
                )
                samples.append(
                    {
                        "question_id": question_id,
                        "conversation_id": conversation_id,
                        "question": str(qa.get("question", "")),
                        "question_date": format_session_date(qa.get("question_date")),
                        "answer": str(qa.get("answer", "")),
                        "answer_session_ids": [str(a) for a in answer_ids],
                        "session_ids": session_ids,
                        "session_dates": session_dates,
                        "sessions": sessions,
                    }
                )
        else:
            # original haystack layout (one question per record)
            question_id = str(record["question_id"])
            samples.append(
                {
                    "question_id": question_id,
                    "conversation_id": question_id,  # each question owns its haystack
                    "question": str(record.get("question", "")),
                    "question_date": format_session_date(record.get("question_date")),
                    "answer": str(record.get("answer", "")),
                    "answer_session_ids": list(record.get("answer_session_ids", [])),
                    "session_ids": [
                        str(s) for s in record.get("haystack_session_ids", [])
                    ],
                    "session_dates": [
                        format_session_date(d) for d in record.get("haystack_dates", [])
                    ],
                    "sessions": [
                        normalize_messages(session)
                        for session in record.get("haystack_sessions", [])
                    ],
                }
            )
    return samples


def _load_locomo(path: str) -> List[Dict]:
    """LoCoMo: many questions per conversation, User-User dialogues."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for record in data:
        conversation_id = str(record["conversation_id"])
        session_ids = [str(s) for s in record.get("sessions_ids", [])]
        session_dates = [
            format_session_date(d) for d in record.get("sessions_dates", [])
        ]
        sessions = [
            parse_labeled_session(session) for session in record.get("sessions", [])
        ]

        for idx, qa in enumerate(record.get("qa", [])):
            question_id = f"{conversation_id}_q{idx}"
            # answer_session_ids may contain duplicates (e.g. ["session_1", "session_1"])
            answer_ids = list(dict.fromkeys(qa.get("answer_session_ids", []) or []))
            samples.append(
                {
                    "question_id": question_id,
                    "conversation_id": conversation_id,
                    "question": str(qa.get("question", "")),
                    "question_date": format_session_date(qa.get("question_date")),
                    "answer": str(qa.get("answer", "")),
                    "answer_session_ids": [str(a) for a in answer_ids],
                    "session_ids": session_ids,
                    "session_dates": session_dates,
                    "sessions": sessions,
                }
            )
    return samples


def _load_longmtbench(path: str) -> List[Dict]:
    """Long-MT-Bench+: multi-turn User-AI sessions, no dates and no
    retrieval ground truth (excluded from retrieval evaluation)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for record in data:
        conversation_id = str(record.get("conversation_id", "mtbench"))
        sessions = [
            parse_labeled_session(session) for session in record.get("sessions", [])
        ]
        session_ids = [f"session_{i + 1}" for i in range(len(sessions))]
        session_dates = [""] * len(sessions)

        for idx, question in enumerate(record.get("questions", [])):
            question_text = str(question)
            # questions are stored with a "[human]: " prefix
            question_text = re.sub(r"^\[human\]\s*:\s*", "", question_text, flags=re.I)
            question_id = f"{conversation_id}_q{idx}"
            answers = record.get("answers", [])
            answer = str(answers[idx]) if idx < len(answers) else ""
            samples.append(
                {
                    "question_id": question_id,
                    "conversation_id": conversation_id,
                    "question": question_text,
                    "question_date": "",
                    "answer": answer,
                    "answer_session_ids": [],  # no retrieval ground truth
                    "session_ids": session_ids,
                    "session_dates": session_dates,
                    "sessions": sessions,
                }
            )
    return samples


_LOADERS = {
    "longmemeval_s": _load_longmemeval,
    "longmemeval_m": _load_longmemeval,
    "locomo": _load_locomo,
    "longmtbench": _load_longmtbench,
}


def load_dataset(path: str, dataset: str) -> List[Dict]:
    """Load any of the four benchmarks into the unified question schema."""
    dataset = normalize_dataset_name(dataset)
    return _LOADERS[dataset](path)


# =====================================================================
# Event memory (output of event_summary.py)
# =====================================================================
def load_event_memory(event_path: str):
    """Load the event summaries produced by ``event_summary.py``.

    Returns:
        event_memory:  {session_key: "(Time):\\n<Event_fine>\\n"}
        question_memory: {question_id: question text}
    """
    with open(event_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    event_memory: Dict[str, str] = {}
    question_memory: Dict[str, str] = {}

    for record in data:
        question_id = str(record["question_id"])
        question_memory[question_id] = str(record.get("question", ""))
        for event in record.get("events", []):
            key = session_key(question_id, str(event["sessid"]))
            event_memory[key] = f"({event.get('Time', '')}):\n{event.get('Event_fine', '')}\n"

    return event_memory, question_memory


def load_session_time_memory(samples: List[Dict]) -> Dict[str, str]:
    """Map every session key to its date string (temporal memory)."""
    time_memory: Dict[str, str] = {}
    for sample in samples:
        question_id = sample["question_id"]
        for session_id, date in zip(sample["session_ids"], sample["session_dates"]):
            time_memory[session_key(question_id, session_id)] = date
    return time_memory


# =====================================================================
# Ground-truth helpers (retrieval evaluation)
# =====================================================================
def ground_truth_session_indices(sample: Dict) -> List[int]:
    """Indices (into ``session_ids``) of the sessions holding the answer."""
    session_ids = sample["session_ids"]
    lookup = {sid: i for i, sid in enumerate(session_ids)}
    indices = []
    for answer_id in dict.fromkeys(sample.get("answer_session_ids", [])):
        index = lookup.get(str(answer_id), -1)
        if index >= 0:
            indices.append(index)
    return indices


def has_retrieval_ground_truth(dataset: str) -> bool:
    return normalize_dataset_name(dataset) != "longmtbench"
