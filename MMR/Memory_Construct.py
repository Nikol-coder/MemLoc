# -*- coding: utf-8 -*-
"""Multi-Granularity Memory Embedding & Graph Construction (§3.3.1 & §3.3.4).

This script builds everything MMR retrieval needs from the *textual*
memory units produced by the two LLM-extraction scripts:

    event_summary.py    -> events_{dataset}.json          (event / time)
    keyword_summary.py  -> memory_units_{dataset}.json    (summary / keywords)
    Memory_Construct.py -> embeddings + graphs            (this script)

Run the extraction scripts **before** this one.  Two stages remain:

1. **Embedding construction** (stage ``embed``)
   Encodes the six granularities of every session with BGE-M3:
   session-level text, per-turn texts, summary, keywords, event summary
   and the session date.

2. **Memory graph construction** (stage ``graph``)
   Builds the cross-memory graph G = (V, E) of §3.3.4 with one node per
   (session, granularity) pair — node ``(i, g)`` has id ``i * 6 + g`` — and
   two kinds of undirected edges:

   * **Semantic edges**: for each node, cosine similarity against all
     previously added same-granularity nodes; keep the top-``K_cand``
     candidates and retain only those in the higher-mean component of a
     two-component GMM fitted to the candidate scores.
   * **Temporal edges** (Eq. 9): connect granularity g of consecutive
     sessions, i.e. ``(m_i^g, m_{i+1}^g)`` for every g and i.

Outputs (written to ``--output_dir``):

* ``memory_units_full_{dataset}.json`` – per-session texts of all granularities
* ``memory_embeddings_{dataset}.pt`` – per-session six-granularity embeddings
                                        (format compatible with the original
                                        release: ``conversation_id`` is
                                        ``{question_id}#sessid#{session_id}``)
* ``memory_graphs_{dataset}.pt``     – {question_id: igraph.Graph}

Example:
    python keyword_summary.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --output_path outputs/memory_units_longmemeval_s.json

    python event_summary.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --output_path outputs/events_longmemeval_s.json

    python Memory_Construct.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --event_path outputs/events_longmemeval_s.json \
        --encoder_path checkpoints/bge-m3
"""

import argparse
import hashlib
import json
import logging
import os
from tqdm import tqdm

import numpy as np
import torch
from igraph import Graph
from sklearn.mixture import GaussianMixture

from data_utils import (
    GRANULARITIES,
    GRANULARITY_EMB_FIELDS,
    NUM_NODES_PER_SESSION,
    format_session_date,
    format_session_text,
    load_dataset,
    load_event_memory,
    session_key,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NO_EVENT_TEXT = "No event data available"
NO_TIME_TEXT = "No time data available"


# =====================================================================
# GMM-based semantic edge selection (§3.3.4)
# =====================================================================
def gmm_edge(sim_scores: torch.Tensor, mem_threshold: int, n_components: int):
    """Keep the GMM higher-mean component among the top-``mem_threshold``
    candidates of ``sim_scores`` (similarities of one node against all
    previously added same-granularity nodes)."""
    sim_scores = sim_scores.numpy()
    sorted_indices = np.argsort(sim_scores)[::-1]
    n_candidates = min(mem_threshold, len(sim_scores))
    top_candidate_indices = sorted_indices[:n_candidates]

    if len(top_candidate_indices) <= 1:
        return top_candidate_indices

    candidate_scores = sim_scores[top_candidate_indices].reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(candidate_scores)

    labels = gmm.predict(candidate_scores)
    means = gmm.means_.flatten()
    higher_class = int(np.argmax(means))

    return top_candidate_indices[labels == higher_class]


def build_memory_graph(
    granularity_vectors,
    mem_threshold: int = 20,
    n_components: int = 2,
    temporal_edges: bool = True,
) -> Graph:
    """Build the memory graph of one question.

    Args:
        granularity_vectors: list of ``[num_sessions, d]`` tensors, one per
            granularity in the canonical order of ``data_utils.GRANULARITIES``.
        mem_threshold: K_cand, number of GMM edge candidates per node.
        n_components: number of GMM components fitted on candidate scores.
        temporal_edges: whether to add the Eq. 9 temporal edges.

    Node ``(session i, granularity g)`` has id ``i * 6 + g`` — the layout
    that ``Dynamic_Router.py`` relies on when aggregating node scores.
    """
    num_sessions = granularity_vectors[0].shape[0]
    edges = set()

    # temporal edges (Eq. 9): consecutive sessions, same granularity
    if temporal_edges:
        for g in range(NUM_NODES_PER_SESSION):
            for i in range(num_sessions - 1):
                edges.add(
                    (
                        i * NUM_NODES_PER_SESSION + g,
                        (i + 1) * NUM_NODES_PER_SESSION + g,
                    )
                )

    # semantic edges: GMM-selected similar same-granularity neighbors
    for g in range(NUM_NODES_PER_SESSION):
        vectors = granularity_vectors[g]
        for i in range(1, num_sessions):
            sim_scores = vectors[i] @ vectors[:i].T
            for j in gmm_edge(sim_scores, mem_threshold, n_components):
                edges.add(
                    (
                        i * NUM_NODES_PER_SESSION + g,
                        int(j) * NUM_NODES_PER_SESSION + g,
                    )
                )

    return Graph(
        n=num_sessions * NUM_NODES_PER_SESSION,
        edges=sorted(edges),
        directed=False,
    )


# =====================================================================
# Memory constructor
# =====================================================================
class MemoryConstructor:
    def __init__(self, args):
        self.args = args
        self.samples = load_dataset(args.data_path, args.dataset)
        logger.info(
            "Loaded %d question samples (%s)", len(self.samples), args.dataset
        )

        # event-level memory (produced by event_summary.py), optional
        self.event_memory = {}
        if args.event_path and os.path.exists(args.event_path):
            self.event_memory, _ = load_event_memory(args.event_path)
            logger.info("Loaded %d event summaries", len(self.event_memory))

        self.encoder = self._load_encoder(args.encoder_path)
        self.units = {}

        os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # BGE-M3 encoder with a content-addressed cache (LoCoMo / Long-MT-Bench+ share identical sessions across questions)
    # -----------------------------------------------------------------
    def _load_encoder(self, path: str):
        from FlagEmbedding import BGEM3FlagModel

        return BGEM3FlagModel(
            path, use_fp16=True, devices=self.args.device, pool_num=0
        )

    def encode_texts(self, texts, cache):
        """Encode a list of texts (in batches), reusing cached vectors."""
        missing = [t for t in texts if t not in cache]
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            # BGE-M3 batches internally, but very long haystacks (e.g.
            # LongMemEval-M with ~500 sessions) need explicit chunking
            for start in range(0, len(unique_missing), self.args.encode_batch_size):
                chunk = unique_missing[start : start + self.args.encode_batch_size]
                dense = self.encoder.encode(chunk)["dense_vecs"]
                for text, vector in zip(chunk, dense):
                    cache[text] = torch.tensor(np.asarray(vector)).float()

        return torch.stack([cache[t] for t in texts], dim=0)

    def _load_units(self):
        """Summary / keyword units produced by ``keyword_summary.py``."""
        if os.path.exists(self.args.units_path):
            with open(self.args.units_path, "r", encoding="utf-8") as f:
                self.units = json.load(f)
            logger.info("Loaded %d summarized sessions", len(self.units))
        else:
            logger.warning(
                "Summary / keyword units not found at %s — the summary and "
                "keyword granularities will fall back to the raw session "
                "text.  Run keyword_summary.py first for full quality.",
                self.args.units_path,
            )

    # -----------------------------------------------------------------
    # Stage 1: six-granularity embeddings
    # -----------------------------------------------------------------
    def build_embeddings(self):
        if not self.units:
            self._load_units()

        cache = {}
        if os.path.exists(self.args.emb_cache_path):
            cache = torch.load(self.args.emb_cache_path, map_location="cpu")

        embeddings = []
        memory_units = {}

        for sample in tqdm(
            self.samples, desc="Encoding memories", unit="question"
        ):
            question_id = sample["question_id"]
            question_text = f"Question:{sample['question']}"

            for session_index, (session_id, messages, date) in enumerate(
                zip(
                    sample["session_ids"],
                    sample["sessions"],
                    sample["session_dates"],
                )
            ):
                key = session_key(question_id, session_id)
                session_text = format_session_text(messages)
                digest = hashlib.md5(session_text.encode("utf-8")).hexdigest()
                unit = self.units.get(
                    digest, {"summary": session_text, "keywords": ""}
                )

                event_text = self.event_memory.get(key, NO_EVENT_TEXT)
                time_text = format_session_date(date) or NO_TIME_TEXT

                turn_texts = [
                    f"[{msg['role'].upper()}]: {msg['content']}"
                    for msg in messages
                ] or [session_text]

                summary_text = unit["summary"] or session_text
                keywords_text = unit["keywords"] or session_text

                session_emb = self.encode_texts([session_text], cache)[0]
                turn_embs = self.encode_texts(turn_texts, cache)
                summary_emb = self.encode_texts([summary_text], cache)[0]
                keyword_emb = self.encode_texts([keywords_text], cache)[0]
                event_emb = self.encode_texts([event_text], cache)[0]
                time_emb = self.encode_texts([time_text], cache)[0]
                question_emb = self.encode_texts([question_text], cache)[0]

                embeddings.append(
                    {
                        # legacy composite key used across the original release
                        "conversation_id": key,
                        "question_id": question_id,
                        "group_id": sample["conversation_id"],
                        "session_id": session_id,
                        "session_index": session_index,
                        "sessions": session_emb,
                        "turns": turn_embs,
                        "summary": summary_emb,
                        "keywords": keyword_emb,
                        "event": event_emb,
                        "time": time_emb,
                        "questions": question_emb,
                    }
                )
                memory_units[key] = {
                    "question_id": question_id,
                    "session_id": session_id,
                    "session_text": session_text,
                    "summary": summary_text,
                    "keywords": keywords_text,
                    "event_text": event_text,
                    "time_text": time_text,
                }

        torch.save(cache, self.args.emb_cache_path)
        torch.save(embeddings, self.args.emb_path)
        with open(self.args.units_full_path, "w", encoding="utf-8") as f:
            json.dump(memory_units, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved %d memory records to %s", len(embeddings), self.args.emb_path
        )
        return embeddings

    # -----------------------------------------------------------------
    # Stage 2: memory graph construction (semantic + temporal edges)
    # -----------------------------------------------------------------
    def build_graphs(self, embeddings):
        grouped = {}
        for record in embeddings:
            grouped.setdefault(record["question_id"], []).append(record)

        graphs = {}
        if os.path.exists(self.args.graph_path):
            torch.serialization.add_safe_globals([Graph])
            graphs = torch.load(self.args.graph_path, weights_only=False)
            logger.info("Resuming: %d graphs already built", len(graphs))

        # conversations sharing the same session set share one graph
        graph_cache = {}

        for question_id, records in tqdm(
            grouped.items(), desc="Building graphs", unit="question"
        ):
            if question_id in graphs:
                continue
            records = sorted(
                records, key=lambda r: r.get("session_index", -1)
            )
            missing = [
                g for g in GRANULARITIES
                if GRANULARITY_EMB_FIELDS[g] not in records[0]
            ]
            if missing:
                raise KeyError(
                    "Graph construction needs the six granularity vectors "
                    f"produced by this script, but the embedding file is "
                    f"missing: {missing}.  Re-run the 'embed' stage."
                )
            cache_key = tuple(r["conversation_id"] for r in records)
            if cache_key in graph_cache:
                graphs[question_id] = graph_cache[cache_key]
                continue

            granularity_vectors = []
            for granularity in GRANULARITIES:
                vectors = []
                for record in records:
                    tensor = record[GRANULARITY_EMB_FIELDS[granularity]]
                    if granularity == "turn":
                        tensor = (
                            tensor.mean(dim=0) if tensor.dim() == 2 else tensor
                        )
                    elif tensor.dim() > 1:
                        tensor = tensor.squeeze()
                    vectors.append(tensor)
                granularity_vectors.append(torch.stack(vectors).float())

            graph = build_memory_graph(
                granularity_vectors,
                mem_threshold=self.args.mem_threshold,
                n_components=self.args.n_components,
                temporal_edges=not self.args.no_temporal_edges,
            )
            graphs[question_id] = graph
            graph_cache[cache_key] = graph

        torch.serialization.add_safe_globals([Graph])
        torch.save(graphs, self.args.graph_path)
        logger.info("Saved %d graphs to %s", len(graphs), self.args.graph_path)
        return graphs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-granularity memory construction"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="longmemeval_s | longmemeval_m | locomo | longmtbench",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--event_path",
        type=str,
        default="",
        help="event summaries from event_summary.py (optional)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--encoder_path",
        type=str,
        default="checkpoints/bge-m3",
        help="path to the BGE-M3 checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0"
    )
    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=32,
        help="texts per BGE-M3 encode call",
    )

    parser.add_argument(
        "--mem_threshold",
        type=int,
        default=20,
        help="K_cand: top candidates for GMM edge selection",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="number of GMM components",
    )
    parser.add_argument(
        "--no_temporal_edges",
        action="store_true",
        help="disable temporal edges (Eq. 9) for ablation",
    )

    parser.add_argument(
        "--stages",
        type=str,
        default="embed,graph",
        help="comma-separated stages to run: embed,graph",
    )

    args = parser.parse_args()

    tag = args.dataset
    # summary / keyword units are produced by keyword_summary.py
    args.units_path = os.path.join(args.output_dir, f"memory_units_{tag}.json")
    args.units_full_path = os.path.join(
        args.output_dir, f"memory_units_full_{tag}.json"
    )
    args.emb_path = os.path.join(
        args.output_dir, f"memory_embeddings_{tag}.pt"
    )
    args.graph_path = os.path.join(
        args.output_dir, f"memory_graphs_{tag}.pt"
    )
    args.emb_cache_path = os.path.join(args.output_dir, f"emb_cache_{tag}.pt")
    return args


def main():
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    constructor = MemoryConstructor(args)

    if "embed" in stages:
        embeddings = constructor.build_embeddings()
    else:
        embeddings = torch.load(args.emb_path, map_location="cpu")
    if "graph" in stages:
        constructor.build_graphs(embeddings)


if __name__ == "__main__":
    main()
