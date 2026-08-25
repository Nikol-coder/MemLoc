# -*- coding: utf-8 -*-
"""MMR: Memory Retrieval with Multi-granularity Routing (§3.3 of the paper).

Given the memories built by ``Memory_Construct.py``, this script answers
"where to look" for every question:

**Inner-Memory Routing (§3.3.2).**  For each of the six granularities g,
the query similarity ``s_i^g = q · e_i^g`` is turned into a distribution
``p^g = softmax_i(s^g / τ)`` whose Shannon entropy ``H^g`` measures how
discriminative granularity g is.  Routing weights follow Eq. (7):

    w^g = softmax_g( 1 − H^g / log N_g )

(low entropy = peaked = discriminative ⇒ large weight).  The unit-level
score of session i is ``r_i = Σ_g w^g · s_i^g`` (Eq. 8).

**Cross-Memory Propagation (§3.3.3–3.3.5).**  The unit-level scores are
sparsified to the top-K_0 entries and used as the reset vector of
Personalized PageRank over the memory graph; session scores aggregate the
six propagated node scores, ``S_i = Σ_g π_i^g`` (Eq. 12), and the top-K
sessions form the coarse recall set.

Single-granularity baselines (Table 8 ablations) are available through
``--granularity``.

Example:
    python Dynamic_Router.py \
        --dataset longmemeval_s \
        --embedding_path outputs/memory_embeddings_longmemeval_s.pt \
        --graph_path outputs/memory_graphs_longmemeval_s.pt \
        --output_path outputs/retrieval_longmemeval_s.jsonl

    # single-granularity ablation
    python Dynamic_Router.py --dataset longmemeval_s \
        --embedding_path outputs/memory_embeddings_longmemeval_s.pt \
        --granularity event --output_path outputs/retrieval_event.jsonl
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils import (
    GRANULARITIES,
    GRANULARITY_EMB_FIELDS,
    NUM_NODES_PER_SESSION,
    load_dataset,
    load_event_memory,
    load_session_time_memory,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =====================================================================
# Query / auxiliary-text encoder (BGE-M3)
# =====================================================================
class TextEncoder:
    """Lazy BGE-M3 encoder, only loaded when embeddings must be computed
    at retrieval time (legacy embedding files without event/time vectors)."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path:
                raise RuntimeError(
                    "BGE-M3 encoder required but --encoder_path not set"
                )
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_path, use_fp16=True, devices=self.device, pool_num=0
            )
        return self._model

    def encode(self, text: str) -> torch.Tensor:
        dense = self.model.encode([text])["dense_vecs"]
        return torch.tensor(np.asarray(dense)[0]).float()

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        vectors = []
        for start in range(0, len(texts), 32):
            dense = self.model.encode(texts[start : start + 32])["dense_vecs"]
            vectors.append(np.asarray(dense))
        return torch.tensor(np.concatenate(vectors, axis=0)).float()


# =====================================================================
# Inner-Memory Routing (§3.3.2)
# =====================================================================
class InnerMemoryRouter:
    """Entropy-based granularity routing (Eq. 5–8)."""

    def __init__(self, temperature: float = 0.1, norm: str = "paper"):
        self.temperature = temperature
        self.norm = norm

    def compute_routing_weights(
        self, query_embedding: torch.Tensor, memory_embeddings
    ) -> torch.Tensor:
        query_embedding = query_embedding.float().squeeze()
        entropies = []

        for embedding in memory_embeddings:
            embedding = embedding.float()
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            similarity = (query_embedding @ embedding.T).squeeze()
            probability = F.softmax(similarity / self.temperature, dim=0)
            entropy = -torch.sum(probability * torch.log(probability + 1e-12))
            entropies.append(entropy)

        entropies = torch.stack(entropies)

        if self.norm == "legacy":
            # original released implementation: w = (1 - H) / sum(1 - H)
            weights = 1.0 - entropies
            weights = weights / weights.sum()
        else:
            # Eq. (7): w^g = softmax_g(1 - H^g / log N_g)
            num_units = torch.tensor(
                [float(emb.shape[0]) for emb in memory_embeddings]
            )
            normalized_entropy = entropies / torch.log(num_units).clamp(min=1e-12)
            weights = F.softmax(1.0 - normalized_entropy, dim=0)

        return weights


# =====================================================================
# Personalized PageRank (§3.3.5)
# =====================================================================
def paper_restart_to_igraph_damping(restart: float) -> float:
    """Map paper restart probability ``d`` (Eq. 10) to igraph ``damping``.

    Paper: ``π = (1 − d) A π + d · r``, where ``d`` is the restart /
    teleport probability.  igraph's ``personalized_pagerank(damping=…)``
    treats ``damping`` as the probability of *following* a link, so
    teleport = ``1 − damping``.  Thus ``damping = 1 − d``.
    """
    return float(np.clip(1.0 - restart, 0.0, 1.0))


def run_ppr(graph, reset_prob, damping: float):
    """Run PPR with an *igraph* damping factor (follow-link probability)."""
    reset_prob = np.asarray(reset_prob, dtype=float)
    reset_prob = np.where(
        np.isnan(reset_prob) | (reset_prob < 0), 0.0, reset_prob
    )
    return graph.personalized_pagerank(
        damping=damping,
        directed=False,
        reset=reset_prob,
        implementation="prpack",
    )


# =====================================================================
# Retriever
# =====================================================================
class MemoryRetriever:
    def __init__(self, args):
        self.args = args

        self.encoder = TextEncoder(args.encoder_path, args.device)
        self.router = InnerMemoryRouter(args.routing_temperature, args.routing_norm)
        self._igraph_damping = self._resolve_igraph_damping(args)

        self.memory_graphs = self._load_graphs(args.graph_path)

        # needed only for legacy embedding files without event/time vectors
        self.samples = None
        self.event_memory = {}
        self.time_memory = {}
        if args.data_path:
            self.samples = load_dataset(args.data_path, args.dataset)
            if args.event_path:
                self.event_memory, _ = load_event_memory(args.event_path)
            self.time_memory = load_session_time_memory(self.samples)

        self.question_text = {
            sample["question_id"]: f"Question:{sample['question']}"
            for sample in (self.samples or [])
        }

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------
    @staticmethod
    def _resolve_igraph_damping(args) -> float:
        """Prefer paper restart ``d`` when set; else use raw igraph damping.

        Paper Implementation Details: restart probability ``d = 0.1``.
        The original experimental scripts passed ``damping=0.1`` directly
        to igraph (i.e. restart ≈ 0.9); keep that via ``--ppr_damping``
        for exact reproduction of the released numbers.
        """
        if args.ppr_restart is not None:
            damping = paper_restart_to_igraph_damping(args.ppr_restart)
            logger.info(
                "PPR: paper restart d=%.3f → igraph damping=%.3f",
                args.ppr_restart,
                damping,
            )
            return damping
        logger.info(
            "PPR: using raw igraph damping=%.3f (restart≈%.3f)",
            args.ppr_damping,
            1.0 - args.ppr_damping,
        )
        return args.ppr_damping

    @staticmethod
    def _load_graphs(graph_path: str) -> Dict:
        if not graph_path or not os.path.exists(graph_path):
            if graph_path:
                logger.warning("Graph file %s not found", graph_path)
            return {}
        from igraph import Graph

        torch.serialization.add_safe_globals([Graph])
        graphs = torch.load(graph_path, weights_only=False)
        logger.info("Loaded %d memory graphs", len(graphs))
        return graphs

    def load_embeddings(self):
        embeddings = torch.load(self.args.embedding_path, map_location="cpu")
        grouped = defaultdict(list)
        for record in embeddings:
            question_id = record.get("question_id") or record[
                "conversation_id"
            ].split("#sessid#")[0]
            grouped[question_id].append(record)
        for records in grouped.values():
            records.sort(key=lambda r: r.get("session_index", -1))
        logger.info(
            "Loaded embeddings for %d questions from %s",
            len(grouped),
            self.args.embedding_path,
        )
        return grouped

    # -----------------------------------------------------------------
    # Per-question granularity matrices
    # -----------------------------------------------------------------
    def granularity_matrices(self, question_id: str, records: List[Dict]):
        """Return the six [num_sessions, d] matrices and the session ids."""
        session_ids = [r["conversation_id"] for r in records]

        matrices = {}
        for granularity in GRANULARITIES:
            field = GRANULARITY_EMB_FIELDS[granularity]
            vectors = []
            for record in records:
                if field in record:  # embedding files built by Memory_Construct.py
                    tensor = record[field]
                    if granularity == "turn":
                        tensor = (
                            tensor.mean(dim=0) if tensor.dim() == 2 else tensor
                        )
                    elif tensor.dim() > 1:
                        tensor = tensor.squeeze()
                    vectors.append(tensor.float())
            if len(vectors) == len(records):
                matrices[granularity] = torch.stack(vectors)
                continue

            # legacy embedding files: encode event / time on the fly
            if granularity in ("event", "time"):
                memory = (
                    self.event_memory
                    if granularity == "event"
                    else self.time_memory
                )
                texts = [
                    memory.get(key, "No %s data available" % granularity)
                    for key in session_ids
                ]
                matrices[granularity] = self.encoder.encode_batch(texts)
            else:
                raise KeyError(
                    f"Embedding file is missing the '{field}' vectors"
                )

        return matrices, session_ids

    def query_embedding(self, question_id: str, records: List[Dict]):
        stored = records[0].get("questions")
        if isinstance(stored, torch.Tensor):
            return stored.float().squeeze()
        text = self.question_text.get(question_id)
        if text is None:
            raise KeyError(
                f"Question text for '{question_id}' not found; "
                "pass --data_path so the query can be encoded."
            )
        return self.encoder.encode(text)

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------
    def retrieve_question(
        self, question_id: str, records: List[Dict]
    ) -> Optional[Dict]:
        matrices, session_ids = self.granularity_matrices(question_id, records)
        query_embedding = self.query_embedding(question_id, records)

        result = {
            "question_id": question_id,
            "conversation_id": records[0].get("group_id", question_id),
            "corpus_id": session_ids,
            "session_ids": [
                r.get("session_id", key.split("#sessid#")[-1])
                for r, key in zip(records, session_ids)
            ],
            "method": self.args.granularity or "all",
        }

        # ---- single-granularity baselines (Table 8) ----
        if self.args.granularity:
            target = matrices[self.args.granularity]
            scores = (query_embedding @ target.T).squeeze()
            rankings = torch.argsort(scores, descending=True).tolist()
            result["retrieval_rank"] = rankings[: self.args.num_seed_nodes]
            result["raw_scores"] = scores.tolist()
            return result

        # ---- Inner-Memory Routing (Eq. 5–8) ----
        memory_embeddings = [matrices[g] for g in GRANULARITIES]
        routing_weights = self.router.compute_routing_weights(
            query_embedding, memory_embeddings
        )

        # multi-granular matrix, session-major: node (i, g) -> row i*6+g
        weighted = [
            weight * matrices[g]
            for weight, g in zip(routing_weights, GRANULARITIES)
        ]
        multi_gran_emb = []
        for i in range(len(records)):
            for matrix in weighted:
                multi_gran_emb.append(matrix[i])
        multi_gran_emb = torch.stack(multi_gran_emb, dim=0)

        scores = (query_embedding @ multi_gran_emb.T).squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        # sparsify the seed vector to its top-K_0 entries (§3.3.5)
        if len(scores) > self.args.num_seed_nodes:
            topk_values, _ = torch.topk(scores, self.args.num_seed_nodes)
            scores = torch.where(
                scores >= topk_values[-1], scores, torch.zeros_like(scores)
            )

        # ---- Cross-Memory Propagation (Eq. 10–12) ----
        graph = self.memory_graphs.get(question_id)
        if graph is not None:
            ppr_scores = np.asarray(
                run_ppr(graph, scores.numpy(), self._igraph_damping)
            )
            session_scores = [
                float(
                    ppr_scores[
                        i * NUM_NODES_PER_SESSION : (i + 1) * NUM_NODES_PER_SESSION
                    ].sum()
                )
                for i in range(len(records))
            ]
            result["ppr"] = True
            result["raw_ppr_scores"] = ppr_scores.tolist()
        else:
            # no graph available: keep the Eq. 8 inner-memory scores
            logger.warning(
                "No graph for question %s; falling back to inner-memory "
                "scores without cross-memory propagation",
                question_id,
            )
            session_scores = [
                float(
                    scores[
                        i * NUM_NODES_PER_SESSION : (i + 1) * NUM_NODES_PER_SESSION
                    ].sum()
                )
                for i in range(len(records))
            ]
            result["ppr"] = False

        rankings = torch.argsort(
            torch.tensor(session_scores), descending=True
        ).tolist()

        result["retrieval_rank"] = rankings
        result["session_scores"] = session_scores
        result["routing_weights"] = {
            g: float(w) for g, w in zip(GRANULARITIES, routing_weights)
        }
        return result

    def retrieve(self):
        grouped = self.load_embeddings()
        results = []
        for question_id, records in tqdm(
            grouped.items(), desc="Retrieving", unit="question"
        ):
            result = self.retrieve_question(question_id, records)
            if result is not None:
                results.append(result)
        return results

    def save_results(self, results: List[Dict]):
        output_path = self.args.output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info("Saved %d retrieval results to %s", len(results), output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMR: memory retrieval with multi-granularity routing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="longmemeval_s | longmemeval_m | locomo | longmtbench",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        required=True,
        help="memory embeddings from Memory_Construct.py",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="",
        help="memory graphs from Memory_Construct.py (optional)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="raw dataset json; needed to encode queries / event / time "
        "texts when the embedding file is in the legacy format",
    )
    parser.add_argument(
        "--event_path",
        type=str,
        default="",
        help="event summaries from event_summary.py (legacy embeddings only)",
    )
    parser.add_argument("--encoder_path", type=str, default="checkpoints/bge-m3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output_path", type=str, default="outputs/retrieval_results.jsonl"
    )

    parser.add_argument(
        "--granularity",
        type=str,
        default="",
        choices=["", *GRANULARITIES],
        help="restrict retrieval to a single granularity (ablation); "
        "empty = full MMR with routing + propagation",
    )
    parser.add_argument(
        "--num_seed_nodes",
        type=int,
        default=15,
        help="K_0: entries kept in the sparsified PPR reset vector",
    )
    parser.add_argument(
        "--routing_temperature",
        type=float,
        default=0.1,
        help="softmax temperature tau of Eq. (5)",
    )
    parser.add_argument(
        "--routing_norm",
        type=str,
        default="legacy",
    )
    parser.add_argument(
        "--ppr_damping",
        type=float,
        default=0.1,
        help="Raw igraph personalized_pagerank damping (follow-link "
        "probability; restart = 1 - damping). ",
    )
    parser.add_argument(
        "--ppr_restart",
        type=float,
        default=None,
        help="Paper Eq. (10) restart / teleport probability d. ",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    retriever = MemoryRetriever(args)
    results = retriever.retrieve()
    retriever.save_results(results)


if __name__ == "__main__":
    main()
