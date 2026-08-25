# -*- coding: utf-8 -*-
"""Retrieval evaluation: Recall@k and NDCG@k (Appendix C, Table 7).

Compares the session ranking produced by ``Dynamic_Router.py`` against
the answer sessions of every question.  Long-MT-Bench+ has no retrieval
ground truth (Table 6) and is therefore not evaluated here.

Example:
    python evaluate_retrieval.py \
        --dataset longmemeval_s \
        --data_path data_example/LongMemEval_S_example.json \
        --retrieval_path outputs/retrieval_longmemeval_s.jsonl \
        --ks 3,5,10
"""

import argparse
import json
import math

from data_utils import (
    ground_truth_session_indices,
    has_retrieval_ground_truth,
    load_dataset,
)


def load_retrieval(retrieval_path: str):
    """retrieval jsonl keyed by question_id."""
    retrieval = {}
    with open(retrieval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            retrieval[item["question_id"]] = item
    return retrieval


def compute_recall_ndcg(gt_indices, retrieval_item, k: int):
    """Recall@k / NDCG@k for one question.

    Args:
        gt_indices: ground-truth session indices (into ``session_ids``).
        retrieval_item: dict with the ``retrieval_rank`` list.
    """
    gt_set = {i for i in gt_indices if i >= 0}
    if not gt_set:
        return None, None  # no usable ground truth

    retrieved = retrieval_item["retrieval_rank"][:k]

    hits = sum(1 for idx in retrieved if idx in gt_set)
    recall = hits / len(gt_set)

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, idx in enumerate(retrieved)
        if idx in gt_set
    )
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ndcg


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="longmemeval_s | longmemeval_m | locomo | longmtbench")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--retrieval_path", type=str, required=True)
    parser.add_argument("--ks", type=str, default="3,5,10",
                        help="comma-separated k values")
    args = parser.parse_args()

    if not has_retrieval_ground_truth(args.dataset):
        print(
            f"Dataset '{args.dataset}' has no retrieval ground truth "
            "(Table 6) and is excluded from retrieval evaluation."
        )
        return

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    if not ks:
        raise ValueError("--ks must contain at least one k value")

    samples = load_dataset(args.data_path, args.dataset)
    retrieval = load_retrieval(args.retrieval_path)

    per_k = {k: {"recall": [], "ndcg": []} for k in ks}
    missing_retrieval = 0
    no_ground_truth = 0

    for sample in samples:
        gt_indices = ground_truth_session_indices(sample)
        if not gt_indices:
            no_ground_truth += 1
            continue

        item = retrieval.get(sample["question_id"])
        if item is None:
            # question missing from the retrieval output counts as failure
            missing_retrieval += 1
            for k in ks:
                per_k[k]["recall"].append(0.0)
                per_k[k]["ndcg"].append(0.0)
            continue

        for k in ks:
            recall, ndcg = compute_recall_ndcg(gt_indices, item, k)
            per_k[k]["recall"].append(recall)
            per_k[k]["ndcg"].append(ndcg)

    num_evaluated = len(per_k[ks[0]]["recall"])
    if num_evaluated == 0:
        print("No questions with retrieval ground truth were evaluated.")
        return

    print(f"Dataset: {args.dataset}")
    print(f"Questions evaluated: {num_evaluated}")
    if no_ground_truth:
        print(f"Questions without ground-truth sessions: {no_ground_truth}")
    if missing_retrieval:
        print(f"Questions missing from retrieval output: {missing_retrieval}")
    print()

    header = "Metric" + "".join(f"@{k:<6}" for k in ks)
    print(header)
    print("-" * len(header))
    recall_row = "Recall "
    ndcg_row = "NDCG   "
    for k in ks:
        recall_row += f"{100 * sum(per_k[k]['recall']) / len(per_k[k]['recall']):<7.2f}"
        ndcg_row += f"{100 * sum(per_k[k]['ndcg']) / len(per_k[k]['ndcg']):<7.2f}"
    print(recall_row)
    print(ndcg_row)


if __name__ == "__main__":
    main()
