import re
from typing import Any, List, Dict, Set

# Metadata
REWARD_NAME = "doc_selection"
REWARD_TYPE = "batch"


def extract_answer_text(response: str) -> str:
    """Extract text content within <answer> tags from response."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not match:
        return ""
    text = match.group(1).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_ids(response: str) -> List[int]:
    """
    Extract numeric IDs from <id> tags in response.
    Supports formats like <id>6, 28</id> or <id>[6, 28]</id>.
    """
    id_match = re.search(r"<id>(.*?)</id>", response, re.DOTALL)
    if not id_match:
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not answer_match:
            return []
        content = answer_match.group(1)
    else:
        content = id_match.group(1)
    
    content = content.replace("[", "").replace("]", "").replace(" ", "")
    tokens = re.split(r"[,;，；、]+", content)
    ids = []
    for t in tokens:
        if t.isdigit():
            ids.append(int(t))
    return sorted(list(set(ids)))


def format_reward(response: str) -> float:
    """Check if response contains <reason>, <id>, and <answer> tags."""
    has_reason = bool(re.search(r"<reason>.*</reason>", response, re.DOTALL))
    has_id = bool(re.search(r"<id>.*</id>", response, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*</answer>", response, re.DOTALL))

    return 1.0 if has_reason and has_id and has_answer else 0.0


def id_match_reward(response: str, ground_truth: str) -> float:
    """
    Compute ID matching reward: matched ID count / total GT IDs.
    Returns 0.0 if ground_truth contains no IDs.
    """
    pred_ids = extract_ids(response)
    gt_ids = extract_ids(ground_truth)
    
    if not gt_ids:
        return 0.0
    
    matched_count = len(set(pred_ids) & set(gt_ids))
    return matched_count / len(gt_ids)


def answer_match_reward(response: str, ground_truth: str) -> float:
    """Check if predicted answer overlaps with ground truth answer (case-insensitive)."""
    pred_answer = extract_answer_text(response)
    gt_answer = extract_answer_text(ground_truth)
    
    if not pred_answer or not gt_answer:
        return 0.0
    
    return 1.0 if pred_answer.lower() in gt_answer.lower() or gt_answer.lower() in pred_answer.lower() else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    id_weight: float = 0.6,
    answer_weight: float = 0.3
) -> List[Dict[str, float]]:
    """
    Compute composite reward score.

    Args:
        reward_inputs: list of dicts, each must contain "response" and "ground_truth"
        format_weight: weight for format reward
        id_weight: weight for ID match reward
        answer_weight: weight for answer match reward

    Returns:
        List of dicts containing individual reward components and overall score
    """
    total_weight = format_weight + id_weight + answer_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight:.4f}")
    
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        
        # Compute individual reward components
        f_score = format_reward(response)
        id_score = id_match_reward(response, ground_truth)
        ans_score = answer_match_reward(response, ground_truth)
        
        # Compute overall score
        overall = (
            format_weight * f_score +
            id_weight * id_score +
            answer_weight * ans_score
        )
        
        scores.append({
            "overall": overall,
            "format": f_score,
            "id_match": id_score,
            "answer_match": ans_score
        })
    
    return scores