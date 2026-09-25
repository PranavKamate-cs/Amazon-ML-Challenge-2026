"""
Evaluation metrics for Amazon ML Challenge 2026: Business Entity Resolution.

Metric: Macro-averaged F_0.5 Score across all Source 1 entities.
Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    - Precision = |True_Matches ∩ Pred_Matches| / |Pred_Matches|
    - Recall = |True_Matches ∩ Pred_Matches| / |True_Matches|
    - Singleton Logic:
        If True_Matches is empty:
            Pred is empty -> Score = 1.0
            Pred is not empty -> Score = 0.0
        If True_Matches is not empty:
            Pred is empty -> Score = 0.0
"""

from typing import Dict, List, Set, Union
import numpy as np


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    """
    is_true_empty = len(true_matches) == 0
    is_pred_empty = len(pred_matches) == 0

    # Singleton cases
    if is_true_empty and is_pred_empty:
        return 1.0
    if is_true_empty and not is_pred_empty:
        return 0.0
    if not is_true_empty and is_pred_empty:
        return 0.0

    # Non-singleton calculation
    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    denominator = (0.25 * precision) + recall
    if denominator == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / denominator
    return f05


def compute_macro_f05(
    ground_truth: Dict[str, Union[List[str], Set[str]]],
    predictions: Dict[str, Union[List[str], Set[str]]]
) -> Dict[str, float]:
    """
    Computes Macro-Averaged F_0.5 score across all Source 1 entities in ground truth.
    
    Args:
        ground_truth: Mapping of source1_entity_id -> set/list of true matching entity_ids.
        predictions: Mapping of source1_entity_id -> set/list of predicted matching entity_ids.
        
    Returns:
        Dictionary with macro_f05, precision_macro, recall_macro, and singleton_accuracy.
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_set in ground_truth.items():
        true_s = set(true_set) if not isinstance(true_set, set) else true_set
        pred_s = set(predictions.get(s1_id, [])) if not isinstance(predictions.get(s1_id, []), set) else predictions.get(s1_id, [])
        
        score = compute_entity_f05(true_s, pred_s)
        scores.append(score)
        
        if len(true_s) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    return {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "total_evaluated_entities": len(scores),
        "total_singletons": len(singleton_scores),
        "total_non_singletons": len(non_singleton_scores),
    }


if __name__ == "__main__":
    # Test with example from problem statement PDF:
    # Predicts: [S2-00047, S2-00193, S3-00812]
    # Ground truth: [S2-00047, S3-00812]
    # Precision = 2/3 = 0.6667, Recall = 2/2 = 1.0 -> F_0.5 = 0.71428...
    gt = {"S1-00001": {"S2-00047", "S3-00812"}}
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    score = compute_entity_f05(gt["S1-00001"], pred["S1-00001"])
    print(f"Test Score: {score:.4f} (Expected: ~0.7143)")
    assert abs(score - 0.7142857) < 1e-4, "Test assertion failed!"
    print("Metric implementation verified successfully!")
