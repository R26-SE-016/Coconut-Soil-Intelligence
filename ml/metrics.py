import numpy as np
from typing import List, Any, Union

def precision_at_k(actual: List[Any], predicted: List[Any], k: int) -> float:
    """
    Calculate Precision at k (P@k) for a single sample.
    
    Parameters:
    -----------
    actual : List[Any]
        The true relevant items/labels for the sample.
    predicted : List[Any]
        The ranked list of predictions/recommendations (sorted by confidence descending).
    k : int
        The number of top predictions to consider (must be >= 1).
        
    Returns:
    --------
    float
        Precision@k score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer >= 1")
        
    # Get top-k predictions
    top_k_predicted = predicted[:k]
    
    # Normalize labels to strings and lower case to handle case sensitivity
    actual_set = {str(item).strip().lower() for item in actual}
    top_k_set = {str(item).strip().lower() for item in top_k_predicted}
    
    if not actual_set:
        return 0.0
        
    # Count how many of top-k predictions are in the actual relevant set
    relevant_in_top_k = len(top_k_set.intersection(actual_set))
    
    # Return fraction of top-k predictions that are relevant
    return relevant_in_top_k / k

def recall_at_k(actual: List[Any], predicted: List[Any], k: int) -> float:
    """
    Calculate Recall at k (R@k) for a single sample.
    
    Parameters:
    -----------
    actual : List[Any]
        The true relevant items/labels for the sample.
    predicted : List[Any]
        The ranked list of predictions/recommendations (sorted by confidence descending).
    k : int
        The number of top predictions to consider (must be >= 1).
        
    Returns:
    --------
    float
        Recall@k score between 0.0 and 1.0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer >= 1")
        
    actual_set = {str(item).strip().lower() for item in actual}
    if not actual_set:
        return 0.0
        
    # Get top-k predictions
    top_k_predicted = predicted[:k]
    top_k_set = {str(item).strip().lower() for item in top_k_predicted}
    
    # Count how many of the actual relevant items are in the top-k predictions
    relevant_in_top_k = len(top_k_set.intersection(actual_set))
    
    # Return fraction of actual relevant items that were successfully retrieved
    return relevant_in_top_k / len(actual_set)

def mean_precision_at_k(actual_list: List[List[Any]], predicted_list: List[List[Any]], k: int) -> float:
    """
    Calculate the Mean Precision at k (MP@k) across a batch of samples.
    """
    if len(actual_list) != len(predicted_list):
        raise ValueError("actual_list and predicted_list must have the same length")
        
    scores = [precision_at_k(act, pred, k) for act, pred in zip(actual_list, predicted_list)]
    return float(np.mean(scores)) if scores else 0.0

def mean_recall_at_k(actual_list: List[List[Any]], predicted_list: List[List[Any]], k: int) -> float:
    """
    Calculate the Mean Recall at k (MR@k) across a batch of samples.
    """
    if len(actual_list) != len(predicted_list):
        raise ValueError("actual_list and predicted_list must have the same length")
        
    scores = [recall_at_k(act, pred, k) for act, pred in zip(actual_list, predicted_list)]
    return float(np.mean(scores)) if scores else 0.0
