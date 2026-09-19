import numpy as np
from typing import List, Union, Dict, Any

def _extract_ids(items: List[Union[str, Dict[str, Any]]]) -> List[str]:
    """Helper to extract document IDs whether given raw IDs or document dicts."""
    extracted = []
    for item in items:
        if isinstance(item, str):
            extracted.append(item)
        elif isinstance(item, dict):
            if "id" in item:
                extracted.append(item["id"])
            elif "document" in item and isinstance(item["document"], dict) and "id" in item["document"]:
                extracted.append(item["document"]["id"])
            else:
                raise ValueError(f"Cannot extract document ID from dict: {item}")
        else:
            raise TypeError(f"Unexpected item type {type(item)} in items list")
    return extracted

def calculate_recall(retrieved: List[Union[str, Dict[str, Any]]], ground_truth: List[str], k: int = None) -> float:
    """
    Calculate Recall@k: fraction of ground truth documents present in retrieved results.
    Accepts retrieved results as either a list of string IDs or document dicts.
    """
    if not ground_truth:
        return 1.0
    
    retrieved_ids = _extract_ids(retrieved)
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
        
    retrieved_set = set(retrieved_ids)
    hits = len(retrieved_set.intersection(ground_truth))
    return hits / len(ground_truth)

def recall_at_k(retrieved_ids: List[Union[str, Dict[str, Any]]], ground_truth_ids: List[str], k: int = 5) -> float:
    return calculate_recall(retrieved_ids, ground_truth_ids, k=k)

def mrr_at_k(retrieved: List[Union[str, Dict[str, Any]]], ground_truth: List[str], k: int = 5) -> float:
    """Mean Reciprocal Rank at k."""
    if not ground_truth:
        return 1.0
    retrieved_ids = _extract_ids(retrieved)[:k]
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids: List[Union[str, Dict[str, Any]]], ground_truth_ids: List[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not ground_truth_ids:
        return 1.0
    extracted_ids = _extract_ids(retrieved_ids)[:k]
    
    dcg = 0.0
    for i, doc_id in enumerate(extracted_ids):
        if doc_id in ground_truth_ids:
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0

def extract_citiations(text: str) -> List[str]:
    import re
    return re.findall(r"\[(doc_\d+)\]", text)

def verify_citiations(citiations: List[str], ground_truth_ids: List[str]) -> bool:
    if not citiations and not ground_truth_ids:
        return True
    return set(citiations) == set(ground_truth_ids)