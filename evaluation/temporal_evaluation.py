import re
from typing import Dict, List, Set, Union
import numpy as np

def evaluate_temporal_aspect(ground_truth: Dict, predicted: Dict) -> float:
    """Evaluate the accuracy of temporal aspect detection."""
    correct = sum(1 for q in ground_truth if ground_truth[q]['has_temporal_aspect'] == predicted[q]['has_temporal_aspect'])
    return correct / len(ground_truth)

def evaluate_recency_weight(ground_truth: Dict, predicted: Dict) -> float:
    """Evaluate the accuracy of recency weight prediction using RMSE."""
    mse_list = []
    for q in ground_truth:
        if ground_truth[q]['has_temporal_aspect'] and predicted[q]['has_temporal_aspect']:
            gt_weight = ground_truth[q]['expected_recency_weight']
            pred_weight = predicted[q]['expected_recency_weight']
            if gt_weight is not None and pred_weight is not None:
                mse = (gt_weight - pred_weight) ** 2
                mse_list.append(mse)
    
    return np.sqrt(sum(mse_list) / len(mse_list)) if mse_list else 0

def parse_year_filter(filter_str: Union[str, None]) -> Set[int]:
    """Parse a year filter string and return a set of years that satisfy the filter."""
    if filter_str is None:
        return set(range(1950, 2025))  # If no filter, assume all years are valid
    
    # Replace logical operators with Python equivalents
    filter_str = filter_str.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not')
    
    years = set()
    for year in range(1950, 2025):
        try:
            # Create a safe local environment for eval
            safe_dict = {'year': year}
            if eval(filter_str, {"__builtins__": None}, safe_dict):
                years.add(year)
        except Exception as e:
            print(f"Warning: Could not evaluate condition '{filter_str}' for year {year}: {str(e)}")
    return years

def evaluate_year_filter(ground_truth: Dict, predicted: Dict) -> float:
    """Evaluate the accuracy of year filter prediction using Intersection over Union."""
    iou_scores = []
    for q in ground_truth:
        if ground_truth[q]['has_temporal_aspect'] and predicted[q]['has_temporal_aspect']:
            gt_years = parse_year_filter(ground_truth[q]['expected_year_filter'])
            pred_years = parse_year_filter(predicted[q]['expected_year_filter'])
            
            intersection = len(gt_years.intersection(pred_years))
            union = len(gt_years.union(pred_years))
            iou = intersection / union if union > 0 else 0
            iou_scores.append(iou)
    
    return sum(iou_scores) / len(iou_scores) if iou_scores else 0

def evaluate_temporal_queries(ground_truth: Dict, predicted: Dict) -> Dict:
    """Evaluate all aspects of temporal query prediction."""
    return {
        'temporal_aspect_accuracy': evaluate_temporal_aspect(ground_truth, predicted),
        'recency_weight_rmse': evaluate_recency_weight(ground_truth, predicted),
        'year_filter_iou': evaluate_year_filter(ground_truth, predicted)
    }