import re
from typing import Dict, List, Set, Union

def evaluate_temporal_aspect(ground_truth: Dict, predicted: Dict) -> float:
    """Evaluate the accuracy of temporal aspect detection."""
    correct = sum(1 for q in ground_truth if ground_truth[q]['has_temporal_aspect'] == predicted[q]['has_temporal_aspect'])
    return correct / len(ground_truth)

def evaluate_recency_weight(ground_truth: Dict, predicted: Dict, tolerance: float = 1.0) -> float:
    """Evaluate the accuracy of recency weight prediction within a tolerance."""
    correct = 0
    total = 0
    for q in ground_truth:
        gt_weight = ground_truth[q]['expected_recency_weight']
        pred_weight = predicted[q]['expected_recency_weight']
        
        if gt_weight is None and pred_weight is None:
            correct += 1
        elif gt_weight is not None and pred_weight is not None:
            if abs(gt_weight - pred_weight) <= tolerance:
                correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def parse_year_filter(filter_str: Union[str, None]) -> Set[int]:
    """Parse a year filter string and return a set of years that satisfy the filter."""
    if filter_str is None:
        return set(range(1950, 2025))  # If no filter, assume all years are valid
    
    years = set()
    for year in range(1950, 2025):
        # Replace 'year' in the filter string with the actual year
        condition = filter_str.replace('year', str(year))
        try:
            if eval(condition):
                years.add(year)
        except:
            print(f"Warning: Could not evaluate condition '{condition}' for year {year}")
    return years

def evaluate_year_filter(ground_truth: Dict, predicted: Dict) -> float:
    """Evaluate the accuracy of year filter prediction using Intersection over Union."""
    iou_scores = []
    for q in ground_truth:
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
        'recency_weight_accuracy': evaluate_recency_weight(ground_truth, predicted),
        'year_filter_iou': evaluate_year_filter(ground_truth, predicted)
    }

# Example usage
ground_truth = {
    'Q001': {
        'has_temporal_aspect': True,
        'expected_year_filter': 'year >= 2020',
        'expected_recency_weight': 8
    },
    'Q002': {
        'has_temporal_aspect': True,
        'expected_year_filter': '(year >= 2010 and year <= 2020) or year > 2022',
        'expected_recency_weight': 5
    },
    'Q003': {
        'has_temporal_aspect': False,
        'expected_year_filter': None,
        'expected_recency_weight': None
    }
}

predicted = {
    'Q001': {
        'has_temporal_aspect': True,
        'expected_year_filter': 'year >= 2019',
        'expected_recency_weight': 7
    },
    'Q002': {
        'has_temporal_aspect': True,
        'expected_year_filter': 'year >= 2010 and year <= 2022',
        'expected_recency_weight': 6
    },
    'Q003': {
        'has_temporal_aspect': True,
        'expected_year_filter': 'year >= 2000',
        'expected_recency_weight': 3
    }
}

results = evaluate_temporal_queries(ground_truth, predicted)
print(results)