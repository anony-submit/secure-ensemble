import numpy as np
from sklearn.metrics import accuracy_score

def calculate_metrics(y_test, individual_predictions, ensemble_results):
    """Calculate metrics for individual models and ensemble methods"""
    individual_accuracies = [accuracy_score(y_test, pred) for pred in individual_predictions]
    
    ensemble_accuracies = {}
    for method_name, results in ensemble_results.items():
        ensemble_accuracies[method_name] = accuracy_score(y_test, results['predictions'])
    
    metrics = {
        'individual_accuracies': individual_accuracies,
        'mean_accuracy': float(np.mean(individual_accuracies)),
        'ensemble_accuracies': ensemble_accuracies
    }
    
    return metrics