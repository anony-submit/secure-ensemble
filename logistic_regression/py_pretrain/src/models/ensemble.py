import numpy as np

def sigmoid(x):
    """Sigmoid function for linear soft voting"""
    return 1 / (1 + np.exp(-x))

def ensemble_predict_with_methods(models, X):
    """
    Perform ensemble prediction using multiple voting methods:
    - Hard voting: sum of binary predictions
    - Soft voting: sum of probability predictions
    - Linear soft voting: sum of decision function outputs, then sigmoid
    """
    n_samples = len(X)
    n_models = len(models)
    
    hard_predictions = np.zeros((n_models, n_samples))
    soft_probabilities = np.zeros((n_models, n_samples))
    linear_scores = np.zeros((n_models, n_samples))
    
    for i, model in enumerate(models):
        hard_predictions[i] = model.predict(X)
        soft_probabilities[i] = model.predict_proba(X)[:, 1]
        linear_scores[i] = model.decision_function(X)
    
    results = {
        'hard_voting': {
            'predictions': (np.sum(hard_predictions, axis=0) >= n_models/2).astype(int),
            'probabilities': np.sum(hard_predictions, axis=0) / n_models
        },
        'soft_voting': {
            'predictions': (np.sum(soft_probabilities, axis=0) >= n_models/2).astype(int),
            'probabilities': np.sum(soft_probabilities, axis=0) / n_models
        },
        'linear_soft_voting': {
            'predictions': (sigmoid(np.sum(linear_scores, axis=0)) >= 0.5).astype(int),
            'probabilities': sigmoid(np.sum(linear_scores, axis=0))
        }
    }
    
    return results