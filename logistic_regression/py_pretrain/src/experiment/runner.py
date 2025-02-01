import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from ..models.logistic import VerticalLogisticRegression
from ..models.ensemble import ensemble_predict_with_methods
from ..splitting.horizontal_split import split_dataset_horizontally
from ..splitting.vertical_split import split_dataset_vertically
from .metrics import calculate_metrics
from ..utils.file_utils import save_json, get_experiment_path

def run_experiment(X_train, X_test, y_train, y_test, n_splits, split_type, method, dataset_name, random_state=42):
    """Run single experiment with given parameters"""
    if method == 'ideal':
        # For ideal case, use entire training set
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Get predictions
        individual_predictions = [model.predict(X_test)]
        ensemble_results = {
            'hard_voting': {
                'predictions': model.predict(X_test),
                'probabilities': model.predict_proba(X_test)[:, 1]
            },
            'soft_voting': {
                'predictions': model.predict(X_test),
                'probabilities': model.predict_proba(X_test)[:, 1]
            },
            'linear_soft_voting': {
                'predictions': model.predict(X_test),
                'probabilities': model.predict_proba(X_test)[:, 1]
            }
        }
        models = [model]
        
    else:
        if split_type == 'horizontal':
            splits = split_dataset_horizontally(X_train, y_train, n_splits, method, random_state=random_state)
            X_splits, y_splits = splits
            
            # Train models
            models = []
            for i in range(n_splits):
                model = LogisticRegression(max_iter=1000, random_state=random_state)
                model.fit(X_splits[i], y_splits[i])
                models.append(model)
            
            # Get predictions on test set
            individual_predictions = [model.predict(X_test) for model in models]
            
        else:  # vertical
            X_splits = split_dataset_vertically(X_train, n_splits, method, random_state=random_state)
            
            # Train models
            models = []
            for i in range(n_splits):
                model = VerticalLogisticRegression(X_splits[i].columns, X_train.columns, random_state)
                model.fit(X_train, y_train)
                models.append(model)
            
            # Get predictions on test set
            individual_predictions = [model.predict(X_test) for model in models]
        
        # Get ensemble predictions
        ensemble_results = ensemble_predict_with_methods(models, X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, individual_predictions, ensemble_results)
    
    # Save model parameters
    model_params = []
    for model in models:
        if hasattr(model, 'get_full_coef'):
            weights = model.get_full_coef()
            intercept = model.model.intercept_
        else:
            weights = model.coef_[0]
            intercept = model.intercept_
        
        params = {
            "weights": weights.tolist(),
            "intercept": intercept.tolist()
        }
        model_params.append(params)
    
    # Save results
    model_path = get_experiment_path(method, 'models', dataset_name, split_type, n_splits, '_models.json')
    save_json(model_params, model_path)
    
    metrics_path = get_experiment_path(method, 'metrics', dataset_name, split_type, n_splits, '_metrics.json')
    save_json(metrics, metrics_path)
    
    return metrics