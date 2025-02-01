import numpy as np
import pandas as pd

def split_dataset_vertically(X, n_splits, method='balanced', alpha=None, random_state=42):
    """Wrapper function for vertical splitting methods"""
    if method == 'balanced':
        return split_dataset_vertically_balanced(X, n_splits, random_state)
    elif method in ['dirichlet_0.5', 'dirichlet_0.1']:
        alpha = 0.5 if method == 'dirichlet_0.5' else 0.1
        return split_dataset_vertically_dirichlet(X, n_splits, alpha, random_state)
    elif method == 'ideal':
        return [X]  # Return the entire feature set for ideal case
    else:
        raise ValueError(f"Unknown splitting method: {method}")

def split_dataset_vertically_balanced(X, n_splits, random_state=42):
    """Split features evenly"""
    np.random.seed(random_state)
    feature_count = X.shape[1]
    
    base_size = feature_count // n_splits
    remainder = feature_count % n_splits
    split_sizes = [base_size + 1 if i < remainder else base_size for i in range(n_splits)]
    
    feature_indices = np.random.permutation(feature_count)
    X_splits = []
    start_idx = 0
    
    for size in split_sizes:
        end_idx = start_idx + size
        selected_features = feature_indices[start_idx:end_idx]
        if isinstance(X, pd.DataFrame):
            selected_columns = X.columns[selected_features]
            X_splits.append(X[selected_columns])
        else:
            X_splits.append(X[:, selected_features])
        start_idx = end_idx
    
    return X_splits

def split_dataset_vertically_dirichlet(X, n_splits, alpha, random_state=42):
    np.random.seed(random_state)
    feature_count = X.shape[1]
    
    max_attempts = 1000
    for attempt in range(max_attempts):
        proportions = np.random.dirichlet([alpha] * n_splits)
        split_sizes = np.maximum((proportions * feature_count).astype(int), 1)
        remaining_features = feature_count - sum(split_sizes[:-1])
        
        if remaining_features >= 1:
            split_sizes[-1] = remaining_features
            feature_indices = np.random.permutation(feature_count)
            X_splits = []
            start_idx = 0
            
            for size in split_sizes:
                end_idx = start_idx + size
                selected_features = feature_indices[start_idx:end_idx]
                if isinstance(X, pd.DataFrame):
                    selected_columns = X.columns[selected_features]
                    X_splits.append(X[selected_columns])
                else:
                    X_splits.append(X[:, selected_features])
                start_idx = end_idx
            
            return X_splits
            
    raise ValueError(f"Could not find valid split after {max_attempts} attempts")