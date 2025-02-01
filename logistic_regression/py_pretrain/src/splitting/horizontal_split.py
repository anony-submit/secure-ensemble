import numpy as np
import pandas as pd

def split_dataset_horizontally(X, y, n_splits, method='balanced', alpha=None, random_state=42):
    if method == 'balanced':
        X_splits, y_splits = split_dataset_horizontally_balanced(X, y, n_splits, random_state)
    elif method.startswith('dirichlet'):
        alpha = float(method.split('_')[1]) if '_' in method else alpha
        if alpha is None:
            raise ValueError("Alpha parameter must be provided for dirichlet splitting")
        X_splits, y_splits = split_dataset_horizontally_dirichlet(X, y, n_splits, alpha, random_state)
    else:
        raise ValueError(f"Unknown splitting method: {method}")
    
    return X_splits, y_splits

def split_dataset_horizontally_balanced(X, y, n_splits, random_state=42):
    np.random.seed(random_state)
    y_values = y.values if isinstance(y, pd.Series) else y
    class_0_indices = np.where(y_values == 0)[0]
    class_1_indices = np.where(y_values == 1)[0]
    
    n_class_0 = len(class_0_indices)
    n_class_1 = len(class_1_indices)
    
    class_0_split_sizes = [n_class_0 // n_splits] * n_splits
    class_1_split_sizes = [n_class_1 // n_splits] * n_splits
    
    for i in range(n_class_0 % n_splits):
        class_0_split_sizes[i] += 1
    for i in range(n_class_1 % n_splits):
        class_1_split_sizes[i] += 1
    
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)
    
    X_splits, y_splits = [], []
    class_0_start = 0
    class_1_start = 0
    
    for i in range(n_splits):
        c0_end = class_0_start + class_0_split_sizes[i]
        c1_end = class_1_start + class_1_split_sizes[i]
        
        split_indices = np.concatenate([
            class_0_indices[class_0_start:c0_end],
            class_1_indices[class_1_start:c1_end]
        ])
        
        class_0_start = c0_end
        class_1_start = c1_end
        
        X_splits.append(X.iloc[split_indices] if isinstance(X, pd.DataFrame) else X[split_indices])
        y_splits.append(y.iloc[split_indices] if isinstance(y, pd.Series) else y[split_indices])
    
    return X_splits, y_splits

def split_dataset_horizontally_dirichlet(X, y, n_splits, alpha, random_state=42):
    np.random.seed(random_state)
    total_samples = len(X)
    
    proportions = np.random.dirichlet([alpha] * n_splits)
    split_sizes = (proportions * total_samples).astype(int)
    split_sizes[-1] = total_samples - sum(split_sizes[:-1])
    
    indices = np.random.permutation(total_samples)
    start_idx = 0
    X_splits, y_splits = [], []
    
    for size in split_sizes:
        end_idx = start_idx + size
        split_indices = indices[start_idx:end_idx]
        X_splits.append(X.iloc[split_indices] if isinstance(X, pd.DataFrame) else X[split_indices])
        y_splits.append(y.iloc[split_indices] if isinstance(y, pd.Series) else y[split_indices])
        start_idx = end_idx
    
    for i in range(n_splits):
        y_values = y_splits[i].values if isinstance(y_splits[i], pd.Series) else y_splits[i]
        unique_classes = np.unique(y_values)

        for missing_class in [0, 1]:
            if missing_class not in unique_classes:
                max_samples_split = -1
                max_samples_count = -1
                
                for j in range(n_splits):
                    if j != i:
                        y_j_values = y_splits[j].values if isinstance(y_splits[j], pd.Series) else y_splits[j]
                        class_count = np.sum(y_j_values == missing_class)
                        if class_count > max_samples_count:
                            max_samples_count = class_count
                            max_samples_split = j
                
                if max_samples_split >= 0 and max_samples_count > 0:
                    y_source_values = y_splits[max_samples_split].values if isinstance(y_splits[max_samples_split], pd.Series) else y_splits[max_samples_split]
                    source_indices = np.where(y_source_values == missing_class)[0]
                    idx_to_move = source_indices[0]
                    
                    X_splits[i] = np.vstack([X_splits[i], 
                                          X_splits[max_samples_split][idx_to_move:idx_to_move+1]])
                    y_splits[i] = np.append(y_splits[i], 
                                          y_splits[max_samples_split][idx_to_move:idx_to_move+1])
                    
                    X_splits[max_samples_split] = np.delete(X_splits[max_samples_split], 
                                                          idx_to_move, axis=0)
                    y_splits[max_samples_split] = np.delete(y_splits[max_samples_split], 
                                                          idx_to_move)
    
    return X_splits, y_splits