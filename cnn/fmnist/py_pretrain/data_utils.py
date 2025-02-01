import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_dirichlet_split(dataset, n_splits, alpha, seed=42, save_path=None):
    np.random.seed(seed)
    n_classes = 10
    class_indices = {i: [] for i in range(n_classes)}
    
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    proportions = np.random.dirichlet(np.repeat(alpha, n_splits), size=n_classes)
    
    split_indices = [[] for _ in range(n_splits)]
    distribution = np.zeros((n_splits, n_classes))
    
    for class_idx, indices in class_indices.items():
        np.random.shuffle(indices)
        n_samples = len(indices)
        
        split_sizes = (proportions[class_idx] * n_samples).astype(int)
        split_sizes[-1] = n_samples - split_sizes[:-1].sum()
        
        start_idx = 0
        for split_idx, size in enumerate(split_sizes):
            split_indices[split_idx].extend(indices[start_idx:start_idx + size])
            distribution[split_idx][class_idx] = size
            start_idx += size
    
    if save_path:
        plt.figure(figsize=(12, 8))
        sns.heatmap(distribution, annot=True, fmt='g', cmap='YlOrRd')
        plt.xlabel('Class')
        plt.ylabel('Split')
        plt.title(f'Data Distribution (α={alpha})')
        plt.savefig(save_path)
        plt.close()
    
    return [torch.utils.data.Subset(dataset, indices) for indices in split_indices], distribution