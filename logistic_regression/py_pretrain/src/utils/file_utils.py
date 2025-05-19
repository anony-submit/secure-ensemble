import os
import json
from datetime import datetime

def create_experiment_directories():
    """Create directory structure for experiment results"""
    base_dirs = ['ideal', 'balanced', 'dirichlet_0.5', 'dirichlet_0.1']
    for base_dir in base_dirs:
        for subdir in ['models', 'metrics']:
            os.makedirs(f'experiment_results/{base_dir}/{subdir}', exist_ok=True)

def get_experiment_path(exp_type, subdir, dataset_name, split_type, n, ext):
    """Generate consistent file paths for experiment results"""
    filename = f"{dataset_name}_{split_type}_n{n}{ext}"
    return os.path.join('experiment_results', exp_type, subdir, filename)

def save_json(data, filepath):
    """Save data as JSON with consistent formatting"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)