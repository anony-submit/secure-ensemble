# src/main/run_experiments.py
from sklearn.model_selection import train_test_split
from ..utils.data_utils import DatasetLoader, preprocess_data, save_test_data
from ..utils.file_utils import create_experiment_directories
from ..experiment.runner import run_experiment
import warnings


def get_n_values(dataset_name, split_type):
    """Get valid n values for each dataset and split type"""
    if split_type == 'horizontal':
        return [2, 5, 10, 20]
    else:  # vertical
        if dataset_name == 'wdbc':
            return [2, 5, 10, 20]
        elif dataset_name == 'heart_disease':
            return [2, 5, 10]
        else:  # diabetes
            return [2, 5]

def main():
    create_experiment_directories()
    
    datasets = [
        ('wdbc', 'data/wdbc/wdbc.data'),
        ('heart_disease', 'data/heart_disease/Heart_disease_cleveland.csv'),
        ('pima', 'data/pima/diabetes.csv')
    ]
    
    methods = ['ideal', 'balanced', 'dirichlet_0.5', 'dirichlet_0.1']
    split_types = ['horizontal', 'vertical']
    
    for dataset_name, data_path in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load and preprocess data
        if dataset_name == 'wdbc':
            X, y = DatasetLoader.load_wdbc(data_path)
        elif dataset_name == 'heart_disease':
            X, y = DatasetLoader.load_heart_disease(data_path)
        else:
            X, y = DatasetLoader.load_diabetes(data_path)
        
        X, y = preprocess_data(X, y)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save test set for future use
        save_test_data(X_test, y_test, dataset_name)
        
        # Run experiments
        for split_type in split_types:
            print(f"\nRunning {split_type} split experiments")
            
            n_values = get_n_values(dataset_name, split_type)
            
            for method in methods:
                print(f"\nMethod: {method}")
                
                if method == 'ideal':
                    metrics = run_experiment(
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        n_splits=1,
                        split_type=split_type,
                        method=method,
                        dataset_name=dataset_name
                    )
                    print(f"Ideal case - Mean accuracy: {metrics['mean_accuracy']:.4f}")
                else:
                    for n in n_values:
                        metrics = run_experiment(
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test,
                            n_splits=n,
                            split_type=split_type,
                            method=method,
                            dataset_name=dataset_name
                        )
                        print(f"n={n} - Mean accuracy: {metrics['mean_accuracy']:.4f}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
    
    
