import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DatasetLoader:
    @staticmethod
    def load_wdbc(path):
        data = pd.read_csv(path, header=None)
        X = data.iloc[:, 2:]
        y = data.iloc[:, 1]
        y = np.where(y == 'M', 1, 0)
        return X, y
    
    @staticmethod
    def load_heart_disease(path):
        data = pd.read_csv(path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y
    
    @staticmethod
    def load_diabetes(path):
        data = pd.read_csv(path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    if isinstance(y, pd.Series):
        y = y.reset_index(drop=True)
    return X_scaled, y

def save_test_data(X_test, y_test, dataset_name):
    test_data = pd.concat([X_test, pd.Series(y_test, name='target')], axis=1)
    test_data.to_csv(f'data/{dataset_name}/{dataset_name}_test.csv', index=False, header=False)