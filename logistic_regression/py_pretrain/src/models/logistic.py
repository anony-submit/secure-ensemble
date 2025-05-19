import numpy as np
from sklearn.linear_model import LogisticRegression

class VerticalLogisticRegression:
    """Wrapper for LogisticRegression for vertical splitting scenario"""
    def __init__(self, feature_subset, all_features, random_state=42):
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.feature_subset = list(feature_subset)
        self.all_features = list(all_features)
        self.feature_indices = [self.all_features.index(f) for f in self.feature_subset]
    
    def fit(self, X, y):
        self.model.fit(X[self.feature_subset], y)
        return self
    
    def predict(self, X):
        return self.model.predict(X[self.feature_subset])
    
    def predict_proba(self, X):
        return self.model.predict_proba(X[self.feature_subset])
    
    def decision_function(self, X):
        return self.model.decision_function(X[self.feature_subset])
    
    def get_full_coef(self):
        """Get coefficients for all features, with zeros for unused features"""
        full_coef = np.zeros(len(self.all_features))
        for idx, coef in zip(self.feature_indices, self.model.coef_[0]):
            full_coef[idx] = coef
        return full_coef