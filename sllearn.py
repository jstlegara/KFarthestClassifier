import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist

class KFarthestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_farthest: int = 5):
        self.n_farthest = n_farthest

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KFarthestClassifier':
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        if len(self.classes_) < 2:
            raise ValueError("KFarthestClassifier requires at least 2 classes in the data.")
        
        self.X_ = X
        self.y_ = y
        self.class_counts_ = Counter(y)
        
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        proba = np.zeros((X.shape[0], len(self.classes_)))
        
        # Calculate all distances in one go using cdist
        distances = cdist(X, self.X_, metric='euclidean')
        
        for i, dist in enumerate(distances):
            # Ensure k does not exceed the number of training samples
            k = min(self.n_farthest, len(dist))
            farthest_indices = np.argpartition(dist, -k)[-k:]
            class_counts = Counter(self.y_[farthest_indices])
            
            for cls in self.classes_:
                if cls not in class_counts:
                    class_counts[cls] = 0

            inverse_proba = {cls: (self.n_farthest - count) / self.n_farthest for cls,
                             count in class_counts.items()}
            total_inverse_proba = sum(inverse_proba.values())
            
            for cls in self.classes_:
                proba[i, self.classes_ == cls] = inverse_proba[cls] / total_inverse_proba
        
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        proba = self.predict_proba(X)
        
        y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)
        
        for i, point_proba in enumerate(proba):
            tied_classes = self.classes_[point_proba == np.max(point_proba)]
            if len(tied_classes) == 1:
                y_pred[i] = tied_classes[0]
            else:
                max_population = max(self.class_counts_[cls] for cls in tied_classes)
                most_populated_classes = [
                    cls for cls in tied_classes if self.class_counts_[cls] == max_population
                ]
                
                if len(most_populated_classes) == 1:
                    y_pred[i] = most_populated_classes[0]
                else:
                    y_pred[i] = sorted(most_populated_classes)[0]
        
        return y_pred
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)