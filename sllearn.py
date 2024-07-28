import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import trange
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier


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
    

def auto_modeller(X, y, model_class, test_size=0.25, simulations=100):
    """
    Runs simulations to determine the best setting for k in a given model.
    
    Parameters:
    - X: Features
    - y: Labels
    - model_class: Class of the model to instantiate with k neighbors
    - test_size: Proportion of the dataset to include in the test split
    - simulations: Number of random simulations to perform
    
    Returns:
    - training_accuracies: DataFrame with training accuracies for each simulation
    - testing_accuracies: DataFrame with testing accuracies for each simulation
    - k_values: Range of k settings tried
    """
    training_accuracies = pd.DataFrame()
    testing_accuracies = pd.DataFrame()
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    for seed in trange(1, simulations + 1, desc="Simulations"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        
        train_accuracy = []
        test_accuracy = []
        k_values = range(1, len(X_train))  # Try n_neighbors from 1 to len(X_train) - 1
        
        for k in k_values:   
            clf = model_class(k)  # Instantiate the model with k neighbors
            clf.fit(X_train, y_train)
            train_accuracy.append(clf.score(X_train, y_train))  # Record training set accuracy
            test_accuracy.append(clf.score(X_test, y_test))  # Record generalization accuracy
        
        training_accuracies[seed] = train_accuracy
        testing_accuracies[seed] = test_accuracy
    
    return training_accuracies, testing_accuracies, k_values


def model_plotter(xlabel, k_values, training_accuracies, testing_accuracies, accuracy_type='test'):
    """
    Plots training and testing accuracies with error bars.
    
    Parameters:
    - xlabel: Label for the x-axis
    - k_values: Range of k settings tried
    - training_accuracies: DataFrame with training accuracies for each simulation
    - testing_accuracies: DataFrame with testing accuracies for each simulation
    - accuracy_type: Type of accuracy to display ('test' or 'train')
    """
    plt.errorbar(k_values, training_accuracies.mean(axis=1),
                 yerr=training_accuracies.std(axis=1) / 4,
                 label="Training Accuracy",
                 color='darkturquoise')
    plt.errorbar(k_values, testing_accuracies.mean(axis=1),
                 yerr=testing_accuracies.std(axis=1) / 4,
                 label=f"{accuracy_type.capitalize()} Accuracy",
                 color='lightcoral')
    plt.ylabel("Accuracy")
    plt.xlabel(xlabel)
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.title(f"{accuracy_type.capitalize()} Accuracy vs. Number of Neighbors")
    plt.show()

    
def compare_model_plots(model_1, model_2, accuracy_type='test'):
    """
    Compares and plots the test results of two models.
    
    Parameters:
    - model_1: Tuple containing (testing_accuracies_1, name_1, k_values_1)
    - model_2: Tuple containing (testing_accuracies_2, name_2, k_values_2)
    - accuracy_type: Type of accuracy to display ('test' or 'train')
    """
    testing_accuracies_1, name_1, k_values_1 = model_1
    testing_accuracies_2, name_2, k_values_2 = model_2
    
    means_1 = testing_accuracies_1.mean(axis=1)
    means_2 = testing_accuracies_2.mean(axis=1)

    plt.errorbar(k_values_1, means_1,
                 yerr=testing_accuracies_1.std(axis=1) / 2,
                 label=f"{name_1} {accuracy_type.capitalize()} Accuracy")
    plt.errorbar(k_values_2, means_2,
                 yerr=testing_accuracies_2.std(axis=1) / 2,
                 label=f"{name_2} {accuracy_type.capitalize()} Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Neighbors (k)")
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.title(f"Comparison of {accuracy_type.capitalize()} Accuracy")
    plt.show()

    print(f'{name_1} Maximum {accuracy_type} accuracy is {means_1[means_1.argmax()]:.6f}'
          f' for {k_values_1[means_1.argmax()]} neighbors')
    print(f'{name_2} Maximum {accuracy_type} accuracy is {means_2[means_2.argmax()]:.6f}'
          f' for {k_values_2[means_2.argmax()]} neighbors')


def combined_model_weighting(X, y, max_neighbor, max_farthest,
                             test_size=0.25, simulations=1000, weight_step=0.01):
    """
    Performs weighted averaging of KNN and KFN model predictions to find the optimal weight.

    Parameters:
    - X: Features
    - y: Labels
    - max_neighbor: Maximum number of neighbors for KNN
    - max_farthest: Maximum number of farthest neighbors for KFN
    - test_size: Proportion of the dataset to include in the test split
    - simulations: Number of random simulations to perform
    - weight_step: Step size for weight settings from 0 to 1
    
    Returns:
    - wa_training: DataFrame with training accuracies for each simulation
    - wa_validation: DataFrame with validation accuracies for each simulation
    - weight_settings: Array of weight settings tried
    """
    wa_training = pd.DataFrame()
    wa_validation = pd.DataFrame()
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    weight_settings = np.arange(0.00, 1.01, weight_step)
    
    for seed in trange(1, simulations + 1, desc="Simulations"):
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        
        model_knn = KNeighborsClassifier(n_neighbors=max_neighbor)
        model_knn.fit(X_train, y_train)
        pred_knn_train = model_knn.predict_proba(X_train)
        pred_knn_val = model_knn.predict_proba(X_validation)

        model_kfn = KFarthestClassifier(k=max_farthest)
        model_kfn.fit(X_train, y_train)
        pred_kfn_train = model_kfn.predict_proba(X_train)
        pred_kfn_val = model_kfn.predict_proba(X_validation)

        train_accuracies = []
        val_accuracies = []
        
        for weight in weight_settings:
            pred_train_combined = pred_knn_train * weight + pred_kfn_train * (1 - weight)
            pred_val_combined = pred_knn_val * weight + pred_kfn_val * (1 - weight)
            
            pred_train_labels = pd.DataFrame(
                pred_train_combined,
                columns=model_knn.classes_
            ).idxmax(axis="columns").values
            pred_val_labels = pd.DataFrame(
                pred_val_combined,
                columns=model_knn.classes_
            ).idxmax(axis="columns").values
            
            train_accuracies.append(accuracy_score(y_train, pred_train_labels))
            val_accuracies.append(accuracy_score(y_validation, pred_val_labels))
        
        wa_training[seed] = train_accuracies
        wa_validation[seed] = val_accuracies
    
    return wa_training, wa_validation, weight_settings
