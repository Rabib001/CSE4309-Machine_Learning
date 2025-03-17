# Name: Rabib Husain
# ID: 1002053770

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Compute Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Implementation from scratch
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# SVM Implementation from scratch (Hard Margin)
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# MLChoice class to handle dataset loading, model training, and visualization
class MLChoice:
    def __init__(self, ml_algorithm, dataset):
        self.ml_algorithm = ml_algorithm
        self.dataset = dataset
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_split_data()

    def load_and_split_data(self):
        if self.dataset == 'iris':
            data = load_iris()
            X = data.data
            y = data.target
        elif self.dataset == 'banknote':
            data = np.loadtxt('data_banknote_authentication.txt', delimiter=',')
            X = data[:, :-1]
            y = data[:, -1]
        elif self.dataset == 'breast_cancer':
            data = load_breast_cancer()
            X = data.data
            y = data.target
        else:
            raise ValueError("Unknown dataset")

        # Split the data into training and testing sets
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def fit_and_predict(self):
        # Scratch implementation
        if self.ml_algorithm == 'knn':
            model_scratch = KNN(k=3)
        elif self.ml_algorithm == 'svm':
            model_scratch = SVM()
        else:
            raise ValueError("Unknown ML algorithm")

        # Train the scratch model
        model_scratch.fit(self.X_train, self.y_train)
        predictions_scratch = model_scratch.predict(self.X_test)
        accuracy_scratch = accuracy_score(self.y_test, predictions_scratch)

        # Scikit-Learn implementation
        if self.ml_algorithm == 'knn':
            model_sklearn = KNeighborsClassifier(n_neighbors=3)
        elif self.ml_algorithm == 'svm':
            model_sklearn = SVC(kernel='linear')

        # Train the Scikit-Learn model
        model_sklearn.fit(self.X_train, self.y_train)
        predictions_sklearn = model_sklearn.predict(self.X_test)
        accuracy_sklearn = accuracy_score(self.y_test, predictions_sklearn)

        # Output results
        print(f"DataSet: {self.dataset}")
        print(f"Machine Learning Algorithm Chosen: {self.ml_algorithm}")
        print(f"Accuracy of Training (Scratch): {accuracy_scratch * 100:.2f}%")
        print(f"Accuracy of ScikitLearn Function: {accuracy_sklearn * 100:.2f}%")
        print(f"Prediction Point: {self.X_test[0]}")
        print(f"Predicted Class (Scratch): {predictions_scratch[0]}")
        print(f"Predicted Class (Scikit-Learn): {predictions_sklearn[0]}")
        print(f"Actual Class: {self.y_test[0]}")

        # Visualize the dataset in 3D
        self.visualize_3d()

    def visualize_3d(self):
        # Reduce dimensions to 3 using PCA
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(self.X_train)

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for each class
        for label in np.unique(self.y_train):
            ax.scatter(
                X_3d[self.y_train == label, 0],  # PCA Component 1
                X_3d[self.y_train == label, 1],  # PCA Component 2
                X_3d[self.y_train == label, 2],  # PCA Component 3
                label=f"Class {label}"
            )

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.set_title(f'3D Visualization of {self.dataset} Dataset (PCA)')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python MLChoice.py <ML> <DataSet>")
        print("ML Options: knn, svm")
        print("DataSet Options: iris, banknote, breast_cancer")
        sys.exit(1)

    ml_algorithm = sys.argv[1]
    dataset = sys.argv[2]

    try:
        ml_choice = MLChoice(ml_algorithm, dataset)
        ml_choice.fit_and_predict()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)