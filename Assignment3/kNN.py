import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Load and preprocess the dataset
file_path = "pima-indians-diabetes.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(file_path, names=column_names)

# Select features and target
X = data[["Glucose", "BMI"]].values
y = data["Outcome"].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implement KNN from scratch
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute distances
        if self.distance_metric == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Invalid distance metric. Choose 'euclidean' or 'manhattan'.")
        
        # Get k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Visualize the results
def plot_knn_results(X_train, y_train, X_test, y_pred, title, ax):
    # Plot training points
    ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0 (No Diabetes)', alpha=0.6)
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1 (Diabetes)', alpha=0.6)
    
    # Plot test points
    ax.scatter(X_test[y_pred == 0][:, 0], X_test[y_pred == 0][:, 1], color='cyan', label='Predicted Class 0', marker='*', s=100)
    ax.scatter(X_test[y_pred == 1][:, 0], X_test[y_pred == 1][:, 1], color='orange', label='Predicted Class 1', marker='x', s=100)
    
    ax.set_xlabel('Glucose (Standardized)')
    ax.set_ylabel('BMI (Standardized)')
    ax.set_title(title)
    ax.legend()

# Ask the user for k
k = int(input("Enter the value of k: "))

# Initialize and train KNN for both distance metrics
knn_euclidean = KNN(k=k, distance_metric='euclidean')
knn_euclidean.fit(X_train, y_train)
y_pred_euclidean = knn_euclidean.predict(X_test)

knn_manhattan = KNN(k=k, distance_metric='manhattan')
knn_manhattan.fit(X_train, y_train)
y_pred_manhattan = knn_manhattan.predict(X_test)

# Create subplots for side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Euclidean distance results
plot_knn_results(X_train, y_train, X_test, y_pred_euclidean, f'KNN Classification (k={k}, Euclidean Distance)', ax1)

# Plot Manhattan distance results
plot_knn_results(X_train, y_train, X_test, y_pred_manhattan, f'KNN Classification (k={k}, Manhattan Distance)', ax2)

# Show the plots
plt.tight_layout()
plt.show()