import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        return np.round(-1/n * np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        # for 3 samples and 4 class, y_true*y_pred gives (3, 4)
        # you need to sum it based on axis=1 to 
        # get the sum of sample loss on 4 classes
        loss = -np.mean(np.sum(y_true*np.log(y_pred), axis = 1))
        return round(loss, 4)
