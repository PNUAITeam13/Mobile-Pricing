import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.__learning_rate = learning_rate
        self.__n_iters = n_iters
        self.__classifiers = {}

    def fit(self, X, y):
        for cls in np.unique(y):
            binary_y = np.where(y == cls, 1, 0)
            self.__classifiers[cls] = self.__train_classifier(X, binary_y)

    def __train_classifier(self, X, y):
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        for _ in range(self.__n_iters):
            linear_pred = np.dot(X, weights) + bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            weights = weights - self.__learning_rate * dw
            bias = bias - self.__learning_rate * db

        return weights, bias

    def predict(self, X):
        class_preds = []
        for cls, classifier in self.__classifiers.items():
            weights, bias = classifier
            linear_pred = np.dot(X, weights) + bias
            y_pred = sigmoid(linear_pred)
            class_preds.append(y_pred)

        class_preds = np.array(class_preds).T
        predicted_class = np.argmax(class_preds, axis=1)
        return predicted_class
