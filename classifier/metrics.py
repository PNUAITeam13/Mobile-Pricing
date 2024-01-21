import numpy as np
from functools import wraps
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class MetricsWrapper():
    @staticmethod
    def np_array(func):
        @wraps(func)
        def wrapper(y_test, y_pred, matrix, *args, **kwargs):
            if not isinstance(y_test, np.ndarray):
                y_test = np.array(y_test)
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.array(y_pred)
                
            return func(y_test, y_pred, matrix, *args, **kwargs)
            
        return wrapper

class Metrics():

    @staticmethod
    def confusion_matrix(y_test, y_pred):
        if not isinstance(y_test, np.ndarray):
                y_test = np.array(y_test)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
            
        true_positive = np.sum((y_test == 1) & (y_pred == 1))
        false_positive = np.sum((y_test == 0) & (y_pred == 1))
        false_negative = np.sum((y_test == 1) & (y_pred == 0))
        true_negative = np.sum((y_test == 0) & (y_pred == 0))

        return np.array([[true_negative, false_positive], [false_negative, true_positive]])

    
    @staticmethod
    def show_conf_matrix(matrix):
        fig, ax = plt.subplots()
        sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()


    @staticmethod
    @MetricsWrapper.np_array
    def accuracy_score(y_test, y_pred, matrix=None):
        if matrix is None:
            true_positive = np.sum((y_test == 1) & (y_pred == 1))
            false_positive = np.sum((y_test == 0) & (y_pred == 1))
            false_negative = np.sum((y_test == 1) & (y_pred == 0))
            true_negative = np.sum((y_test == 0) & (y_pred == 0))
            
            return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        
        return (matrix[1][1] + matrix[0][0]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])


    @staticmethod
    @MetricsWrapper.np_array
    def recall_score(y_test, y_pred, matrix=None):
        if matrix is None:
            true_positive = np.sum((y_test == 1) & (y_pred == 1))
            false_negative = np.sum((y_test == 1) & (y_pred == 0))
    
            return true_positive / (true_positive + false_negative)

        return matrix[1][1] / (matrix[1][1] + matrix[1][0])

    
    @staticmethod
    @MetricsWrapper.np_array
    def precision_score(y_test, y_pred, matrix=None):
        if matrix is None:
            true_positive = np.sum((y_test == 1) & (y_pred == 1))
            false_positive = np.sum((y_test == 0) & (y_pred == 1))
    
            return true_positive / (true_positive + false_positive)
   
        return matrix[1][1] / (matrix[1][1] + matrix[0][1])
        

    @staticmethod
    @MetricsWrapper.np_array
    def f1_score(y_test, y_pred, matrix=None):
        precision = Metrics.precision_score(y_test, y_pred, matrix=matrix)
        recall = Metrics.recall_score(y_test, y_pred, matrix=matrix)

        return (2 * precision * recall) / (precision + recall)
