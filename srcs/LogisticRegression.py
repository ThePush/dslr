import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import srcs.ml_toolkit as ml
from numpy.random import rand
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score



class LogisticRegression:
    def __init__(self, alpha=0.05, max_iter=1500, batch_size=None):
        try:
            assert isinstance(alpha, float), 'alpha must be a float'
            assert isinstance(max_iter, int), 'max_iter must be an int'
            assert isinstance(
                batch_size, int) or batch_size is None, 'batch_size must be an int or None'
            assert alpha > 0, 'alpha must be positive'
            assert max_iter > 0, 'max_iter must be positive'
            assert batch_size is None or batch_size > 0, 'batch_size must be positive'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        self.X = np.array([])
        self.y = np.array([])
        self.batch_size = batch_size
        self.classes = np.array([])
        self.theta = np.array([])
        self.one_hot_y = np.array([])
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = 1e-7

    def load_train_set(self, X: np.ndarray, y: np.ndarray):
        try:
            assert isinstance(X, np.ndarray), 'X must be a numpy.ndarray'
            assert isinstance(y, np.ndarray), 'y must be a numpy.ndarray'
            assert X.shape[0] == y.shape[0], 'X and y must have the same number of rows'
            assert X.shape[0] > 0, 'X must have at least one row'
            assert y.shape[0] > 0, 'y must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        self.X = self.add_intercept(X)
        self.y = y
        self.classes = np.unique(y)
        self.theta = rand(self.classes.shape[0], self.X.shape[1])
        self.one_hot_y = self.one_hot_encode(y)
        self.cost_history = []

    def set_classes(self, classes: list):
        try:
            assert isinstance(classes, list), 'classes must be a list'
            assert len(classes) > 0, 'classes must have at least one element'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        self.classes = np.array(classes)

    @staticmethod
    def add_intercept(X) -> np.ndarray:
        try:
            assert isinstance(X, np.ndarray), 'X must be a numpy.ndarray'
            assert X.shape[0] > 0, 'X must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        return np.c_[np.ones(X.shape[0]), X]

    @staticmethod
    def one_hot_encode(y: np.ndarray) -> np.ndarray:
        try:
            assert isinstance(y, np.ndarray), 'y must be a numpy.ndarray'
            assert y.shape[0] > 0, 'y must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        one_hot_y = np.zeros((y.shape[0], np.unique(y).shape[0]))
        for i in range(y.shape[0]):
            one_hot_y[i][y[i]] = 1
        return one_hot_y

    def hypothesis(self, X, theta) -> np.ndarray:
        try:
            assert isinstance(X, np.ndarray), 'X must be a numpy.ndarray'
            assert isinstance(
                theta, np.ndarray), 'theta must be a numpy.ndarray'
            assert X.shape[0] > 0, 'X must have at least one row'
            assert theta.shape[0] > 0, 'theta must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        return 1 / (1 + np.exp(-(np.dot(X, theta.T)))) - self.epsilon

    def cost(self, X, y, theta) -> float:
        h = self.hypothesis(X, theta)
        return (-1 / len(X)) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient_descent(self, X, y, theta, alpha, epochs) -> np.ndarray:
        m = len(X)
        for i in range(epochs):
            if self.batch_size is None:  # Classic/Batch Gradient Descent
                X_batch = X
                y_batch = y
            elif self.batch_size == 1:  # Stochastic Gradient Descent
                random_idx = np.random.randint(0, m)
                X_batch = X[random_idx].reshape(1, -1)
                y_batch = y[random_idx].reshape(1, -1)
            else:  # Mini-Batch Gradient Descent
                X_batch, y_batch = shuffle(X, y, random_state=42)
                X_batch = X_batch[:self.batch_size]
                y_batch = y_batch[:self.batch_size]

            for j in range(self.classes.shape[0]):
                h = self.hypothesis(X_batch, theta[j])
                theta[j] -= alpha * (1 / len(X_batch)) * \
                    np.dot((h - y_batch[:, j]), X_batch)

            self.cost_history.append(self.cost(X, y, theta))
            if i % 100 == 0 or i == self.max_iter - 1:
                print(
                    f'Epoch: {i}/{self.max_iter}\nCost: {self.cost(X, y, theta)}')

        return theta

    def fit(self):
        try:
            assert self.X.shape[0] > 0, 'X must have at least one row'
            assert self.y.shape[0] > 0, 'y must have at least one row'
            assert self.classes.shape[0] > 0, 'classes must have at least one row'
            assert self.theta.shape[0] > 0, 'theta must have at least one row'
            assert self.one_hot_y.shape[0] > 0, 'one_hot_y must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        self.theta = self.gradient_descent(
            self.X, self.one_hot_y, self.theta, self.alpha, self.max_iter)

    def predict(self, X) -> np.ndarray:
        try:
            assert isinstance(X, np.ndarray), 'X must be a numpy.ndarray'
            assert X.shape[0] > 0, 'X must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        X = self.add_intercept(X)
        try:
            assert X.shape[1] == self.theta.shape[1], 'X must have the same number of features as theta'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        return np.argmax(self.hypothesis(self.theta, X), axis=0)

    def save_thetas(self, filename='thetas/thetas.csv'):
        if not os.path.exists('thetas'):
            os.makedirs('thetas')
        if not filename.endswith('.csv'):
            filename += '.csv'

        df = pd.DataFrame(self.theta)
        df.to_csv(filename, index=False)

    def load_thetas(self, filename):
        try:
            ml.check_file(filename)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        df = pd.read_csv(filename)
        self.theta = df.to_numpy()

    def accuracy(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

    def print_predictions(self, X, y):
        try:
            assert isinstance(X, np.ndarray), 'X must be a numpy.ndarray'
            assert isinstance(y, np.ndarray), 'y must be a numpy.ndarray'
            assert X.shape[0] > 0, 'X must have at least one row'
            assert y.shape[0] > 0, 'y must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        y_hat = self.predict(X)
        for i in range(y.shape[0]):
            print(
                f'Prediction: {self.classes[y_hat[i]]} | Actual: {self.classes[y[i]]} | Correct: {y_hat[i] == y[i]}')

    def plot_cost_history(self):
        fig, ax = plt.subplots()
        ax.plot(self.cost_history)
        ax.set(xlabel='Iterations', ylabel='Cost', title='Cost History')
        plt.show()

    def confusion_matrix(self, X, y):
        y_hat = self.predict(X)
        cm = np.zeros((self.classes.shape[0], self.classes.shape[0]))
        for i in range(y.shape[0]):
            cm[y[i]][y_hat[i]] += 1
        return cm

    def plot_cm(self, X, y):
        try:
            assert isinstance(X, np.ndarray), 'X must be a numpy.ndarray'
            assert isinstance(y, np.ndarray), 'y must be a numpy.ndarray'
            assert X.shape[0] > 0, 'X must have at least one row'
            assert y.shape[0] > 0, 'y must have at least one row'
        except AssertionError as e:
            print(e)
            sys.exit(1)

        cm = self.confusion_matrix(X, y)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.classes,
               yticklabels=self.classes,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center", color="w")
        plt.show()
