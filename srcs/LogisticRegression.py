import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from numpy.random import rand
from sklearn.utils import shuffle


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

        self.X = []
        self.y = []
        self.batch_size = batch_size
        self.classes = []
        self.theta = []
        self.one_hot_y = []
        self.cost_history = []
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_history = []
        self.epsilon = 1e-7

    def load_train_set(self, X: np.ndarray, y: np.ndarray):
        self.X = self.add_intercept(X)
        self.y = y
        self.classes = np.unique(y)
        self.theta = rand(self.classes.shape[0], self.X.shape[1])
        self.one_hot_y = self.one_hot_encode(y)
        self.cost_history = []

    def set_classes(self, classes: np.ndarray):
        self.classes = np.array(classes)

    @staticmethod
    def add_intercept(X) -> np.ndarray:
        return np.c_[np.ones(X.shape[0]), X]

    @staticmethod
    def one_hot_encode(y: np.ndarray) -> np.ndarray:
        one_hot_y = np.zeros((y.shape[0], np.unique(y).shape[0]))
        for i in range(y.shape[0]):
            one_hot_y[i][y[i]] = 1
        return one_hot_y

    def hypothesis(self, theta, X) -> np.ndarray:
        return 1 / (1 + np.exp(-(np.dot(theta, X.T)))) - self.epsilon

    def cost(self, X, y, theta) -> float:
        h = self.hypothesis(X, theta)
        return (-1 / len(X)) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient_descent(self, X, y, theta, alpha, epochs) -> np.ndarray:
        m = len(X)
        for i in range(epochs):
            if self.batch_size is None: # Classic/Batch Gradient Descent
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
                h = self.hypothesis(theta[j], X_batch)
                theta[j] -= alpha * (1 / len(X_batch)) * \
                    np.dot((h - y_batch[:, j]), X_batch)

            self.cost_history.append(self.cost(X, y, theta))
            if i % 100 == 0 or i == self.max_iter - 1:
                print(
                    f'Epoch: {i}/{self.max_iter}\nCost: {self.cost(X, y, theta)}')

        return theta

    def fit(self):
        self.theta = self.gradient_descent(
            self.X, self.one_hot_y, self.theta, self.alpha, self.max_iter)

    def predict(self, X) -> np.ndarray:
        X = self.add_intercept(X)
        try:
            assert X.shape[1] == self.theta.shape[1], 'X must have the same number of features as theta'
        except AssertionError as e:
            print(e)
            sys.exit(1)
        return np.argmax(self.hypothesis(self.theta, X), axis=0)

    def save_thetas(self, filename='thetas/thetas.csv'):
        df = pd.DataFrame(self.theta)
        df.to_csv(filename, index=False)

    def load_thetas(self, filename):
        df = pd.read_csv(filename)
        self.theta = df.to_numpy()

    def accuracy(self, X, y) -> float:
        return np.mean(self.predict(X) == y)

    def print_predictions(self, X, y):
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
