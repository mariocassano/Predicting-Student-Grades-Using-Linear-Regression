import numpy as np
np.random.seed(1)
class LinearRegression:
    def __init__(self, alpha, steps, features):
        self.alpha = alpha
        self.steps = steps
        self.theta = np.random.rand(features)

    def train(self, x, y):
        cost = np.zeros(self.steps)
        m = len(x)
        for i in range(0, self.steps):
            predictions = np.dot(x, self.theta)
            error = predictions - y
            cost[i] = (1/(2*m)) * np.dot(error.T,error)
            self.theta -= self.alpha * (1/m) * np.dot(x.T, error)
        return cost

    def predict(self, x):
        return np.dot(x, self.theta)