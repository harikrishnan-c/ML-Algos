import numpy as np


class LinearRegression:
    def __init__(self):
        self.__W = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, fit_intercept=True, is_gd=False, iterations=1000, lr=0.01):
        n = X.shape[0]
        m = X.shape[1]
        X_b = X
        # Initializing random weights
        self.__W = np.random.randn(m, 1)
        # Adding Bias Term
        if fit_intercept:
            X_b = np.append(np.ones(n).reshape(n, 1), X, 1)
            self.__W = np.random.randn(m + 1, 1)
        if is_gd:
            for i in range(iterations):
                # Calculating Gradient
                grad = 2 / n * X_b.T @ (X_b @ self.__W - y)
                self.__W = self.__W - lr * grad
        else:
            self.__W = np.linalg.inv(X_b.T @ (X_b)) @ (X_b.T) @ (y)
        if fit_intercept:
            self.coef_ = self.__W[1:]
            self.intercept_ = self.__W[0]
        else:
            self.coef_ = self.__W

    def predict(self, X):
        if self.coef_ is not None:
            if self.intercept_ is not None:
                return np.append(np.ones(X.shape[0]).reshape(X.shape[0], 1), X, 1) @ self.__W
            return X @ self.__W
        else:
            raise Exception("Linear Regression Instance not fitted Yet")
