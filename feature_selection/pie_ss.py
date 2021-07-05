import numpy as np
import os

from settings import ROOT_DIR
os.chdir(ROOT_DIR)

# print(os.getcwd())


class Coordinate_Descent_Lasso:
    """
    Parameters
    -----------
    :param alpha: alpha value
    :param max_iter: maximum number of iterations
    :param fit_intercept: boolean indicating if the intercept should be adjusted or not

    :type alpha: float
    :type max_iter: int
    :type fit_intercept: bool


    Methods
    -----------
    normalize(X)
        :return Applies standard scaling to the input feature matrix

    _soft_thresholding_operator(X)
        :return Private method that applies soft thresholding to the input feature matrix

    fit(X,y)
        :return Performs coordinate descent for lasso on the input feature matrix and ground truth labels

    Attributes
        :param coef_: Vector of size(n_features,) containing the coefficients corresponding to each feature
        :param intercept_: A floating point number that represents the intercept of the fitted model

    """
    def __init__(self, alpha=1.0, max_iter=1000, fit_intercept=False):
        """
        Constructor

        :param alpha: alpha value
        :param max_iter: maximum number of iterations
        :param fit_intercept: boolean indicating if the intercept should be adjusted or not

        :type alpha: float
        :type max_iter: int
        :type fit_intercept: bool

        """

        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None


    def normalize(self, X):
        """
        Normalizes the input feature matrix

        :param X: Input feature matrix
        :type X: np.ndarray
        :return: Normalized feature matrix
        :rtype: np.ndarray
        """
        X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
        return X

    def _soft_thresholding_operator(self, x, lambda_):
        """
        Applied soft thresholding operation on the input feature matrix
        :param x: Input feature matrix
        :param lambda_: scaling parameter (alpha * x.shape[0])
        :return: Scaled input feature matrix
        """
        if x > 0.0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0.0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0.0

    def fit(self, X, y):
        """
        Performs coordinate descent for lasso on the input feature matrix and ground truth labels
        :param X: Input feature matrix
        :type X: np.ndarray
        :param y: Vector of ground truth labels
        :type y: np.ndarray
        :return: None
        """

        X = self.normalize(X)

        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        beta = np.zeros(X.shape[1])
        if self.fit_intercept:
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        for iteration in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                tmp_beta = beta.copy()
                tmp_beta[j] = 0.0
                r_j = y - np.dot(X, tmp_beta)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.alpha * X.shape[0]

                beta[j] = self._soft_thresholding_operator(arg1, arg2) / (X[:, j] ** 2).sum()

                if self.fit_intercept:
                    beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self

    # def predict(self, X):
    #     y = np.dot(X, self.coef_)
    #     if self.fit_intercept:
    #         y += self.intercept_ * np.ones(len(y))
    #     return y
    #



