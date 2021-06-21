import os
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from feature_selection.pie_ss import Coordinate_Descent_Lasso
import matplotlib.pyplot as plt

# os.chdir(os.path.pardir)
print(os.getcwd())

data = pd.read_csv(os.path.join(os.getcwd(), "data", "data.csv"))


def get_x_and_y(data, normalize=True):
    """
    Returns the input and output arrays split from the input dataframe

    :param data: Dataframe with embeddings as feature values for each feature
    :param normalize: If normalize is true, standard scaling is applied to data
    :return: Data split into input and output arrays
    :type data: pandas.core.frame.DataFrame
    :type normalize: bool
    :rtype: np.ndarray, np.ndarray
    """
    if normalize == True:
        X = StandardScaler().fit_transform(data.iloc[:, :-1].values)
    else:
        X = data.iloc[:, :-1].values

    y = data.iloc[:, -1].values
    return X, y

def get_selected_columns_custom(X,y,alpha=0.01, max_iter = 1000, fit_intercept = False):
    """
    Returns selected columns from the dataframe. (Custom Coordinate descent applied for performing lasso)
    :param X: Input feature matrix
    :param y:  Vector consisting of ground truth labels
    :param alpha: Alpha value
    :param max_iter: Maximum number of iterations
    :param fit_intercept: Boolean indicating if the intercept should be scaled or not
    :type X: np.ndarray
    :type y: np.ndarray
    :type alpha: float
    :type max_iter: int
    :type fit_intercept: bool
    :return: Lasso coefficients and selected columns
    :rtype: np.ndarray, np.ndarray
    """
    model = Coordinate_Descent_Lasso(alpha=alpha,max_iter=max_iter,fit_intercept=fit_intercept)
    model.fit(X, y)
    coefs = model.coef_
    importance = np.abs(coefs)
    return coefs, np.array(data.columns[:-1])[importance > 0]


def get_selected_columns_sklearn(X, y, alpha=0.01, max_iter=1000, fit_intercept=False):
    """
    Returns selected columns from the dataframe. (Scikit-Learn implementation of Coordinate descent used for performing lasso)
    :param X: Input feature matrix
    :param y:  Vector consisting of ground truth labels
    :param alpha: Alpha value
    :param max_iter: Maximum number of iterations
    :param fit_intercept: Boolean indicating if the intercept should be scaled or not
    :type X: np.ndarray
    :type y: np.ndarray
    :type alpha: float
    :type max_iter: int
    :type fit_intercept: bool
    :return: Lasso coefficients and selected columns
    :rtype: np.ndarray, np.ndarray
    """
    lasso = Lasso(alpha=alpha, fit_intercept=fit_intercept)
    lasso.fit(X, y)
    coefs = lasso.coef_
    importance = np.abs(coefs)
    return coefs, np.array(data.columns[:-1])[importance > 0]

def compare_custom_sklearn(data, custom_coefs,sklearn_coefs):
    """
    Compares the coefficients of custom and scikit learn's implementation of Coordinate descent on lasso
    :param data: Dataframe with embeddings as feature values for each feature
    :param custom_coefs: Lasso coefficients obtained using custom implementation of Coordinate descent on lasso
    :param sklearn_coefs: Lasso coefficients obtained using scikit-learn's implementation of Coordinate descent on lasso
    :type data: pandas.core.frame.DataFrame
    :type custom_coefs: np.ndarray
    :type sklearn_coefs: np.ndarray
    """

    plt.figure(figsize = (10,6))
    cols = data.columns[:-1]
    plt.plot(cols,sklearn_coefs, label="Scikit Learn")
    plt.plot(cols,custom_coefs+0.01, label="Custom")
    plt.xticks(rotation = 60,fontweight='bold');
    plt.yticks(fontweight='bold')
    plt.title("Lasso Coefficients for Custom and Sklearn Implementation of Coordinate Descent", fontweight='bold')


def plot_coefficient_path(data):
    """
    Plots the coefficient path for both the custom and scikit-learns implementation of coordinate descent on lasso
    :param data: Dataframe with embeddings as feature values for each feature
    :type data: pandas.core.frame.DataFrame
    """
    X, y = get_x_and_y(data)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    n_alphas = 10
    alphas = np.logspace(-10, 2, n_alphas)
    coefs_scikit_learn = []
    coefs_custom = []

    for a in alphas:
        lasso_sklearn = Lasso(alpha=a, fit_intercept=False)
        lasso_sklearn.fit(X, y)
        coefs_scikit_learn.append(lasso_sklearn.coef_)

    for a in alphas:
        lasso_custom = Coordinate_Descent_Lasso(alpha=a, fit_intercept=False)
        lasso_custom.fit(X, y)
        coefs_custom.append(lasso_custom.coef_)

    ax[0].plot(alphas, coefs_scikit_learn)
    ax[0].set_title('Lasso coefficient path (Sklearn)', fontweight='bold', fontsize=10)

    ax[1].plot(alphas, coefs_custom)
    ax[1].set_title('Lasso coefficient path (Custom)', fontweight='bold', fontsize=10)

    for axes in ax:
        axes.set_xscale('log')
        axes.set_xlabel('Alpha', fontweight='bold', fontsize=8)
        axes.set_ylabel('Coefficients', fontweight='bold', fontsize=8)

    plt.axis('tight')
    plt.legend(data.columns[:-1], bbox_to_anchor=(1.05, 1.05))
