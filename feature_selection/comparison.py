import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.linear_model import Lasso, LogisticRegression

from settings import ROOT_DIR
os.chdir(ROOT_DIR)
# print(os.getcwd())


class Feature_Selection_Comparison:
    """
    Comparison of various feature selection techniques using the data generated using Spectral Embeddings and Vectorized data generated from the RAW MTS segments

    Parameters
    -----------
    :param X_vect: Input feature matrix of vectorized data.
    :param y_vect: Output vector of vectorized data.
    :param X_embed: Input feature matrix of embed data.
    :param y_embed: Output vector of embed data.

    Methods
    ---------
    __score_dataframe(scores,reverse):
        :return sorted score dataframe generated from scores array

    chi_2:
        :return Applies chi-2 for feature selection

    mutual_info:
        :return Applied mutual information for feature selection.

    rfe:
        :return Performs recursive feature elimination using base learner passed in params.

    random_forest_feature_importance:
        :return Applies Random Forest Classifier to obtain feature importance.

    Lasso:
        :return Applies Lasso Regression for performing feature selection
    """

    def __init__(self, X_vect, y_vect, X_embed, y_embed):
        """
        Constructor
        :param X_vect: Input feature matrix of vectorized data.
        :param y_vect: Output vector of vectorized data.
        :param X_embed: Input feature matrix of embed data.
        :param y_embed: Output vector of embed data.
        :type X_vect: pandas.core.frame.DataFrame
        :type y_vect: pandas.core.series.Series
        :type X_embed: pandas.core.frame.DataFrame
        :type y_embed: pandas.core.series.Series
        """
        self.X_vect = X_vect
        self.y_vect = y_vect
        self.X_embed = X_embed
        self.y_embed = y_embed

    def __score_dataframe(self, scores, reverse=True):
        """
        Returns sorted score dataframe generated from scores array
        :param scores: List containing the scores of the performed test
        :param reverse: Boolean indication if the array should be sorted in descending order or ascending order
        :type scores: list
        :type reverse: bool
        :return: DataFrames with feature names and corresponding scores.
        :rtype: pandas.core.frame.DataFrame
        """
        return pd.DataFrame(sorted(scores, key=lambda x: x[1], reverse=reverse),
                            columns=['Feature', 'Score'])

    def chi_2(self, num_features_vectorized=24, num_features_embed=24):

        """
        Applies chi-2 for feature selection
        :param num_features_vectorized: Maximum number of features to be selected from vectorized data.
        :param num_features_embed: Maximum number of features to be selected from embed data.
        :type num_features_vectorized: int
        :type num_features_embed: int
        :return: DataFrames with feature names and corresponding scores.
        :rtype: pandas.core.frame.DataFrame
        """
        test = SelectKBest(score_func=chi2)
        test.fit(self.X_vect, self.y_vect)
        scores_chi_vect = []
        for i in range(num_features_vectorized):
            score = test.scores_[i]
            scores_chi_vect.append((self.X_vect.columns[i], score))
        scores_chi_vect_df = self.__score_dataframe(scores_chi_vect)

        test = SelectKBest(score_func=chi2)
        test.fit(self.X_embed, self.y_embed)
        scores_chi_embed = []
        for i in range(num_features_embed):
            score = test.scores_[i]
            scores_chi_embed.append((self.X_embed.columns[i], score))
        scores_chi_embed_df = self.__score_dataframe(scores_chi_embed)

        return scores_chi_vect_df, scores_chi_embed_df

    def mutual_info(self, num_features_vectorized=24, num_features_embed=24):
        """
        Applied mutual information for feature selection.
        :param num_features_vectorized: Maximum number of features to be selected from vectorized data.
        :param num_features_embed: Maximum number of features to be selected from embed data.
        :type num_features_vectorized: int
        :type num_features_embed: int
        :return: DataFrames with feature names and corresponding scores.
        :rtype: pandas.core.frame.DataFrame
        """
        test = SelectKBest(score_func=mutual_info_classif)
        test.fit(self.X_vect, self.y_vect)
        scores_info_vect = []
        for i in range(num_features_vectorized):
            score = test.scores_[i]
            scores_info_vect.append((self.X_vect.columns[i], score))
        scores_info_vect_df = self.__score_dataframe(scores_info_vect)


        test = SelectKBest(score_func=mutual_info_classif)
        test.fit(self.X_embed, self.y_embed)
        scores_info_embed = []
        for i in range(num_features_embed):
            score = test.scores_[i]
            scores_info_embed.append((self.X_embed.columns[i], score))
        scores_info_embed_df = self.__score_dataframe(scores_info_embed)

        return scores_info_vect_df, scores_info_embed_df

    def rfe(self,base_learner, num_features_vectorized=24, num_features_embed=24):
        """
        Performs recursive feature elimination using base learner passed in params.
        :param base_learner: Base Learner to be used for Recursive Feature Elimination
        :param num_features_vectorized: Maximum number of features to be selected from vectorized data.
        :param num_features_embed: Maximum number of features to be selected from embed data.
        :type num_features_vectorized: int
        :type num_features_embed: int
        :return: DataFrames with feature names and corresponding scores.
        :rtype: pandas.core.frame.DataFrame
        """
        rfe = RFE(base_learner)
        rfe.fit(self.X_vect, self.y_vect)
        scores_rfe_lr_vect = []
        for i in range(num_features_vectorized):
            scores_rfe_lr_vect.append((self.X_vect.columns[i], rfe.ranking_[i]))

        scores_rfe_lr_vect_df = self.__score_dataframe(scores_rfe_lr_vect, reverse=False)

        rfe = RFE(base_learner)
        rfe.fit(self.X_embed, self.y_embed)
        scores_rfe_lr_embed = []
        for i in range(num_features_embed):
            scores_rfe_lr_embed.append((self.X_embed.columns[i], rfe.ranking_[i]))

        scores_rfe_lr_embed_df = self.__score_dataframe(scores_rfe_lr_embed, reverse=False)

        return scores_rfe_lr_vect_df, scores_rfe_lr_embed_df

    def random_forest_feature_importance(self, clf, num_features_vectorized=24, num_features_embed=24):
        """

        :param clf: Random Forest Classifier to be used for obtaining feature importance
        :param num_features_vectorized: Maximum number of features to be selected from vectorized data.
        :param num_features_embed: Maximum number of features to be selected from embed data.
        :type num_features_vectorized: int
        :type num_features_embed: int
        :return: DataFrames with feature names and corresponding ranks.
        :rtype: pandas.core.frame.DataFrame
        """
        clf.fit(self.X_vect, self.y_vect)
        scores_rf_vect = []
        for i in range(num_features_vectorized):
            scores_rf_vect.append((self.X_vect.columns[i], clf.feature_importances_[i]))

        scores_rf_vect_df = self.__score_dataframe(scores_rf_vect)

        clf.fit(self.X_embed, self.y_embed)
        scores_rf_embed = []
        for i in range(num_features_embed):
            scores_rf_embed.append((self.X_embed.columns[i], clf.feature_importances_[i]))

        scores_rf_embed_df = self.__score_dataframe(scores_rf_embed)

        return scores_rf_vect_df, scores_rf_embed_df

    def Lasso(self, alpha=0.01):
        """
        Applies Lasso Regression for performing feature selection
        :param alpha: regularization parameter
        :type alpha: float
        :return: DataFrames with feature names and corresponding coefficients.
        :rtype: pandas.core.frame.DataFrame
        """
        clf = Lasso(alpha=alpha)
        clf.fit(self.X_embed, self.y_embed)
        coefs = clf.coef_
        scores_lasso = pd.DataFrame(data=self.X_embed.columns, columns=['Feature'])
        scores_lasso['Score'] = np.abs(coefs)
        selected_lasso_embed = scores_lasso[scores_lasso['Score'] > 0].reset_index(drop=True)

        clf = Lasso(alpha=0.01)
        clf.fit(self.X_vect, self.y_vect)
        coefs = clf.coef_
        scores_lasso = pd.DataFrame(data=self.X_vect.columns, columns=['Feature'])
        scores_lasso['Score'] = np.abs(coefs)
        selected_lasso_vect = scores_lasso[scores_lasso['Score'] > 0].reset_index(drop=True)

        return selected_lasso_vect, selected_lasso_embed
