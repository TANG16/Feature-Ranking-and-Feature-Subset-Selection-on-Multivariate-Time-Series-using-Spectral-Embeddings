from data_cleaning.helper_functions import Sample

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dtaidistance import dtw
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph

from graspologic.embed import AdjacencySpectralEmbed
from sklearn.cluster import SpectralClustering
from sklearn.manifold import  SpectralEmbedding
from sklearn.metrics import normalized_mutual_info_score,adjusted_mutual_info_score,mutual_info_score
import scipy
from sklearn.cluster import KMeans
from scipy.stats import entropy
import scipy
from scipy.sparse import csgraph
from numpy import linalg as LA

# os.chdir(os.path.pardir)
# print(os.getcwd())



class PIE_RANK:

    """
    Implementation of Ranking algorithm as proposed in the paper

    Parameters
    ----------
    :param flare_path: Directory pointing to all the flare files
    :param non_flare_path: Directory pointing to all the non-flare files
    :type flare_path: str
    :type non_flare_path: str

    Methods
    ---------
    __get_column_mapping():
        :return mapping from column to index

    generate_flare_set():
        :return files from flare directory

    generate_non_flare_set():
        :return 400 samples from each non-flare class

    get_embed_matrix():
        :return embedded matrix of given feature

    relevance_score():
        :return compares ground truth labels and embed labels as proposed in the paper

    get_score():
        :return dataframe with score for each feature
    """


    def __init__(self, flare_path, non_flare_path):
        """
        Constructor
        :param mapping: label encoding of flare classes

        :param flare_path: Directory pointing to all the flare files
        :param non_flare_path: Directory pointing to all the non-flare files
        :type flare_path: str
        :type non_flare_path: str
        """
        self.flare_path = flare_path
        self.non_flare_path = non_flare_path
        self.mapping = {
            "B": 0,
            "C": 1,
            "F": 2,
            "M": 3,
            "X": 4
        }

    def __get_column_mapping(self):

        """
        Private method that maps the columns in the file to an integer index
        :returns Mapping from column to index
        :rtype dict
        """

        s1 = list(Sample("FL", "M1.0@265_Primary_ar115_s2010-08-06T06_36_00_e2010-08-06T18_24_00.csv").get_data().columns)[:25]
        column_mapping = {}
        for i in range(len(s1)):
            column_mapping[i] = s1[i]

        return column_mapping

    def generate_flare_set(self):

        """
        Returns all files from flare directory

        :returns All files from flare directory
        :rtype np.ndarray
        """

        #         x_count = 10
        #         m_count = 10
        files = []
        for file in os.listdir(self.flare_path):
            files.append(file)
        #             if file[0] == 'M' and m_count >0:
        #                 files.append(file)
        #                 m_count-=1
        #             elif file[0] == "X" and x_count >0:
        #                 files.append(file)
        #                 x_count-=1

        return files

    def generate_non_flare_set(self):

        """
        Returns all files from non-flare directory

        :returns All files from non-flare directory
        :rtype np.ndarray
        """
        b_count = 400
        c_count = 400
        f_count = 400
        files = []
        for file in os.listdir(self.non_flare_path):

            if file[0] == 'B' and b_count > 0:
                files.append(file)
                b_count -= 1
            elif file[0] == "C" and c_count > 0:
                files.append(file)
                c_count -= 1

            elif file[0] == "F" and f_count > 0:
                files.append(file)
                f_count -= 1

        return files

    def eigenDecomposition(self, A, plot=True, topK=5):
        """
        :param A: Affinity matrix
        :param plot: plots the sorted eigen values for visual inspection
        :return A tuple containing:
        - the optimal number of clusters by eigengap heuristic
        - all eigen values
        - all eigen vectors

        This method performs the eigen decomposition on a given affinity matrix,
        following the steps recommended in the paper:
        1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
        2. Find the eigenvalues and their associated eigen vectors
        3. Identify the maximum gap which corresponds to the number of clusters
        by eigengap heuristic

        References:
        https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
        """
        L = csgraph.laplacian(A, normed=True)
        n_components = A.shape[0]

        # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
        # the euclidean norm of complex numbers.
        #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
        eigenvalues, eigenvectors = LA.eig(L)

        if plot:
            plt.title('Largest eigen values of input matrix')
            plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
            plt.grid()

        # Identify the optimal number of clusters as the index corresponding
        # to the larger gap between eigen values

        index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
        nb_clusters = index_largest_gap + 1

        return nb_clusters[0], eigenvalues, eigenvectors

    def get_embed_vector(self, timeseries):
        """
        Step 1: Calculate DTW distance for a feature across all the timesteps
        Step 2: Generate K-Nearest Neighbour graph of the distance matrix
        Step 3: Convert the directed graph to acyclic graph by multiplying the edge weights
        Step 4: Fill all diagonal elements with "1"
        Step 5: Generate adjacency matrix for the distance graph
        Step 6: Generate embedding matrix using the adjacency matrix

        :param timeseries: Values of a feature across flare and non-flare samples
        :type timeseries: list
        :return: Embedded matrix of the feature
        :rtype: np.ndarray
        """
        ds = dtw.distance_matrix_fast(timeseries)
        ds = np.nan_to_num(ds)
        A = kneighbors_graph(ds, 5, mode='connectivity', include_self=True)
        A = 0.5 * (A + A.T)
        adj = A.toarray()
        ase = AdjacencySpectralEmbed(n_components=1)
        embed = ase.fit_transform(adj)
        # k,_,_ = self.eigenDecomposition(adj, plot=False)
        # cluster = SpectralClustering(n_clusters=k).fit(adj)
        # normzalized_adj = cluster.affinity_matrix_
        # embedding = SpectralEmbedding(n_components=1, affinity = 'precomputed')
        # embed = embedding.fit_transform(normzalized_adj)
        return embed



    def relevance_score(self, embed, y):
        """
        Calculates the relevance score using the formula specified in the paper

        :param embed: Embedding matrix of a feature
        :param y: Label encoded ground truth labels
        :return: Score calculated based on the formula in the paper
        :rtype: np.float64
        """
        mututal_information = mutual_info_score(y, embed)
        entropy_embed = entropy(embed)
        entropy_y = entropy(y)
        relevance_score = mututal_information / (np.sqrt(entropy_embed * entropy_y))
        return relevance_score

    def get_score(self):
        """
        Returns a dataframe with all the sensors as index and their respective relevance score as column

        :return: Dataframe with Column name as index and relevance score as column
        :rtype: pandas.core.frame.DataFrame
        """
        files_flare = self.generate_flare_set()
        files_non_flare = self.generate_non_flare_set()
        timeseries = []
        y = []
        scores = {}
        column_mapping = self.__get_column_mapping()
        for col in tqdm(range(1, 25)):
            for file in files_flare:
                s = Sample("FL", file).get_data().iloc[:, col].values
                y.append(self.mapping[file[0]])
                timeseries.append(s)

            for file in files_non_flare:
                s = Sample("NF", file).get_data().iloc[:, col].values
                y.append(self.mapping[file[0]])
                timeseries.append(s)
            embed = self.get_embed_vector(timeseries)


            embed_y = KMeans(n_clusters=5).fit_predict(embed)
            y = np.array(y).flatten()
            scores[column_mapping[col]] = self.relevance_score(embed_y, y)
            timeseries = []
            y = []
        scores_data = pd.DataFrame.from_dict(scores, orient='index', columns=['Relevance Score']).sort_values(
            by='Relevance Score', ascending=False)
        return scores_data
