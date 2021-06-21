from data_cleaning.helper_functions import Sample, save_table
from tslearn.datasets import CachedDatasets
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dtaidistance import dtw, clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from sklearn.manifold import spectral_embedding
from sklearn.metrics import normalized_mutual_info_score,adjusted_mutual_info_score,mutual_info_score
import scipy
from sklearn.cluster import KMeans
from scipy.stats import entropy

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

    def get_embed_matrix(self, timeseries):
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
        adj = A.todense()
        np.fill_diagonal(adj, 0)
        A = scipy.sparse.csr_matrix(adj)
        G = nx.Graph(A)
        adj = nx.adjacency_matrix(G)
        embed = spectral_embedding(adj, 1)
        return embed.flatten()

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
            embed = self.get_embed_matrix(timeseries)

            kmeans = KMeans(n_clusters=5, random_state=0).fit(embed.reshape(-1, 1))
            embed_y = kmeans.labels_.flatten()
            y = np.array(y).flatten()
            scores[column_mapping[col]] = self.relevance_score(embed_y, y)
            timeseries = []
            y = []
        scores_data = pd.DataFrame.from_dict(scores, orient='index', columns=['Relevance Score']).sort_values(
            by='Relevance Score', ascending=False)
        return scores_data
