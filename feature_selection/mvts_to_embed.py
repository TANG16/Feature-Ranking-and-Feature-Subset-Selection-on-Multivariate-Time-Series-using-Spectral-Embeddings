from feature_selection.pie_rank import PIE_RANK
from data_cleaning.helper_functions import Sample
import pandas as pd
import numpy as np
from tqdm import tqdm

# os.chdir(os.path.pardir)
# print(os.getcwd())

def mvts_to_df_embed(path_FL, path_NF):
    """
    Converts the multivariate time series data into a single dataframe with every feature value represented as embedding.
    Every feature in a file is represented by embedding

    :param path_FL: Directory pointing to all the flare files
    :param path_NF:  Directory pointing to all the non-flare files
    :type path_FL: str
    :type path_NF: str
    :return: Dataframe with every feature value represented as embedding (n_files * n_features)
    :rtype: pandas.core.frame.DataFrame
    """
    pie_rank = PIE_RANK(path_FL,path_NF)
    files_flare = pie_rank.generate_flare_set()
    files_non_flare = pie_rank.generate_non_flare_set()
    mapping = pie_rank.mapping
    timeseries = []
    y = []
    labels = []
    subset_data = pd.DataFrame(columns = Sample("FL",files_flare[0]).get_data().columns[1:25])
    subset_data['FLARE_CLASS'] = np.nan
    for col in tqdm(range(1,25)):
        for file in tqdm(files_flare):
            s = Sample("FL",file).get_data().iloc[:,col].values


            timeseries.append(s)
            y.append(mapping[file[0]])

        for file in tqdm(files_non_flare):
            s = Sample("NF",file).get_data().iloc[:,col].values
            y.append(mapping[file[0]])

            timeseries.append(s)
        labels.append(y)
        embed = pie_rank.get_embed_matrix(timeseries)
        subset_data.iloc[:,col-1] = embed
        timeseries = []
        y = []
    subset_data.iloc[:,-1] = np.array(labels[0])
    return subset_data