import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

plt.style.use('ggplot')
plt.style.use('ggplot')
os.chdir('..')
path = os.getcwd()


class Sample:
    """
    A class to parse files and display basic information of the file.

    Attributes
    ----------
    :param flare_type: "FL" or "NF"
    :type flare_type str

    :param file_name: name of the file to extract information about
    :type file_name str


    Methods
    -------
    get_flare_class():
        :return Flare class

    get_start_time():
        :return start time of multivariate time series

    get_end_time():
        :return end time of multivariate time series

    get_number_of_files():
        :return number of files in the current working directory

    get_data():
        :return mvts in the form of pandas dataframe
    """

    def __init__(self, flare_type, file_name):
        """
        Constructor

        :param flare_type: "FL" or "NF"
        :type flare_type str

        :param file_name: name of the file to extract information about
        :type file_name str
        """

        self.flare_type = flare_type
        self.file_name = file_name
        self.path = os.path.join(os.getcwd(), 'data\partition1')

    def get_flare_class(self):
        """
        Return the flare class associated with the file name
        :returns: flare class
         :rtype: str
        """
        return self.file_name[0]

    def __get_start_end_index(self):
        """
        Private method that returns the starting and the ending index of start time and end time.
        :returns: start_time, end_time
        """
        start_index = self.file_name.find("_s")
        end_index = self.file_name.find("_e")
        return start_index, end_index

    def get_start_time(self):
        """
        Return the starting time of the multivariate time series
        :returns: starting time of the mvts
         :rtype: int
        """
        s, e = self.__get_start_end_index()
        return self.file_name[s: e]

    def get_end_time(self):
        """
        Return the ending time of the multivariate time series
        :returns: starting time of the mvts
        :rtype: int
        """
        _, e = self.__get_start_end_index()
        return self.file_name[e:e - 4]

    def get_number_of_files(self):
        """:returns number of files in the working directory
           :rtype: int
            """
        return len(os.listdir(os.path.join(self.path, self.flare_type)))

    def get_data(self):
        """

        Returns mvts in the form of pandas dataframe

        :returns mvts in pandas dataframe format
        :rtype: pandas.core.frame.DataFrame

        """
        try:
            return pd.read_csv(os.path.join(self.path, self.flare_type, self.file_name), sep="\t")

        except FileNotFoundError:
            print("File not found")


def get_features(data):
    """
    Adds min,mean, median,max and stddev as suffix to features from the function parameter

    :param data: Dataframe to be used to add features
    :returns: Returns list of features with newly added columns
    """
    features = []
    cols = np.array(data.columns[1:25])
    for col in cols:
        features.append(f'{col}_MIN')
        features.append(f'{col}_MEAN')
        features.append(f'{col}_MEDIAN')
        features.append(f'{col}_MAX')
        features.append(f'{col}_STDDEV')

    return features


def calculate_descriptive_features(data):
    """
    Calculates descriptive statistics on the features across all samples from both FL and NF
    :param data: Dataframe on which descriptive statistics are computed
    :returns Returns the computed statistics
    """
    variates_to_calc_on = list(data.columns[1:25])

    features_to_return = get_features(data)
    features = pd.DataFrame(columns=features_to_return)
    mins = [np.min(data.loc[:, col]) for col in variates_to_calc_on]
    means = [np.mean(data.loc[:, col]) for col in variates_to_calc_on]
    medians = [np.median(data.loc[:, col]) for col in variates_to_calc_on]
    maxs = [np.max(data.loc[:, col]) for col in variates_to_calc_on]
    std = [np.std(data.loc[:, col]) for col in variates_to_calc_on]
    measures = (mins, means, medians, maxs, std)
    features.loc[len(features)] = np.concatenate(measures)

    return features


def process_flare_data(partition_location, final_data_fl):
    """
    Calculates descriptive statistics across all the Flare samples

    :param partition_location: Location of FL samples
    :param final_data_fl: Empty dataframe with all the columns generated using generate_features()

    :returns: Dataframe with descriptive statistics across all the Flare samples
    """
    path_FL = os.path.join(partition_location, "FL")
    column_index = final_data_fl.columns.get_loc('FLARE_TYPE')
    i = 0
    print("Processing Flare files")

    for file in tqdm(os.listdir(path_FL)[:10]):
        sample = Sample("FL", file)
        data = sample.get_data()
        features = calculate_descriptive_features(data)
        final_data_fl = final_data_fl.append(features)
        final_data_fl.iat[i, column_index] = sample.get_flare_class()
        i += 1

    return final_data_fl.reset_index(drop=True)


def process_non_flare_data(partition_location, final_data_nf):
    """
    Calculates descriptive statistics across all the Non-Flare samples

    :param partition_location: Location of NF samples
    :param final_data_nf: Empty dataframe with all the columns generated using generate_features()

    :returns: Dataframe with descriptive statistics across all the Non-Flare samples
    """
    column_index = final_data_nf.columns.get_loc('FLARE_TYPE')
    i = 0
    path_NF = os.path.join(partition_location, "NF")
    print('Processing Non-Flare files')

    for file in tqdm(os.listdir(path_NF)[:10]):
        sample = Sample("NF", file)
        data = sample.get_data()
        features = calculate_descriptive_features(data)
        final_data_nf = final_data_nf.append(features)
        final_data_nf.iat[i, column_index] = sample.get_flare_class()
        i += 1

    return final_data_nf.reset_index(drop=True)


def process_partition(partition_location, data):
    """

    Concatenates descriptive statistics computed on flare and non-flare samples
    :param partition_location: Location of NF samples
    :param data: Empty dataframe with all the columns generated using generate_features()
    """
    abt_header = ['FLARE_TYPE'] + get_features(data)

    final_data_fl = pd.DataFrame(columns=abt_header)
    final_data_nf = pd.DataFrame(columns=abt_header)

    flare_data = process_flare_data(partition_location, final_data_fl)
    non_flare_data = process_non_flare_data(partition_location, final_data_nf)

    table = pd.concat([flare_data, non_flare_data], axis=0, ignore_index=True)

    # Saving dataframe as csv with name extracted from abt_name
    #     table.to_csv(path,index=False,header=True)

    return table


def save_table(path, table, index=False, header=True):
    """
    Stores table to the specified path
    :param index: Write row names if true, false otherwise
    :param header: Write column name if true, false otherwise
    :param table: Dataframe to be saved
    :param path: Location where the dataframe is to be stored
    :type table: pandas.core.frame.DataFrame
    :type index: bool
    :type header: bool
    :returns: None
    """
    table.to_csv(path, index=index, header=header)


def column_stat(feature_name, data):
    """
    Calculates descriptive stats on the feature passed in the function

    :param feature_name: Feature on which the stats are to be computed
    :param data: Dataframe of which the feature is a part of

    :type feature_name: str
    :type data: pandas.core.frame.DataFrame
    :returns: list of stats computed on feature passed as the parameter
    """
    summary_feature_names = ['Feature Name', 'Cardinality', 'Non-null Count', 'Null Count', 'Min', '25th', 'Mean',
                             '50th', '75th', 'Max', 'Outlier Count Low', 'Outlier Count High']

    feature_name = feature_name
    descriptive_stats = data[feature_name].describe()
    cardinality = data[feature_name].nunique()
    non_null_values = descriptive_stats['count']
    null_count = data.shape[0] - non_null_values
    min_value = descriptive_stats['min']
    percentile_25 = descriptive_stats['25%']
    mean = descriptive_stats['mean']
    percentile_50 = descriptive_stats['50%']
    percentile_75 = descriptive_stats['75%']
    max_value = descriptive_stats['max']
    std = descriptive_stats['std']
    iqr = percentile_75 - percentile_25
    lower_thershold = percentile_25 - (1.5 * iqr)
    upper_threshold = percentile_75 + (1.5 * iqr)
    lower_outlier_count = data[data[feature_name] < lower_thershold].shape[0]
    upper_outlier_count = data[data[feature_name] > upper_threshold].shape[0]

    feature_values = [
        [feature_name, cardinality, non_null_values, null_count, min_value, percentile_25, mean, percentile_50,
         percentile_75, max_value, lower_outlier_count, upper_outlier_count]]
    features = pd.DataFrame(columns=summary_feature_names, data=feature_values)

    return features


def report(data):
    """
    :param data: Dataframe using which the descriptive stats are computed
    :type data: pandas.core.frame.DataFrame
    :returns: A new dataframe with stats computed for every columns in the dataframe passed as a parameter
    """
    excluded_columns = ['FLARE_TYPE']

    summary_feature_names = ['Feature Name', 'Cardinality', 'Non-null Count', 'Null Count', 'Min', '25th', 'Mean',
                             '50th', '75th', 'Max', 'Outlier Count Low', 'Outlier Count High']

    df = pd.DataFrame(columns=summary_feature_names)

    for feature in tqdm(data.columns):
        if feature not in excluded_columns:
            row = column_stat(feature, data)
            df = df.append(row)

    df = df.reset_index(drop=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

    return df


def drop_nan_cols(threshold, summary_table, data):
    cols = []
    for col in data.columns:
        if data[col].isna().sum() > threshold * data.shape[0]:
            cols.append(col)
    summary_table.drop(summary_table[summary_table['Feature Name'].isin(cols)].index, inplace=True)
    data.drop(cols, axis=1, inplace=True)
    print(
        f'{len(cols)} feature dropped from summary table and the extracted features table on using {threshold * 100} % as the threshold')


class Transformations:
    """
    Class to perform basic transformation on the extracted features based on statistics computed in summary table

    Attributes
    ----------
    :param summary_table: Dataframe constituting the descriptive stats computed for every column in the extract features dataframe.
    :param data: Extracted feature dataframe

    :type summary_table: pandas.core.frame.DataFrame
    :type data: pandas.core.frame.DataFrame



    """

    def __init__(self, summary_table, data):
        """
        Constructor

        :param summary_table: Dataframe constituting the descriptive stats computed for every column in the extract features dataframe.
        :param data: Extracted feature dataframe
        :type summary_table: pandas.core.frame.DataFrame
        :type data: pandas.core.frame.DataFrame
        """
        self.summary_table = summary_table
        self.data = data
        self.features_large_outliers =[]
        self.features_small_outliers = []

    def get_features_with_large_outliers(self):
        """
        Get features with large range(>100000) and more than 10% of feature values > 1.5* IQR

        :returns: Features with range > 100000 and more than 10% of feature values > 1.5* IQR
        """

        self.summary_table['Range'] = self.summary_table['Max'] - self.summary_table['Min']
        self.summary_table['is_outlier'] = self.summary_table['Outlier Count Low'] + \
                                           self.summary_table['Outlier Count High'] > self.summary_table[
                                               'Non-null Count'] * 0.1
        for feature in self.summary_table[(self.summary_table['is_outlier']) & (self.summary_table['Range'] > 10000)][
            'Feature Name']:
            self.features_large_outliers.append(feature)
        return self.features_large_outliers

    def get_features_with_small_outliers(self):
        """
          Get features with large range(<100000) and more than 5% of feature values < 1.5* IQR

          :returns: Features with range > 100000 and more than 5% of feature values < 1.5* IQR
          """

        self.summary_table['Range'] = self.summary_table['Max'] - self.summary_table['Min']
        self.summary_table['is_outlier_low'] = self.summary_table['Outlier Count Low'] + \
                                               self.summary_table['Outlier Count High'] < self.summary_table[
                                                   'Non-null Count'] * 0.05
        for feature in \
                self.summary_table[(self.summary_table['is_outlier_low']) & (self.summary_table['Range'] < 10000)][
                    'Feature Name']:
            self.features_small_outliers.append(feature)
        return self.features_small_outliers

    def clamp_to_third_quartile(self, cols):
        """
        Clamp high outlier features to third quartile =(1.5*IQR)

        :param cols: features on which the transformation is to be performed

        :return: Dataframe with transformation applied to features passed in function call
        """
        for col in cols:
            feature = self.data[col]
            upper = np.nanpercentile(feature, 75) + 1.5 * (
                        np.nanpercentile(feature, 75) - np.nanpercentile(feature, 25))
            self.data[col] = self.data[col].apply(lambda x: upper if x > upper else x)

    def log_transformation(self, cols):
        """
        Applies logarithm to the features passed in the function call

        :param cols: features on which the transformation is to be performed
        :return: Dataframe with transformation applied to features passed in function call
        """
        for col in cols:
            self.data[col] = np.log(self.data[col] + 0.01)

    def sqaure_root_transformation(self, cols):
        """
        Applies square root transformation to the features passed in the function call

        :param cols: features on which the transformation is to be performed
        :return: Dataframe with transformation applied to features passed in function call

        """
        for col in cols:
            self.data[col] = np.sqrt(self.data[col] + 0.01)

    def z_score_normalisation(self, cols):
        """
        Applies z-score normalisation  to the features passed in the function call

        :param cols: features on which the transformation is to be performed
        :return: Dataframe with transformation applied to features passed in function call

        """
        for col in cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            self.data[col] = (self.data[col] - mean) / std

    def plot_histogram(self, data, col, ax=None, bins=20):
        """

        :param data: Dataframe containing the feature to be plotted
        :param col: Feature to be plotted
        :param ax: Axis on which the histogram is plotted
        :param bins: Number of bins in the histrogtam
        :return: Histogram with arguments passed in the function call
        """
        sns.histplot(ax=ax, data=data, x=col, bins=bins);
