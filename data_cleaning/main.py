
import pandas as pd
import os
from data_cleaning.utils import report, drop_nan_cols, Transformations, save_table
from settings import ROOT_DIR
os.chdir(ROOT_DIR)

"""
Uncomment the code below to extract descriptive features from the flare and non-flare samples
"""
# from data_cleaning.helper_functions import Sample, process_partition
# path = os.getcwd()
# sample = Sample("FL", "M1.0@265_Primary_ar115_s2010-08-06T06_36_00_e2010-08-06T18_24_00.csv")
# data = sample.get_data()
# extracted_data = process_partition(os.path.join(path,"data\\partition1"),data)
# save_path = os.path.join(os.getcwd(),"data","extracted_features.csv")
# extracted_data.to_csv(save_path,index=False,header=True)

"""
Load extracted features from parition 1
"""
extracted_features = pd.read_csv(os.path.join(os.getcwd(),"data","extracted_features.csv"))


"""
Generate Summary table using extracted features
"""
summary_table = report(extracted_features)
drop_nan_cols(0.05, summary_table, extracted_features)

"""
Perform transformation on the extracted features based on summary table stats
"""
t = Transformations(summary_table, extracted_features)
small_range = t.get_features_with_small_outliers()
large_range = t.get_features_with_large_outliers()
t.log_transformation(large_range)
t.z_score_normalisation(small_range)

"""
Save cleaned tables
"""
save_path_1 = os.path.join(os.getcwd(),"data","extracted_features_cleaned.csv")
save_path_2 = os.path.join(os.getcwd(),"data","summary_table.csv")
save_table(save_path_1, extracted_features,"extracted_features")
save_table(save_path_2, summary_table,"summary_table")
