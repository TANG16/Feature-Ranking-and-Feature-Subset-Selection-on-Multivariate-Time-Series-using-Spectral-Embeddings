from data_cleaning.helper_functions import Sample, calculate_descriptive_features,get_features,report, drop_nan_cols, Transformations

import os
import pandas as pd

path = os.getcwd()

sample = Sample("FL", "M1.0@265_Primary_ar115_s2010-08-06T06_36_00_e2010-08-06T18_24_00.csv")
print("File stats")
print("----------")
print(sample.get_flare_class())
print(sample.get_start_time())
print(sample.get_start_time())
print(sample.get_end_time())
print(sample.get_number_of_files())

print("------------")
print("Data")
data = sample.get_data()
print(data.head(5))


print("--------------")
print("Descriptive Features")
print(calculate_descriptive_features(data))


print("Features")
print("-----------------")
print(get_features(data))


print("Extraced features")
print("-----------------")
extracted_features = pd.read_csv(os.path.join(os.getcwd(),"data","extracted_features.csv")).sample(50)


print("\n")
print("Before Transformations")
print("----------------------")
print(extracted_features.head(5))
print(extracted_features['FLARE_TYPE'].value_counts(normalize=True)*100)

summary_table = report(extracted_features)
drop_nan_cols(0.05, summary_table, extracted_features)

t = Transformations(summary_table, extracted_features)
small_range = t.get_features_with_small_outliers()
large_range = t.get_features_with_large_outliers()
t.log_transformation(large_range)
t.z_score_normalisation(small_range)

print("\n")
print("After Transformations")
print("----------------------")
print(extracted_features.head(5))