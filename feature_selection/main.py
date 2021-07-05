import pandas as pd
from feature_selection import pie_rank
from data_cleaning.utils import save_table
from feature_selection.mvts_to_embed import mvts_to_df_embed
from feature_selection.viz import get_x_and_y,get_selected_columns_custom,get_selected_columns_sklearn, compare_custom_sklearn,plot_coefficient_path
import os
from settings import ROOT_DIR
os.chdir(ROOT_DIR)



path_FL = os.path.join(os.getcwd(),"data\\partition1\\FL")
path_NF = os.path.join(os.getcwd(),"data\\partition1\\NF")

"""
PIE Ranking Algorithm
"""
pie_rank = pie_rank.PIE_RANK(path_FL,path_NF)

# Parse flare and non-flare files
files_flare = pie_rank.generate_flare_set()
files_non_flare = pie_rank.generate_non_flare_set()

# Calculate relevance scores based on PIE-Ranking
scores_data = pie_rank.get_score()
save_path = os.path.join(os.getcwd(),"data")
print(save_path)
save_table(save_path, pie_rank,"pie_rank_scores", index = True)

# Convert MVTS into embedding dataframe
data = mvts_to_df_embed(path_FL, path_NF)
save_table(save_path, data,"data", index = False, header=True)


"""
PIE Subset Selection
"""
# Calculate lasso coefficients using custom coordinate descent on lasso
X,y = get_x_and_y(data,normalize=False)
custom_coefs,selected_subset_custom = get_selected_columns_custom(X,y)
print(selected_subset_custom)

# Calculate lasso coefficients using scikit-learns implementation of coordinate descent on lasso
X,y = get_x_and_y(data,normalize=True)
sklearn_coefs,selected_subset_sklearn = get_selected_columns_sklearn(X,y)
print(selected_subset_sklearn)

# Compare both the implementations of coordinate descent on lasso
compare_custom_sklearn(data, custom_coefs,sklearn_coefs)

# Plot coefficient path for both the implementation of coordinate descent on lasso
plot_coefficient_path(data)

# Convert the coefficient values into a dataframe for visualisation
coefficients = pd.DataFrame(index = data.columns[:-1])
coefficients['Scikit-Learn'] = sklearn_coefs
coefficients['Custom'] = custom_coefs + 0.01
save_table(save_path, coefficients,"coefficients", index = True)


