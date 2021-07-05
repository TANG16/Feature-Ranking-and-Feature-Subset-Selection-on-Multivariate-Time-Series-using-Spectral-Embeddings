from data_cleaning.utils import save_table
from feature_selection.comparison import Feature_Selection_Comparison
from feature_selection.viz import plot_scores
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
plt.interactive(False)

from settings import ROOT_DIR
os.chdir(ROOT_DIR)



print(os.getcwd())
data_path = os.path.join(os.getcwd(),"pet_data")

embed = pd.read_csv(os.path.join(data_path,"embed.csv"))
vectorized_data = pd.read_csv(os.path.join(data_path,"vectorized.csv"))

X_vect = vectorized_data.drop("FLARE_CLASS",axis=1)
y_vect = vectorized_data.loc[:,'FLARE_CLASS']

X_embed = embed.drop("FLARE_CLASS",axis=1)
y_embed = embed.loc[:,'FLARE_CLASS']


ffs = Feature_Selection_Comparison(X_vect,y_vect,X_embed,y_embed)
"""
UNIVARIATE FEATUERE SELECTION
"""
# CHI-2
scores_chi_vect_df, scores_chi_embed_df = ffs.chi_2(num_features_vectorized=24,num_features_embed=24)
save_table(data_path, scores_chi_vect_df, "scores_chi_vect_df")
save_table(data_path, scores_chi_embed_df, "scores_chi_embed_df")
print("CHI-2 done")
plot_scores(scores_chi_vect_df,scores_chi_embed_df)


# Mutual-Info
scores_info_vect_df, scores_info_embed_df = ffs.mutual_info(num_features_vectorized=24,num_features_embed=24)
save_table(data_path, scores_info_vect_df, "scores_info_vect_df")
save_table(data_path, scores_info_embed_df, "scores_info_embed_df")
print("-----------------")
print("Mutual Info done")
print("-----------------")
plot_scores(scores_info_vect_df,scores_info_embed_df)


"""
Recursive Feature Elimination
"""

# Base-Learner -> Logistic Regression
base_learner = LogisticRegression(solver='liblinear', multi_class='ovr')
scores_rfe_lr_vect_df, scores_rfe_lr_embed_df = ffs.rfe(base_learner,num_features_vectorized=24,num_features_embed=24)
save_table(data_path, scores_rfe_lr_vect_df,"scores_rfe_lr_vect_df")
save_table(data_path, scores_rfe_lr_embed_df, "scores_rfe_lr_embed_df")
print("RFE + LR done")
print("-----------------")
plot_scores(scores_rfe_lr_vect_df,scores_rfe_lr_embed_df)


# Base-Learner -> Random Forest
base_learner = RandomForestClassifier()
scores_rfe_rf_vect_df, scores_rfe_rf_embed_df = ffs.rfe(base_learner,num_features_vectorized=24,num_features_embed=24)
save_table(data_path, scores_rfe_rf_vect_df,"scores_rfe_rf_vect_df")
save_table(data_path, scores_rfe_rf_embed_df, "scores_rfe_rf_embed_df")

print("RFE + Random Forest done")
print("-----------------")

plot_scores(scores_rfe_rf_vect_df,scores_rfe_rf_embed_df)


"""
Random Forest Feature Importance
"""
clf = RandomForestClassifier()
scores_rf_vect_df, scores_rf_embed_df = ffs.random_forest_feature_importance(clf,num_features_vectorized=24,num_features_embed=24)
save_table(data_path, scores_rf_vect_df,"scores_rf_vect_df")
save_table(data_path, scores_rf_embed_df, "scores_rf_embed_df")

print("Random Forest Feature Importance Done")
print("-----------------")
plot_scores(scores_rf_vect_df,scores_rf_embed_df)



""""
Lasso Regression
"""
selected_lasso_vect, selected_lasso_embed = ffs.Lasso(alpha=0.01)

save_table(data_path, selected_lasso_vect,"selected_lasso_vect")
save_table(data_path, selected_lasso_embed, "selected_lasso_embed")

print("Lasso  Done")
plot_scores(selected_lasso_vect,selected_lasso_embed)
