import os
# os.chdir(os.path.pardir)
# print(os.getcwd())
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import StratifiedKFold

from feature_selection.viz import get_selected_columns_sklearn
import pandas as pd
import numpy as np
from data_cleaning.helper_functions import save_table as savee_table

data = pd.read_csv(os.path.join(os.getcwd(),"data","data.csv"))

def scale(X):
    return StandardScaler().fit_transform(X)

def kfold(model, n_splits, X, y):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_score_recall = []
    cv_score_precision = []
    cv_score_f1 = []
    i = 1
    for train_index, test_index in kf.split(X, y):
        xtr, xvl = X[train_index], X[test_index]
        ytr, yvl = y[train_index], y[test_index]

        # model
        model.fit(xtr, ytr)
        score_precision = precision_score(yvl, model.predict(xvl), average='macro')
        score_recall = recall_score(yvl, model.predict(xvl), average='macro')
        score_f1 = f1_score(yvl, model.predict(xvl), average="macro")
        cv_score_precision.append(score_precision)
        cv_score_recall.append(score_recall)
        cv_score_f1.append(score_f1)
        i += 1

    return cv_score_precision, cv_score_recall, cv_score_f1



"""
Testing data
"""
X, y = data.iloc[:,:-1], data.iloc[:,-1]
# X = StandardScaler().fit_transform(X)
coefs_1,get_selected_columns_1 = get_selected_columns_sklearn(X,y,0.0001)
coefs_2,get_selected_columns_2 = get_selected_columns_sklearn(X,y,0.001)
coesfs_3,get_selected_columns_3 = get_selected_columns_sklearn(X,y,0.01)
coesfs_4,get_selected_columns_4 = get_selected_columns_sklearn(X,y,0.1)


X = data.iloc[:,:-1]
X_set_1 = data.loc[:,get_selected_columns_1]
X_set_2 = data.loc[:,get_selected_columns_2]
X_set_3 = data.loc[:,get_selected_columns_3]
X_set_4 = data.loc[:,get_selected_columns_4]


X = scale(X)
X_set_1 = scale(X_set_1)
X_set_2 = scale(X_set_2)
X_set_3 = scale(X_set_3)
X_set_4 = scale(X_set_4)

precision_B = {}
precision_C = {}
precision_F = {}
precision_M = {}
precision_X = {}
precision_avg = {}

recall_B = {}
recall_C = {}
recall_F = {}
recall_M = {}
recall_X = {}
recall_avg = {}

f1_B = {}
f1_C = {}
f1_F = {}
f1_M = {}
f1_X = {}
f1_avg = {}


precision_score_models_all = {}
recall_scores_models_all = {}
f1_scores_models_all = {}

precision_score_models_3 = {}
recall_scores_models_3 = {}
f1_scores_models_3 = {}

precision_score_models_4 = {}
recall_scores_models_4 = {}
f1_scores_models_4 = {}


"""
Modelling
"""
lr = LogisticRegression(multi_class = 'ovr', penalty = 'l2')
knn = KNeighborsClassifier(weights='uniform')
dt = DecisionTreeClassifier(criterion = 'entropy',max_depth = 10, max_leaf_nodes=100)
rf = RandomForestClassifier(n_estimators=50,criterion='entropy',max_features='log2')

"""
Precision, Recall and F-1 Score per class
"""


lr.fit(X,y)
knn.fit(X,y)
dt.fit(X,y)
rf.fit(X,y)
precision_all_lr, recall_all_lr, fscore_all_lr, _ = score(y, lr.predict(X))
precision_all_knn, recall_all_knn, fscore_all_knn, _ = score(y, knn.predict(X))
precision_all_dt, recall_all_dt, fscore_all_dt, _ = score(y, dt.predict(X))
precision_all_rf, recall_all_rf, fscore_all_rf, _ = score(y, rf.predict(X))


# # alpha = 0.01
X = X_set_3
lr.fit(X,y)
knn.fit(X,y)
dt.fit(X,y)
rf.fit(X,y)
precision_3_lr, recall_3_lr, fscore_3_lr, _ = score(y, lr.predict(X))
precision_3_knn, recall_3_knn, fscore_3_knn, _ = score(y, knn.predict(X))
precision_3_dt, recall_3_dt, fscore_3_dt, _ = score(y, dt.predict(X))
precision_3_rf, recall_3_rf, fscore_3_rf, _ = score(y, rf.predict(X))

# # alpha = 0.1
X = X_set_4
lr.fit(X,y)
knn.fit(X,y)
dt.fit(X,y)
rf.fit(X,y)
precision_4_lr, recall_4_lr, fscore_4_lr, _ = score(y, lr.predict(X))
precision_4_knn, recall_4_knn, fscore_4_knn, _ = score(y, knn.predict(X))
precision_4_dt, recall_4_dt, fscore_4_dt, _ = score(y, dt.predict(X))
precision_4_rf, recall_4_rf, fscore_4_rf, _ = score(y, rf.predict(X))

"""
Converting scores to dataframe
"""

precision_df_all = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])
precision_df_3 = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])
precision_df_4 = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])


recall_df_all = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])
recall_df_3 = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])
recall_df_4 = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])


fscore_df_all = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])
fscore_df_3 = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])
fscore_df_4 = pd.DataFrame(columns = ['B','C','F','M','X'], index =  ["LR", "KNN", "DT", "RF"])




precision_df_all.loc["LR"] =  precision_all_lr
precision_df_all.loc["KNN"] = precision_all_knn
precision_df_all.loc["DT"] =  precision_all_dt
precision_df_all.loc["RF"] =  precision_all_rf


recall_df_all.loc["LR"] =  recall_all_lr
recall_df_all.loc["KNN"] = recall_all_knn
recall_df_all.loc["DT"] =  recall_all_dt
recall_df_all.loc["RF"] =  recall_all_rf


fscore_df_all.loc["LR"] =  fscore_all_lr
fscore_df_all.loc["KNN"] = fscore_all_knn
fscore_df_all.loc["DT"] =  fscore_all_dt
fscore_df_all.loc["RF"] =  fscore_all_rf


"""
3
"""

precision_df_3.loc["LR"] =  precision_3_lr
precision_df_3.loc["KNN"] = precision_3_knn
precision_df_3.loc["DT"] =  precision_3_dt
precision_df_3.loc["RF"] =  precision_3_rf


recall_df_3.loc["LR"] =  recall_3_lr
recall_df_3.loc["KNN"] = recall_3_knn
recall_df_3.loc["DT"] =  recall_3_dt
recall_df_3.loc["RF"] =  recall_3_rf


fscore_df_3.loc["LR"] =  fscore_3_lr
fscore_df_3.loc["KNN"] = fscore_3_knn
fscore_df_3.loc["DT"] =  fscore_3_dt
fscore_df_3.loc["RF"] =  fscore_3_rf



"""
4
"""

precision_df_4.loc["LR"] =  precision_4_lr
precision_df_4.loc["KNN"] = precision_4_knn
precision_df_4.loc["DT"] =  precision_4_dt
precision_df_4.loc["RF"] =  precision_4_rf


recall_df_4.loc["LR"] =  recall_4_lr
recall_df_4.loc["KNN"] = recall_4_knn
recall_df_4.loc["DT"] =  recall_4_dt
recall_df_4.loc["RF"] =  recall_4_rf


fscore_df_4.loc["LR"] =  fscore_4_lr
fscore_df_4.loc["KNN"] = fscore_4_knn
fscore_df_4.loc["DT"] =  fscore_4_dt
fscore_df_4.loc["RF"] =  fscore_4_rf


path = os.path.join(os.getcwd(),"data")
savee_table(path,precision_df_all,"precision_df_all",index=True)
savee_table(path,precision_df_3,"precision_df_3",index=True)
savee_table(path,precision_df_4,"precision_df_4",index=True)
savee_table(path,recall_df_all,"recall_df_all",index=True)
savee_table(path,recall_df_3,"recall_df_3",index=True)
savee_table(path,recall_df_4,"recall_df_4",index=True)
savee_table(path,fscore_df_all,"fscore_df_all",index=True)
savee_table(path,fscore_df_3,"fscore_df_3",index=True)
savee_table(path,fscore_df_4,"fscore_df_4",index=True)


"""
Calculating results for visualisation
"""
# # LR
# ## ALL
prec, recall, f1 = kfold(lr, 5, X, y)
precision_score_models_all['Logistic Regression'] = np.mean(prec)
recall_scores_models_all['Logistic Regression'] = np.mean(recall)
f1_scores_models_all['Logistic Regression'] = np.mean(f1)

# ## 3
prec, recall, f1 = kfold(lr, 5, X_set_3, y)
precision_score_models_3['Logistic Regression'] = np.mean(prec)
recall_scores_models_3['Logistic Regression'] = np.mean(recall)
f1_scores_models_3['Logistic Regression'] = np.mean(f1)

# ## 4
prec, recall, f1 = kfold(lr, 5, X_set_4, y)
precision_score_models_4['Logistic Regression'] = np.mean(prec)
recall_scores_models_4['Logistic Regression'] = np.mean(recall)
f1_scores_models_4['Logistic Regression'] = np.mean(f1)


# KNN
## ALL
prec, recall, f1 = kfold(knn, 5, X, y)
precision_score_models_all['KNN'] = np.mean(prec)
recall_scores_models_all['KNN'] = np.mean(recall)
f1_scores_models_all['KNN'] = np.mean(f1)

# ## 3
prec, recall, f1 = kfold(knn, 5, X_set_3, y)
precision_score_models_3['KNN'] = np.mean(prec)
recall_scores_models_3['KNN'] = np.mean(recall)
f1_scores_models_3['KNN'] = np.mean(f1)

# ## 4
prec, recall, f1 = kfold(knn, 5, X_set_4, y)
precision_score_models_4['KNN'] = np.mean(prec)
recall_scores_models_4['KNN'] = np.mean(recall)
f1_scores_models_4['KNN'] = np.mean(f1)

# DT
## ALL
prec, recall, f1 = kfold(dt, 5, X, y)
precision_score_models_all['DT'] = np.mean(prec)
recall_scores_models_all['DT'] = np.mean(recall)
f1_scores_models_all['DT'] = np.mean(f1)

## 3
prec, recall, f1 = kfold(dt, 5, X_set_3, y)
precision_score_models_3['DT'] = np.mean(prec)
recall_scores_models_3['DT'] = np.mean(recall)
f1_scores_models_3['DT'] = np.mean(f1)

## 4
prec, recall, f1 = kfold(dt, 5, X_set_4, y)
precision_score_models_4['DT'] = np.mean(prec)
recall_scores_models_4['DT'] = np.mean(recall)
f1_scores_models_4['DT'] = np.mean(f1)

# RF
## ALL
prec, recall, f1 = kfold(rf, 5, X, y)
precision_score_models_all['RF'] = np.mean(prec)
recall_scores_models_all['RF'] = np.mean(recall)
f1_scores_models_all['RF'] = np.mean(f1)

## 3
prec, recall, f1 = kfold(rf, 5, X_set_3, y)
precision_score_models_3['RF'] = np.mean(prec)
recall_scores_models_3['RF'] = np.mean(recall)
f1_scores_models_3['RF'] = np.mean(f1)

## 4
prec, recall, f1 = kfold(rf, 5, X_set_4, y)
precision_score_models_4['RF'] = np.mean(prec)
recall_scores_models_4['RF'] = np.mean(recall)
f1_scores_models_4['RF'] = np.mean(f1)

# # LR
# ## ALL
prec, recall, f1 = kfold(lr, 5, X, y)
precision_score_models_all['Logistic Regression'] = np.mean(prec)
recall_scores_models_all['Logistic Regression'] = np.mean(recall)
f1_scores_models_all['Logistic Regression'] = np.mean(f1)

# ## 3
prec, recall, f1 = kfold(lr, 5, X_set_3, y)
precision_score_models_3['Logistic Regression'] = np.mean(prec)
recall_scores_models_3['Logistic Regression'] = np.mean(recall)
f1_scores_models_3['Logistic Regression'] = np.mean(f1)

# ## 4
prec, recall, f1 = kfold(lr, 5, X_set_4, y)
precision_score_models_4['Logistic Regression'] = np.mean(prec)
recall_scores_models_4['Logistic Regression'] = np.mean(recall)
f1_scores_models_4['Logistic Regression'] = np.mean(f1)


# KNN
## ALL
prec, recall, f1 = kfold(knn, 5, X, y)
precision_score_models_all['KNN'] = np.mean(prec)
recall_scores_models_all['KNN'] = np.mean(recall)
f1_scores_models_all['KNN'] = np.mean(f1)

# ## 3
prec, recall, f1 = kfold(knn, 5, X_set_3, y)
precision_score_models_3['KNN'] = np.mean(prec)
recall_scores_models_3['KNN'] = np.mean(recall)
f1_scores_models_3['KNN'] = np.mean(f1)

# ## 4
prec, recall, f1 = kfold(knn, 5, X_set_4, y)
precision_score_models_4['KNN'] = np.mean(prec)
recall_scores_models_4['KNN'] = np.mean(recall)
f1_scores_models_4['KNN'] = np.mean(f1)

# DT
## ALL
prec, recall, f1 = kfold(dt, 5, X, y)
precision_score_models_all['DT'] = np.mean(prec)
recall_scores_models_all['DT'] = np.mean(recall)
f1_scores_models_all['DT'] = np.mean(f1)

## 3
prec, recall, f1 = kfold(dt, 5, X_set_3, y)
precision_score_models_3['DT'] = np.mean(prec)
recall_scores_models_3['DT'] = np.mean(recall)
f1_scores_models_3['DT'] = np.mean(f1)

## 4
prec, recall, f1 = kfold(dt, 5, X_set_4, y)
precision_score_models_4['DT'] = np.mean(prec)
recall_scores_models_4['DT'] = np.mean(recall)
f1_scores_models_4['DT'] = np.mean(f1)

# RF
## ALL
prec, recall, f1 = kfold(rf, 5, X, y)
precision_score_models_all['RF'] = np.mean(prec)
recall_scores_models_all['RF'] = np.mean(recall)
f1_scores_models_all['RF'] = np.mean(f1)

## 3
prec, recall, f1 = kfold(rf, 5, X_set_3, y)
precision_score_models_3['RF'] = np.mean(prec)
recall_scores_models_3['RF'] = np.mean(recall)
f1_scores_models_3['RF'] = np.mean(f1)

## 4
prec, recall, f1 = kfold(rf, 5, X_set_4, y)
precision_score_models_4['RF'] = np.mean(prec)
recall_scores_models_4['RF'] = np.mean(recall)
f1_scores_models_4['RF'] = np.mean(f1)


"""
Converting results to dataframe
"""

results_all = pd.DataFrame(columns = ['Model', 'Precision', 'Recall', 'F-1 Score'])
results_3 = pd.DataFrame(columns = ['Model', 'Precision', 'Recall', 'F-1 Score'])
results_4 = pd.DataFrame(columns = ['Model', 'Precision', 'Recall', 'F-1 Score'])


results_all['Model'] = ['Logistic Regression','KNN','Decision Tree','Random Forest']
results_3['Model'] = ['Logistic Regression','KNN','Decision Tree','Random Forest']
results_4['Model'] = ['Logistic Regression','KNN','Decision Tree','Random Forest']


results_all['Precision'] = list(precision_score_models_all.values())
results_all['Recall'] = list(recall_scores_models_all.values())
results_all['F-1 Score'] = list(f1_scores_models_all.values())

results_3['Precision'] = list(precision_score_models_3.values())
results_3['Recall'] = list(recall_scores_models_3.values())
results_3['F-1 Score'] = list(f1_scores_models_3.values())

results_4['Precision'] = list(precision_score_models_4.values())
results_4['Recall'] = list(recall_scores_models_4.values())
results_4['F-1 Score'] = list(f1_scores_models_4.values())


savee_table(path,results_all,"results_all",index=False)
savee_table(path,results_3,"results_3",index=False)
savee_table(path,results_4,"results_4",index=False)
