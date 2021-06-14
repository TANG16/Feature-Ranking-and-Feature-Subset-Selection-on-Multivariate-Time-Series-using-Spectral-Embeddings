
import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score, \
    precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

plt.style.use('ggplot')


# TODO: Create test and model classes
# TODO: Change the code in this library to oops code
# TODO: Write docstring for each class method

def plot_roc_cm(model_name, cm, n_classes, fpr, tpr, roc_auc, class_report_df, average="macro"):
    fig, ax = plt.subplots(1, 3, figsize=(25, 6))
    if average == "micro":

        ax[2].plot(fpr['avg / total'], tpr['avg / total'],
                   label='micro-average ROC curve (area = {0:0.2f})'
                         ''.format(roc_auc['avg / total']),
                   color='violet', linestyle='-', linewidth=4)

    else:

        ax[2].plot(fpr[average], tpr[average],
                   label='macro-average ROC curve (area = {0:0.2f})'
                         ''.format(roc_auc['avg / total']),
                   color='violet', linestyle='-', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #         label='macro-average ROC curve (area = {0:0.2f})'
    #               ''.format(roc_auc["macro"]),
    #         color='orange', linestyle=':', linewidth=4)
    lw = 2
    colors = cycle(['red', 'green', 'blue', 'purple', 'orange'])
    mapping = {
        0: 'F',
        1: 'C',
        2: 'B',
        3: 'M',
        4: 'X'
    }
    for i, color in zip(range(n_classes), colors):
        ax[2].plot(fpr[i], tpr[i], color=color, lw=lw,
                   label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(mapping[i], roc_auc[i]))
    labels = ['F', 'C', 'B', 'M', 'X', 'avg/total']
    sns.heatmap(cm, ax=ax[0], annot=True)
    ax[0].set_title(f"Confusion Matrix for {model_name}")
    ax[0].set_xticklabels(labels[:-1], rotation=45, ha='left', weight='bold', fontsize=14)
    ax[0].set_yticklabels(labels[:-1], rotation=45, ha='right', weight='bold', fontsize=14)
    sns.heatmap(class_report_df.iloc[:, :].drop(['support', 'pred'], axis=1), annot=True, ax=ax[1])
    ax[1].set_title(f'Classificaton report for {model_name}')

    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='left', weight='bold', fontsize=12)
    ax[1].set_yticklabels(labels, rotation=45, ha='right', weight='bold', fontsize=14)
    # ax[1].set_xticklabels(rotation=45)
    # ax[1].set_yticklabels(rotation=45, weight = 'bold')
    ax[2].plot([0, 1], [0, 1], 'k--', lw=lw)
    ax[2].axis(xmin=0.0, xmax=1, ymin=0, ymax=1.05)

    ax[2].set_xlabel('False Positive Rate', weight='bold', fontsize=14)
    ax[2].set_ylabel('True Positive Rate', weight='bold', fontsize=14)
    ax[2].set_title(f'ROC Curve for {model_name}')
    ax[2].legend(loc="lower right")
    ax[2].set_xticklabels(ax[2].get_xticklabels(), weight='bold', fontsize=10)
    ax[2].set_yticklabels(ax[2].get_yticklabels(), weight='bold', fontsize=10)


def class_report(model_name, y_true, y_pred, y_score=None, average='macro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
            y_true.shape,
            y_pred.shape)
              )
        return
    cm = confusion_matrix(y_true, y_pred)

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    # Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        ac_score = (cm.diagonal() / cm.sum(axis=1)).tolist()

        class_report_df['Accuracy'] = pd.Series(ac_score)
        class_report_df.loc['avg / total', 'Accuracy'] = accuracy_score(y_true, y_pred)
        # np.append(ac_score,accuracy_score(y_true,y_pred))
        # class_report_df['Accuracy']  = pd.Series(ac_score)
        # class_report_df['AUC'] = pd.Series(roc_auc)

    # for score in ac_score:
    #   print(score)

    # print(accuracy_score(y_true,y_pred))
    plot_roc_cm(model_name, cm, n_classes, fpr, tpr, roc_auc, class_report_df, average)
    return class_report_df


def kfold(model, n_splits, X, y):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_score_auc = []
    cv_score_f1 = []
    i = 1
    for train_index, test_index in kf.split(X, y):
        xtr, xvl = X[train_index], X[test_index]
        ytr, yvl = y[train_index], y[test_index]

        # model
        model.fit(xtr, ytr)
        score_auc = roc_auc_score(yvl, model.predict_proba(xvl), multi_class='ovr', average='macro')
        score_f1 = f1_score(yvl, model.predict(xvl), average='macro')
        # print('ROC AUC score:',score_auc)
        # print('F1 score: ', score_f1)
        cv_score_auc.append(score_auc)
        cv_score_f1.append(score_f1)
        # pred_test = lr.predict_proba(X_test)[:,1]
        # pred_test_full +=pred_test
        i += 1

    return cv_score_f1, cv_score_auc


def grid_search(estimator, params, X_train, y_train, cv=10):
    grid = GridSearchCV(estimator=estimator, param_grid=params, cv=cv, refit=True)
    grid.fit(X_train, y_train)
    print(f'Best params : {grid.best_params_}')
    print(f'Best score: {grid.best_score_}')


def __get_x_and_y():
    extracted_features = pd.read_csv(os.path.join(os.getcwd(), "data", "extracted_features_cleaned.csv"))
    X = extracted_features.iloc[:, 1:].dropna(axis=1)
    y = extracted_features.iloc[:, 0]
    y_encoded = LabelEncoder().fit_transform(y)
    return X, y_encoded


X, y_encoded = __get_x_and_y()

f1_score_models = {}
auc_score_models = {}
acc_score_models = {}

## Logistic Regression baseline model
lr = LogisticRegression(multi_class='ovr', penalty='l2')
f1_array_log, auc_array_log = kfold(lr, 5, X.values, y_encoded)
f1_score_models['Logistic Regression'] = np.mean(f1_array_log)
auc_score_models['Logistic Regression'] = np.mean(auc_array_log)
plt.figure(figsize=(10, 6))
plt.plot(auc_array_log, label='macro ROC');
plt.plot(f1_array_log, label='macro F-1 score');
plt.title("Logistic Regression K-fold cross validation")
plt.legend();

## Random Forest baseline model
rf = 'Random Forest'
rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_features='log2')
f1_array_rf, auc_array_rf = kfold(rf, 5, X.values, y_encoded)
f1_score_models['Random Forest'] = np.mean(f1_array_rf)
auc_score_models['Random Forest'] = np.mean(auc_array_rf)
plt.figure(figsize=(10, 6))
plt.plot(auc_array_rf, label='macro ROC');
plt.plot(f1_array_rf, label='macro F-1 score');
plt.title("Random Forest cross validation")
plt.legend();

## Decision tree baseline model
dt = 'Decision Tree'
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_leaf_nodes=100)
f1_array_dt, auc_array_dt = kfold(dt, 5, X.values, y_encoded)
f1_score_models['DT'] = np.mean(f1_array_dt)
auc_score_models['DT'] = np.mean(auc_array_dt)
plt.figure(figsize=(10, 6))
plt.plot(auc_array_dt, label='macro ROC');
plt.plot(f1_array_dt, label='macro F-1 score');
plt.title("Decision Tree cross validation")
plt.legend();

## KNN baseline model
knn = 'KNN'
knn = KNeighborsClassifier(weights='uniform')
f1_array_knn, auc_array_knn = kfold(knn, 5, X.values, y_encoded)
f1_score_models['KNN'] = np.mean(f1_array_knn)
auc_score_models['KNN'] = np.mean(auc_array_knn)
plt.figure(figsize=(10, 6))
plt.plot(auc_array_knn, label='macro ROC');
plt.plot(f1_array_knn, label='macro F-1 score');
plt.title("KNN cross validation")
plt.legend();
