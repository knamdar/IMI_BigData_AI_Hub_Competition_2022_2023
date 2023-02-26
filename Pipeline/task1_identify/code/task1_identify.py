#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Dec 2022

Team 44
2022-23 IMI BIGDataAIHUB Case Competition
@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import copy
from cf_matrix import make_confusion_matrix

def optimal_threshold_jstat(tpr, fpr, thresholds):
    # Calculate the Youden's J statistic
    youdenJ = tpr - fpr
    # Find the optimal threshold
    index = np.argmax(youdenJ)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    youdenJOpt = round(youdenJ[index], ndigits = 4)
    fprOpt = round(fpr[index], ndigits = 4)
    tprOpt = round(tpr[index], ndigits = 4)
    print('Best Threshold: {} with Youden J statistic: {}'.format(thresholdOpt, youdenJOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

def last_arg_max(arr):
    rev_arr = arr[::-1]
    i = len(rev_arr)-np.argmax(rev_arr)-1
    return i


if __name__ == "__main__":
    risk_sheet_path = "../data/risk_sheet.xlsx" # Loading the preprocessed spreadsheet of features and ground truths
    risk_sheet_df = pd.read_excel(risk_sheet_path)

    risk_sheet_df["RISK"].replace(['low', 'medium', 'high'],
                                                 [0, 1, 2], inplace=True)
    y = risk_sheet_df['RISK']
    X = np.array([[value] for value in risk_sheet_df['QUERY_RISK'].values])

    lbl_counts = list(y.value_counts())

    # fitting to the whole data
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y)
    predictions = clf.predict_proba(X)
    print("ovr AUC score:", roc_auc_score(y, predictions, multi_class="ovr"))

    # fitting to the whole data with class weight
    clf = DecisionTreeClassifier(class_weight=dict([(i, y.shape[0]/n) for i,n in enumerate(lbl_counts)])
                                 , random_state=0)
    clf.fit(X, y)
    predictions = clf.predict_proba(X)
    print("ovr AUC score with class wight:", roc_auc_score(y, predictions, multi_class="ovr"))

    # focusing on the high-risk customers
    y_bin = copy.deepcopy(y)
    y_bin[y_bin<2]=0
    y_bin[y_bin>0]=1
    bin_lbl_counts = list(y_bin.value_counts())

    # fitting to the whole binary data
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X, y_bin)
    predictions = clf.predict_proba(X)[:,1]
    print("AUC score:", roc_auc_score(y_bin, predictions))

    # fitting to the whole data with class weight
    clf = DecisionTreeClassifier(class_weight=dict([(i, y.shape[0]/n) for i,n in enumerate(bin_lbl_counts)])
                                 , random_state=0)
    clf.fit(X, y_bin)
    predictions = clf.predict_proba(X)[:, 1]
    print("AUC score with class wight:", roc_auc_score(y_bin, predictions))

    # ROC Analysis
    fpr, tpr, thresholds = roc_curve(y_bin, predictions)
    thresholdOpt = thresholds[np.argmax(tpr==1)]

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - DecisionTree')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], ls="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # catching all actual positives
    cf_matrix = confusion_matrix(y_bin, predictions>=thresholdOpt)
    make_confusion_matrix(cf_matrix, group_names=['Low_Risk', 'High_Risk'])

    # catching all actual negatives
    thresholdOpt = thresholds[last_arg_max(fpr==0)]
    cf_matrix = confusion_matrix(y_bin, predictions>=thresholdOpt)
    make_confusion_matrix(cf_matrix, group_names=['Low_Risk', 'High_Risk'])

    #Actors that a model train with QUERY risk as input and Risk as output confidently detects
    Bad_Actors = risk_sheet_df[predictions==1]
    Bad_Actors.to_csv("../results/bad_actors.csv", index=False)

