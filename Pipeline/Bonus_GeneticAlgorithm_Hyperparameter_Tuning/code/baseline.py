#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Mar 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition

@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF and cuML from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""

import pandas as pd #replace with cudf
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import time
import pickle


def balance_dset(df, target, seed):
    np.random.seed(seed)
    high_risk_data = df[df[target]==2].copy()
    labels = df[target].copy()
    all_mid_risk = labels[labels == 1]
    all_low_risk = labels[labels == 0]
    mid_risk_to_keep = np.random.choice(all_mid_risk.index, size=high_risk_data.shape[0], replace=False)
    low_risk_to_keep = np.random.choice(all_low_risk.index, size=high_risk_data.shape[0], replace=False)
    mid_risk_data = df.iloc[mid_risk_to_keep].copy()
    low_risk_data = df.iloc[low_risk_to_keep].copy()
    new_df = pd.concat([high_risk_data, mid_risk_data, low_risk_data], axis=0)
    return new_df


def supervised_model_training(X, y, clf, seed, threshold):
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #randomly spliting the data to train/test
    if threshold == 1:
        X_train, y_train = X_dev, y_dev
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=1-threshold, random_state=seed)
    clf.fit(X_train, y_train)
    #predictions = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)
    return clf, roc_auc_score(y_test, predictions, multi_class="ovr")


def experiment(threshold):
    threshold_catboost_perf = []
    for seed in range(N):
        balanced_risk_sheet_df = balance_dset(risk_sheet_df, "RISK", seed) # make the dataset balanced

        y = balanced_risk_sheet_df['RISK']
        to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT']
        X = balanced_risk_sheet_df.drop(to_drop, axis=1)

        #print("Evaluating CatBoost on the balanced dataset")
        clf = CatBoostClassifier(random_state=seed)
        clf, auc_catboost = supervised_model_training(X, y, clf, seed, threshold)
        threshold_catboost_perf.append(auc_catboost)
    return threshold_catboost_perf


if __name__ == "__main__":
    start = time.time()

    risk_sheet_path = "../data/risk_sheet.xlsx" # Loading the preprocessed spreadsheet of features and ground truths
    risk_sheet_df = pd.read_excel(risk_sheet_path)


    # quantizing the features
    risk_sheet_df["COUNTRY_RISK_INCOME"].replace(['Low', 'Moderate', 'High'],
                                                 [0, 1, 2], inplace=True)
    risk_sheet_df["COUNTRY_RISK_RESIDENCY"].replace(['Low', 'Moderate', 'High'],
                                                 [0, 1, 2], inplace=True)
    risk_sheet_df["OCPTN_RISK"].replace(['Low', 'Moderate', 'High'],
                                                 [0, 1, 2], inplace=True)
    risk_sheet_df["RISK"].replace(['low', 'medium', 'high'],
                                                 [0, 1, 2], inplace=True)
    risk_sheet_df["GENDER"].replace(['Female', 'Male'],
                                                 [0, 1], inplace=True)
    # separating year, month, and day of DOBs andwhen  customers joined the bank as standalone features
    risk_sheet_df['BIRTH_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).year
    risk_sheet_df['BIRTH_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).month
    risk_sheet_df['BIRTH_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).day
    risk_sheet_df['CUST_ADD_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).year
    risk_sheet_df['CUST_ADD_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).month
    risk_sheet_df['CUST_ADD_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).day

# #########################Balanced Dataset####################################

    N = 30 #Number of experiments
    THRESHOLD = 0.2
    catboost_aucs = experiment(THRESHOLD)
    pickle.dump(catboost_aucs, open("../results/catboost_aucs_baseline.p", "wb"))

    end = time.time()
    duration = end-start

    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")

