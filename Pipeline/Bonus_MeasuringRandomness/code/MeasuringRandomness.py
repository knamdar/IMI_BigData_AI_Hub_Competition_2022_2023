#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Created in Jan 2023

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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from numpy import percentile
from boxplots import plot_boxes
import time


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


def supervised_model_training(X, y, clf, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #randomly spliting the data to train/test
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)
    return clf, roc_auc_score(y_test, predictions, multi_class="ovr")


def experiment(seed):
    balanced_risk_sheet_df = balance_dset(risk_sheet_df, "RISK", seed) # make the dataset balanced

    y = balanced_risk_sheet_df['RISK']
    to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT']
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    #print("Evaluating XGB on the balanced dataset")
    clf = XGBClassifier(random_state=seed)
    clf, auc_xgb = supervised_model_training(X, y, clf, seed)

    #print("Evaluating CatBoost on the balanced dataset")
    clf = CatBoostClassifier(random_state=seed)
    clf, auc_catboost = supervised_model_training(X, y, clf, seed)
    return auc_xgb, auc_catboost


def five_number_summary(lst):
    quartiles = percentile(lst, [25,50,75])
    data_min, data_max = min(lst), max(lst)
    return data_min, quartiles[0], quartiles[1], quartiles[2], data_max


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
    xgb_aucs = []
    catboost_aucs = []
    for i in range(N):
        xgb_auc, catboost_auc = experiment(i)
        xgb_aucs.append(xgb_auc)
        catboost_aucs.append(catboost_auc)
    min_xg, q1_xg, med_xg, q3_xg, max_xg = five_number_summary(xgb_aucs)
    mean_xg = np.mean(xgb_aucs)
    sd_xg = np.std(xgb_aucs)

    min_cb, q1_cb, med_cb, q3_cb, max_cb = five_number_summary(catboost_aucs)
    mean_cb = np.mean(catboost_aucs)
    sd_cb = np.std(catboost_aucs)
    plot_boxes(mean_xg, sd_xg, [min_xg, q1_xg, med_xg, q3_xg, max_xg],
               mean_cb, sd_cb, [min_cb, q1_cb, med_cb, q3_cb, max_cb])
    print("xgb stats (mean, sd, 5stats):", [mean_xg, sd_xg, min_xg, q1_xg, med_xg, q3_xg, max_xg])
    print("catboost stats (mean, sd, 5stats):", [mean_cb, sd_cb, min_cb, q1_cb, med_cb, q3_cb, max_cb])

    end = time.time()
    duration = end-start

    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")

