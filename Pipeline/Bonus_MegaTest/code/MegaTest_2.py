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
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from numpy import percentile
from boxplots_2 import plot_boxes
import time

HYPERS = {'learning_rate': 0.03, 'iterations': 2000, 'depth': 6, 'silent':True}

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

    ex_mid_risk_data = df.iloc[list(set(all_mid_risk.index).difference(set(mid_risk_to_keep)))].copy()
    ex_low_risk_data = df.iloc[list(set(all_low_risk.index).difference(set(low_risk_to_keep)))].copy()

    new_df = pd.concat([high_risk_data, mid_risk_data, low_risk_data], axis=0)
    ex_df = pd.concat([ex_mid_risk_data, ex_low_risk_data], axis=0)
    return new_df, ex_df


def small_test(X, y, clf, seed):
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #randomly spliting the data to train/test
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.8, random_state=seed)
    clf.fit(X_train, y_train)
    #predictions = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)
    return clf, roc_auc_score(y_test, predictions, multi_class="ovr")


def mega_test(X, y, clf, seed, X_unused, y_unused):
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #randomly spliting the data to train/test
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.8, random_state=seed)
    X_mt = pd.concat([X_val, X_test, X_unused])
    X_mt.reset_index(inplace=True, drop=True)
    y_mt = pd.concat([y_val, y_test, y_unused])
    y_mt.reset_index(inplace=True, drop=True)
    clf.fit(X_train, y_train)
    #predictions = clf.predict(X_mt)
    predictions = clf.predict_proba(X_mt)
    return clf, roc_auc_score(y_mt, predictions, multi_class="ovr")


def experiment(seed):
    balanced_risk_sheet_df, excluded_df = balance_dset(risk_sheet_df, "RISK", seed) # make the dataset balanced

    y = balanced_risk_sheet_df['RISK']
    to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT']
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    y_unused = excluded_df['RISK']
    X_unused = excluded_df.drop(to_drop, axis=1)

    #print("Evaluating CatBoost on the balanced dataset")
    HYPERS["random_state"] = seed
    clf = CatBoostClassifier(**HYPERS)
    clf, st_auc = small_test(X, y, clf, seed)

    clf2 = CatBoostClassifier(**HYPERS)
    clf2, mt_auc = mega_test(X, y, clf2, seed, X_unused, y_unused)
    return st_auc, mt_auc


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
    small_test_aucs = []
    mega_test_aucs = []
    for i in range(N):
        print("working on experiment number", i+1)
        small_test_auc, mega_test_auc = experiment(i)
        small_test_aucs.append(small_test_auc)
        mega_test_aucs.append(mega_test_auc)
    min_st, q1_st, med_st, q3_st, max_st = five_number_summary(small_test_aucs)
    mean_st = np.mean(small_test_aucs)
    sd_st = np.std(small_test_aucs)

    min_mt, q1_mt, med_mt, q3_mt, max_mt = five_number_summary(mega_test_aucs)
    mean_mt = np.mean(mega_test_aucs)
    sd_mt = np.std(mega_test_aucs)
    plot_boxes(mean_st, sd_st, [min_st, q1_st, med_st, q3_st, max_st],
               mean_mt, sd_mt, [min_mt, q1_mt, med_mt, q3_mt, max_mt])
    print("small test stats (mean, sd, 5stats):", [mean_st, sd_st, min_st, q1_st, med_st, q3_st, max_st])
    print("mega test stats (mean, sd, 5stats):", [mean_mt, sd_mt, min_mt, q1_mt, med_mt, q3_mt, max_mt])

    end = time.time()
    duration = end-start

    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")

