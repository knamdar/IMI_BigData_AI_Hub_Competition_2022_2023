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
import copy
from itertools import product

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


def val_experiment_gridsearch(model_name, X_dev, y_dev, N, hypers=None, test_size=0.5):
    hyper_params = copy.deepcopy(hypers)
    experiments_performances = []
    for i in range(N):
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=test_size, random_state=i)
        if hyper_params is None:
            exec("clf = eval(model_name)(random_state=i)")
        else:
            hyper_params["random_state"] = i
            #exec("clf = eval(model_name)(**hyper_params)")
            clf = eval(model_name)(**hyper_params)
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_val)
        experiments_performances.append(roc_auc_score(y_val, predictions, multi_class="ovr"))
        #print(roc_auc_score(y_val, predictions))

    mean_perf = np.mean(experiments_performances)
    return mean_perf


def grid_search(model_name, param_grid, X_dev, y_dev, N):
    params = param_grid.keys()
    best_params = {}
    for param in params:
        best_params[param] = None
    best_performance = 0
    combinations = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]
    for comb in combinations:
        performance = val_experiment_gridsearch(model_name, X_dev, y_dev, 2, hypers=comb)
        if performance > best_performance:
            best_performance = performance
            best_params = comb
    return best_params


def supervised_model_training(X, y, seed, threshold):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.25, random_state=seed) #randomly spliting the data to train/test
    X_dev, _, y_dev, _ = train_test_split(X_temp, y_temp, test_size=1-threshold, random_state=seed)
    best_params = grid_search("CatBoostClassifier", cb_clf_param_grid, X_dev, y_dev, N)
    clf = CatBoostClassifier(**best_params)

    clf.fit(X_dev, y_dev)
    predictions = clf.predict_proba(X_test)
    return clf, roc_auc_score(y_test, predictions, multi_class="ovr"), best_params


def experiment(threshold):
    threshold_catboost_perf = []
    best_params_list = []
    for seed in range(N):
        balanced_risk_sheet_df = balance_dset(risk_sheet_df, "RISK", seed) # make the dataset balanced

        y = balanced_risk_sheet_df['RISK']
        to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT']
        X = balanced_risk_sheet_df.drop(to_drop, axis=1)

        #print("Evaluating CatBoost on the balanced dataset")
        clf, auc_catboost, best_params = supervised_model_training(X, y, seed, threshold)
        threshold_catboost_perf.append(auc_catboost)
        best_params_list.append(best_params)
    return threshold_catboost_perf, best_params_list


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
    cb_clf_param_grid = {
        "learning_rate":[0.03, 0.05, 0.1],
        "iterations":[500, 1000, 2000],
        "depth": [6, 10]}

    N = 30 #Number of experiments
    THRESHOLD = 0.2
    catboost_aucs, best_params_list = experiment(THRESHOLD)
    pickle.dump(catboost_aucs, open("../results/catboost_aucs_GridSearch.p", "wb"))
    pickle.dump(best_params_list, open("../results/best_params_list_GridSearch.p", "wb"))

    end = time.time()
    duration = end-start

    print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")

