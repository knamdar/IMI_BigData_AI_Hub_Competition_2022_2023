#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Created in Jan 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition
@author: Ernest (Khashayar) Namdar, Jay Yoo
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF and cuML from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""

import pandas as pd #replace with cudf
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix
from cf_matrix import make_confusion_matrix
from sklearn.metrics import roc_auc_score
import shap


def balance_dset(df, target):
    high_risk_data = df[df[target]==1].copy()
    labels = df[target].copy()
    all_low_risk = labels[labels == 0]
    low_risk_to_keep = np.random.choice(all_low_risk.index, size=high_risk_data.shape[0], replace=False)
    low_risk_data = df.iloc[low_risk_to_keep].copy()
    new_df = pd.concat([high_risk_data, low_risk_data], axis=0)
    return new_df


def supervised_model_training(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) #randomly spliting the data to train/test
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    make_confusion_matrix(cm, group_names=['low', 'medium', 'high'])
    plt.show()
    predictions = clf.predict_proba(X_test)[:, 1]
    print("AUC score:", roc_auc_score(y_test, predictions))

    return clf


if __name__ == "__main__":
    risk_sheet_path = "../data/risk_sheet.xlsx" # Loading the preprocessed spreadsheet of features and ground truths
    risk_sheet_df = pd.read_excel(risk_sheet_path)

    bad_actors_v2_path = "../data/bad_actors_v2.csv"
    bad_actors_v2_df = pd.read_csv(bad_actors_v2_path)
    bad_actors_v2_df = bad_actors_v2_df.drop(["NAME"], axis=1)

    overlap_df = pd.merge(risk_sheet_df, bad_actors_v2_df, how="inner", on=["CUSTOMER_ID"], suffixes=["", "_"])
    nonoverlap_df = risk_sheet_df[~risk_sheet_df["CUSTOMER_ID"].isin(overlap_df["CUSTOMER_ID"])]
    overlap_df.insert(0, "TargetLabel", "BadActor")
    nonoverlap_df.insert(0, "TargetLabel", "Normal")

    risk_sheet_df = overlap_df.append(nonoverlap_df)


# #########################Whole Dataset#######################################
    # EDA_Plotting Pie plot of different classes
    lbl_counts = pd.DataFrame(risk_sheet_df['TargetLabel'].value_counts())
    # whole dataset class contribution
    labels = list(lbl_counts.index)
    values = [int(value) for value in lbl_counts.values]
    pie = plt.figure()
    plt.pie(values, labels=labels, autopct=lambda p:f'{p:.2f}%, {p*sum(values)/100 :.0f}')
    plt.title("Class Contributions Over the Whole Dataset")
    plt.axis('off')

    # quantizing the features
    ordered_satisfaction = ['High', 'Low', 'Moderate']
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
    risk_sheet_df["TargetLabel"].replace(['Normal', 'BadActor'],
                                                  [0, 1], inplace=True)
    # separating year, month, and day of DOBs andwhen  customers joined the bank as standalone features
    risk_sheet_df['BIRTH_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).year
    risk_sheet_df['BIRTH_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).month
    risk_sheet_df['BIRTH_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).day
    risk_sheet_df['CUST_ADD_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).year
    risk_sheet_df['CUST_ADD_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).month
    risk_sheet_df['CUST_ADD_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).day

    # creating the GT vector and the feature matrix
    y = risk_sheet_df['TargetLabel']
    to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT', 'TargetLabel'] #we are skipping two DTs which is not ok
    X = risk_sheet_df.drop(to_drop, axis=1)

    print("Evaluating XGB on the whole dataset")
    clf = XGBClassifier(random_state=0)
    supervised_model_training(X, y, clf) #supervised learning over the whole dataset


# #########################Balanced Dataset####################################
    balanced_risk_sheet_df = balance_dset(risk_sheet_df, "TargetLabel") # make the dataset balanced

    lbl_counts = pd.DataFrame(balanced_risk_sheet_df['TargetLabel'].value_counts())
    # whole dataset class contribution
    labels = list(lbl_counts.index)
    values = [int(value) for value in lbl_counts.values]
    pie = plt.figure()
    plt.pie(values, labels=labels, autopct=lambda p:f'{p:.2f}%, {p*sum(values)/100 :.0f}')
    plt.title("Class Contributions Over the Balanced Dataset")
    plt.axis('off')

    y = balanced_risk_sheet_df['TargetLabel']
    to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT', "TargetLabel"]
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    print("Evaluating XGB on the balanced dataset")
    clf = XGBClassifier(random_state=0)
    clf = supervised_model_training(X, y, clf)

    ax = xgb.plot_importance(clf)
    fig = ax.figure
    fig.set_size_inches(10, 10)

    print("Evaluating CatBoost on the balanced dataset")
    clf = CatBoostClassifier(random_state=0)
    clf = supervised_model_training(X, y, clf)


    # XAI for the CatBoost using shap
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Pool(X_train, y_train))
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar")


    y = balanced_risk_sheet_df['TargetLabel']
    to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT', "TargetLabel", "QUERY_RISK"]
    X = balanced_risk_sheet_df.drop(to_drop, axis=1)

    print("Evaluating XGB on the balanced dataset without QUERY_RISK")
    clf = XGBClassifier(random_state=0)
    clf = supervised_model_training(X, y, clf)

    ax = xgb.plot_importance(clf)
    fig = ax.figure
    fig.set_size_inches(10, 10)


    print("Evaluating CatBoost on the balanced dataset without QUERY_RISK")
    clf = CatBoostClassifier(random_state=0)
    clf = supervised_model_training(X, y, clf)


    # XAI for the CatBoost using shap
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Pool(X_train, y_train))
    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar")