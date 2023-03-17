#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Mar 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition

@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################

import numpy as np
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # load the saved AUCs
    xgb_aucs = pickle.load(open("../data/xgb_aucs.p", "rb"))
    catboost_aucs = pickle.load(open("../data/catboost_aucs.p", "rb"))
    thresholds = list(np.linspace(0.1, 1, 19))

    meanAUC_XGBoost = []
    meanAUC_CatBoost = []
    minAUC_XGBoost = []
    minAUC_CatBoost = []
    maxAUC_XGBoost = []
    maxAUC_CatBoost = []
    for i in range(len(thresholds)):
        meanAUC_XGBoost.append(np.mean(xgb_aucs[i]))
        minAUC_XGBoost.append(np.min(xgb_aucs[i]))
        maxAUC_XGBoost.append(np.max(xgb_aucs[i]))
        meanAUC_CatBoost.append(np.mean(catboost_aucs[i]))
        minAUC_CatBoost.append(np.min(catboost_aucs[i]))
        maxAUC_CatBoost.append(np.max(catboost_aucs[i]))

    fig, ax = plt.subplots(1)
    ax.set_title("Test AUC vs dataset size")
    ax.plot(thresholds, meanAUC_XGBoost, linestyle='-', lw=2, color='b', label='Test mean AUC (XGBoost)', alpha=.8)
    ax.fill_between(thresholds, minAUC_XGBoost, maxAUC_XGBoost, color='b', alpha=0.1)

    ax.plot(thresholds, meanAUC_CatBoost, linestyle='-', lw=2, color='r', label='Test mean AUC (CatBoost)', alpha=.8)
    ax.fill_between(thresholds, minAUC_CatBoost, maxAUC_CatBoost, color='r', alpha=0.1)


    ax.set(xlim=[0, 1], ylim=[0.95, 1.0])
    ax.legend(loc="lower right")
    ax.set(ylabel='AUC')
    ax.legend(loc="lower right")
    ax.set(xlabel ='Training Dataset Size', ylabel='AUC')

    print("mean AUC for XGBoost:", np.mean(meanAUC_XGBoost), "with average range of", np.mean(np.array(maxAUC_XGBoost)-np.array(minAUC_XGBoost)))
    print("mean AUC for CatBoost:", np.mean(meanAUC_CatBoost), "with average range of", np.mean(np.array(maxAUC_CatBoost)-np.array(minAUC_CatBoost)))