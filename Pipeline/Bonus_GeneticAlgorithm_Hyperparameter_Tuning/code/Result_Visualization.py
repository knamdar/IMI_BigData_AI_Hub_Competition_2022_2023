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
from numpy import percentile
from scipy.stats import t as t_dist

def five_number_summary(lst):
    quartiles = percentile(lst, [25,50,75])
    data_min, data_max = min(lst), max(lst)
    return data_min, quartiles[0], quartiles[1], quartiles[2], data_max

def ttest2(m1,m2,s1,s2,n1,n2,m0=0,equal_variance=False):
    if equal_variance is False:
        se = np.sqrt((s1**2/n1) + (s2**2/n2))
        # welch-satterthwaite df
        df = ((s1**2/n1 + s2**2/n2)**2)/((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    else:
        # pooled standard deviation, scaled by the sample sizes
        se = np.sqrt((1/n1 + 1/n2) * ((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
        df = n1+n2-2

    t = (m1-m2-m0)/se
    dat = {"Difference of means":m1-m2, "Std Error":se, "t":t, "p-value":2*t_dist.cdf(-abs(t),df)}
    return dat


def plot_boxes(m1_mean, m1_sd, m1_5stats, m2_mean, m2_sd, m2_5stats, m3_mean, m3_sd, m3_5stats):
    pval12 = ttest2(m1_mean,m2_mean,m1_sd,m2_sd,30,30)["p-value"]
    pval23 = ttest2(m2_mean,m3_mean,m2_sd,m3_sd,30,30)["p-value"]

    # #########################################################################
    plt.figure(figsize=(10, 8))
    ticks = ["Baseline", "Conventional\nGridSearch", "GA\nGridSearch"]
    bpl = plt.boxplot([m1_5stats,m2_5stats,m3_5stats], positions=np.array(range(3))*2.0, widths=0.6, whis=(0, 100))
    plt.scatter(np.array(range(3))*2.0, [m1_mean, m2_mean, m3_mean],c='firebrick', marker='*', s=100)


    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=14)
    # plt.yticks(fontsize=14)
    plt.xlim(-2, len(ticks)*2)
    #plt.ylim(0, 1)
    # plt.xlabel("binWidth", fontsize=18)
    plt.ylabel("AUROC", fontsize=18)

    # statistical annotation
    positions=list(np.array(range(2))*2.0)
    x1, x2 = positions[0], positions[1]
    col = "midnightblue"
    plt.plot([x1, x1, x2, x2], [0.978, 0.981, 0.981, 0.9807], lw=1.5, c=col)
    plt.text((x1+x2)*.5, 0.981, "p-value="+"{:.2E}".format(pval12), ha='center', va='bottom', color=col)

    positions=list((np.array(range(2))+1)*2.0)
    x1, x2 = positions[0], positions[1]
    col = "midnightblue"
    plt.plot([x1, x1, x2, x2], [0.982, 0.983, 0.983, 0.9825], lw=1.5, c=col)
    plt.text((x1+x2)*.5, 0.983, "p-value="+"{:.2E}".format(pval23), ha='center', va='bottom', color=col)

    plt.tight_layout()
    # #plt.savefig('binWidths.svg')

if __name__ == "__main__":
    # load the saved AUCs
    baseline_aucs = pickle.load(open("../data/catboost_aucs_baseline.p", "rb"))
    catboost_aucs_GridSearch = pickle.load(open("../data/catboost_aucs_GridSearch.p", "rb"))
    catboost_aucs_GA = pickle.load(open("../data/catboost_aucs_GA.p", "rb"))
    best_params_list_GridSearch = pickle.load(open("../data/best_params_list_GridSearch.p", "rb"))

    min_baseline, q1_baseline, med_baseline, q3_baseline, max_baseline = five_number_summary(baseline_aucs)
    mean_baseline = np.mean(baseline_aucs)
    sd_baseline = np.std(baseline_aucs)
    min_cb, q1_cb, med_cb, q3_cb, max_cb = five_number_summary(catboost_aucs_GridSearch)
    mean_cb = np.mean(catboost_aucs_GridSearch)
    sd_cb = np.std(catboost_aucs_GridSearch)
    min_cbga, q1_cbga, med_cbga, q3_cbga, max_cbga = five_number_summary(catboost_aucs_GA)
    mean_cbga = np.mean(catboost_aucs_GA)
    sd_cbga = np.std(catboost_aucs_GA)

    print("baseline stats (mean, sd, 5stats):", [mean_baseline, sd_baseline, min_baseline, q1_baseline, med_baseline, q3_baseline, max_baseline])
    print("conventional gridsearch stats (mean, sd, 5stats):", [mean_cb, sd_cb, min_cb, q1_cb, med_cb, q3_cb, max_cb])
    print("GA gridsearch stats (mean, sd, 5stats):", [mean_cbga, sd_cbga, min_cbga, q1_cbga, med_cbga, q3_cbga, max_cbga])

    plot_boxes(mean_baseline, sd_baseline, [min_baseline, q1_baseline, med_baseline, q3_baseline, max_baseline],
               mean_cb, sd_cb, [min_cb, q1_cb, med_cb, q3_cb, max_cb],
               mean_cbga, sd_cbga, [min_cbga, q1_cbga, med_cbga, q3_cbga, max_cbga])

    array_best_params = [np.array(list(best_params_list_GridSearch[i].values())) for i in range(len(best_params_list_GridSearch))]
    unique, counts = np.unique(array_best_params, axis=0, return_counts=True)
    for i in range(unique.shape[0]):
        print("param grid", unique[i], "has frequecy of", counts[i])
