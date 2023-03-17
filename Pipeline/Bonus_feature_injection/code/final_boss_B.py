#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4 Created in March 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition
@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF and cuML from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""

from dataset_B import ScotiaDataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from sklearn.metrics import confusion_matrix
from cf_matrix import make_confusion_matrix
os.environ['DGLBACKEND'] = 'pytorch'
from catboost import CatBoostClassifier
HYPERS = {'learning_rate': 0.03, 'iterations': 2000, 'depth': 6, 'silent':True}
import pandas as pd
from sklearn.metrics import roc_auc_score


def balance_graph(lbls, node_ids):
    labels = np.array(lbls.tolist())
    nids = np.array(node_ids.tolist())
    high_risk_nids = nids[labels==1]
    low_risk_nids = nids[labels==0]
    low_risk_to_keep = np.random.choice(low_risk_nids, size=high_risk_nids.shape[0], replace=False)
    balanced_nodeids = np.concatenate((high_risk_nids, low_risk_to_keep), axis=0)
    balanced_lbls = np.concatenate((np.ones(high_risk_nids.shape[0]), np.zeros(high_risk_nids.shape[0])), axis=0)
    return balanced_nodeids, balanced_lbls.astype(np.int64)


if __name__ == "__main__":
    seed=14
    HYPERS["random_state"] = seed
    dataset = ScotiaDataset()
    whole_dataset_graph = dataset[0]
    whole_dataset_labels=dataset.graph.ndata["label"]
    print(whole_dataset_graph)
    whole_dataset_graph.ndata["feat"] = torch.nan_to_num(whole_dataset_graph.ndata["feat"], nan=0.0)
    #The following lines are not needed anymore because they are implemented in dataset
    #sub_g.edata["weight"] = sub_g.edata["weight"].abs()
    #sub_g.ndata["feat"] = sub_g.ndata["feat"].abs()

    num_classes = 3

    N = len(whole_dataset_labels) #Number of patients
    inds = np.array([i for i in range(N)])
    initial_filter = (whole_dataset_labels!=4).tolist()
    no_lbl4_nodes = inds[initial_filter]
    no_lbl4_labels=whole_dataset_labels[np.isin(whole_dataset_labels, range(num_classes))]


    # get subgraph and make bidirectional/undirected
    no_lbl4_graph=whole_dataset_graph.subgraph(no_lbl4_nodes)
    #sub_g=dgl.to_bidirected(sub_g, copy_ndata=True)
    print(no_lbl4_graph)

    balanced_nodeids, balanced_lbls = balance_graph(no_lbl4_labels, no_lbl4_nodes)
    balanced_graph=whole_dataset_graph.subgraph(balanced_nodeids)
    print(balanced_graph)

    dev_idx, test_idx, y_dev, y_test = train_test_split(balanced_nodeids, balanced_lbls, test_size=0.25, random_state=seed)
    dev_graph = whole_dataset_graph.subgraph(dev_idx)
    test_graph = whole_dataset_graph.subgraph(test_idx)
    #baseline experiment
    y_dev = np.array(dev_graph.ndata["label"])
    X_dev = np.array(dev_graph.ndata["feat"])
    X_dev = pd.DataFrame(X_dev, columns=dataset.colnames[1:]) #[1:] omits customerID
    y_test = np.array(test_graph.ndata["label"])
    X_test = np.array(test_graph.ndata["feat"])
    X_test = pd.DataFrame(X_test, columns=dataset.colnames[1:])


    clf = CatBoostClassifier(**HYPERS)
    clf.fit(X_dev, y_dev)
    predictions = clf.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, predictions)
    print("baseline_auc", baseline_auc)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    make_confusion_matrix(cm, group_names=['low', 'medium', 'high'])

    # #########################################################################
    test_idx_set = set(test_idx.tolist())
    whole_dataset_labels_npy = whole_dataset_labels.numpy()
    feature_1 = []
    feature_2 = []
    feature_3 = []
    feature_4 = []
    for i in range(inds.shape[0]):
        if i%10000==0:
            print("loop", i, "completed")
        list_succs = list(set(whole_dataset_graph.successors(i).tolist())-test_idx_set)
        list_preds = list(set(whole_dataset_graph.predecessors(i).tolist())-test_idx_set)
        if list(list_succs)==[]:
            feature_1.append(-1)
            feature_3.append(-1)
        else:
            feature_1.append(np.sum(whole_dataset_labels_npy[list_succs]))
            feature_3.append(np.max(whole_dataset_labels_npy[list_succs]))
        if list(list_preds)==[]:
            feature_2.append(-1)
            feature_4.append(-1)
        else:
            feature_2.append(np.sum(whole_dataset_labels_npy[list_preds]))
            feature_4.append(np.max(whole_dataset_labels_npy[list_preds]))
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    feature_3 = np.array(feature_3)
    feature_4 = np.array(feature_4)
    feature_1_dev = feature_1[dev_idx]
    feature_2_dev = feature_2[dev_idx]
    feature_3_dev = feature_3[dev_idx]
    feature_4_dev = feature_4[dev_idx]
    feature_1_test = feature_1[test_idx]
    feature_2_test = feature_2[test_idx]
    feature_3_test = feature_3[test_idx]
    feature_4_test = feature_4[test_idx]
    X_dev["InjectedFeature1"] = feature_1_dev
    X_dev["InjectedFeature2"] = feature_2_dev
    X_dev["InjectedFeature3"] = feature_3_dev
    X_dev["InjectedFeature4"] = feature_4_dev
    X_test["InjectedFeature1"] = feature_1_test
    X_test["InjectedFeature2"] = feature_2_test
    X_test["InjectedFeature3"] = feature_3_test
    X_test["InjectedFeature4"] = feature_4_test
    clf = CatBoostClassifier(**HYPERS)
    clf.fit(X_dev, y_dev)
    predictions = clf.predict_proba(X_test)[:, 1]
    new_auc = roc_auc_score(y_test, predictions)
    print("new_auc", new_auc)
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    make_confusion_matrix(cm, group_names=['low', 'medium', 'high'])
