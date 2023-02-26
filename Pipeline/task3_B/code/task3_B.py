#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Feb 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition
@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF and cuML from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""

from dataset import ScotiaDataset
from engine import train
import numpy as np
from sklearn.model_selection import train_test_split
from models import NGNN_GCN, BuiltinGCN, SimpleGraphNet, GCN
import os
import torch
import dgl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from cf_matrix import make_confusion_matrix
os.environ['DGLBACKEND'] = 'pytorch'
from sklearn.metrics import precision_recall_fscore_support as score


if __name__ == "__main__":
    dataset = ScotiaDataset()
    graph = dataset[0]
    labels=dataset.graph.ndata["label"]
    print(graph)
    graph.ndata["feat"] = torch.nan_to_num(graph.ndata["feat"], nan=0.0)
    #The following lines are not needed anymore because they are implemented in dataset
    #sub_g.edata["weight"] = sub_g.edata["weight"].abs()
    #sub_g.ndata["feat"] = sub_g.ndata["feat"].abs()

    num_classes = 2
    seed=0
    N = len(labels) #Number of patients
    inds = np.array([i for i in range(N)])
    initial_filter = (labels!=4).tolist()
    sub_nodes = inds[initial_filter]
    sub_labels=labels[np.isin(labels, range(num_classes))]
    
    dev_idx, test_idx, y_dev, y_test = train_test_split(sub_nodes, sub_labels, test_size=0.2, random_state=seed)
    train_idx, val_idx, y_train, y_val = train_test_split(dev_idx, y_dev, test_size=0.25, random_state=seed)

    # get subgraph and make bidirectional/undirected
    sub_g=graph.subgraph(sub_nodes)
    #sub_g=dgl.to_bidirected(sub_g, copy_ndata=True)
    print(sub_g)

    # print parent node IDs
    parent_nodes=sub_g.ndata[dgl.NID]
    print("Parent node IDs: {}".format(parent_nodes))

    # sample subset for the train, valid, and test set using parent_nodes
    train_mask = []
    valid_mask = []
    test_mask = []
    for idx in parent_nodes.tolist():
        if idx in train_idx:
            train_mask.append(True)
        else:
            train_mask.append(False)
        if idx in val_idx:
            valid_mask.append(True)
        else:
            valid_mask.append(False)
        if idx in test_idx:
            test_mask.append(True)
        else:
            test_mask.append(False)

    print("{} nodes for training: \n{} nodes for validation: \n{} nodes for testing. ".format(sum(train_mask), sum(valid_mask), sum(test_mask)))

    # # #####################BuiltinGCN##########################################
    # # instantiate GNN model using built-in GraphConv layers
    # model=BuiltinGCN(sub_g.ndata['feat'].shape[1], 32, len(sub_labels.unique()))

    # # add self-loop to ensure nodes consider their own features
    # sub_g=dgl.add_self_loop(sub_g)
    
    # # print model architecture
    # print(model)
    
    # # start training
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    # num_epochs = 150
    # if torch.cuda.is_available():
    #     cuda_g = sub_g.to('cuda:0')
    #     indices, lbls = train(model.to('cuda:0'), optimizer, num_epochs, cuda_g, sub_labels.to('cuda:0'), train_mask, valid_mask, test_mask)
    # else:
    #     indices, lbls = train(model, optimizer, num_epochs, sub_g, sub_labels, train_mask, valid_mask, test_mask)


    # cm = confusion_matrix(lbls, indices)
    # make_confusion_matrix(cm, group_names=['low', 'medium', 'high'])
    # plt.show()
    # precision, recall, fscore, support = score(lbls, indices)
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))
    # # #########################################################################


    # #######################NGNN_GCN##########################################
    model=NGNN_GCN(sub_g.ndata['feat'].shape[1], 64, len(sub_labels.unique()))
    
    # add self-loop to ensure nodes consider their own features
    sub_g=dgl.add_self_loop(sub_g)
    
    # print model architecture
    print(model)
    
    # start training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    num_epochs = 150
    if torch.cuda.is_available():
        cuda_g = sub_g.to('cuda:0')
        indices, lbls = train(model.to('cuda:0'), optimizer, num_epochs, cuda_g, sub_labels.to('cuda:0'), train_mask, valid_mask, test_mask)
    else:
        indices, lbls = train(model, optimizer, num_epochs, sub_g, sub_labels, train_mask, valid_mask, test_mask)



    cm = confusion_matrix(lbls, indices)
    make_confusion_matrix(cm, group_names=['low', 'medium', 'high'])
    plt.show()
    precision, recall, fscore, support = score(lbls, indices)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    # #########################################################################
