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


import pandas as pd #replace with cudf
import numpy as np
import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import dgl
from dgl.data import DGLDataset



def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = (df_scaled[column]-df_scaled[column].mean())/df_scaled[column].std()
    return df_scaled



class ScotiaDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="ScotiaDataset")

    def process(self):
        risk_sheet_path = "../data/preprocessed_risk_sheet.xlsx"
        risk_sheet_df = pd.read_excel(risk_sheet_path)

        edges_sheet_path = "../data/UofT_edges.csv"
        edges_data = pd.read_csv(edges_sheet_path)
        edges_data = edges_data.astype({'source':'int', 'target':'int'})

        risk_sheet_df = risk_sheet_df.astype({'CUSTOMER_ID':'int', 'TargetLabel':'int'})

        y = risk_sheet_df['TargetLabel']
        to_drop = ["RISK", "NAME", 'BIRTH_DT', 'CUST_ADD_DT', "TargetLabel"] #we are skipping two DTs which is not ok
        nodes_data = risk_sheet_df.drop(to_drop, axis=1)
        self.colnames = nodes_data.columns
        nodes_data = maximum_absolute_scaling(nodes_data)

        #print(nodes_data.keys())
        #print(nodes_data[:, 1:].shape)
        node_features = torch.from_numpy(nodes_data.drop(["CUSTOMER_ID"], axis=1).to_numpy())
        node_labels = torch.from_numpy(y.to_numpy())

        edge_features = torch.from_numpy((edges_data["emt"]/edges_data["emt"].max()).to_numpy())
        edges_src = torch.from_numpy(edges_data["source"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["target"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )
        self.graph.ndata["feat"] = node_features.float()
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = edge_features.float()

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


if __name__ == "__main__":
    dataset = ScotiaDataset()
    graph = dataset[0]
    labels=dataset.graph.ndata["label"]
    print(graph)
    graph.ndata["feat"] = torch.nan_to_num(graph.ndata["feat"], nan=0.0)

    num_classes = 3
    seed=0
    N = len(labels) #Number of customers
    inds = np.array([i for i in range(N)])
    initial_filter = (labels!=4).tolist()
    sub_nodes = inds[initial_filter]
    sub_labels=labels[np.isin(labels, range(num_classes))]

    sub_g=graph.subgraph(sub_nodes)
    print(sub_g)

    # print parent node IDs
    parent_nodes=sub_g.ndata[dgl.NID]
    print("Parent node IDs: {}".format(parent_nodes))

