#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Dec 2022

Team 44
2022-23 IMI BIGDataAIHUB Case Competition
@author: Ernest (Khashayar) Namdar
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""
import pandas as pd
import faiss # Meta's library for efficient similarity search
from transformers import DistilBertTokenizer, DistilBertModel # working with pretrained models
from torch.nn import functional as F
import torch
import numpy as np


def convert_to_str(lst): # convert DOB and Name to string
    for i in range(len(lst)):
        if type(lst[i]) != str:
            lst[i] = str(lst[i])
    return lst


if __name__ == "__main__":
    open_data_path = "../data/opensanctions.csv"
    open_Data_df = pd.read_csv(open_data_path)
    vector_dimension = 768 #DistilBertModel
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    rows = []
    vectors = []
    # implementation idea: tokenizer and model can work batch-based

    # convering rows of the opensanctions dataset to embedding vectors
    for i in range(open_Data_df.shape[0]):
        if i%500 == 0:
            print("working on row", i+1, "of", open_Data_df.shape[0])
        row_str = ','.join(convert_to_str(list(open_Data_df.iloc[i].values))) # convert a row of the reference data to string
        rows.append(row_str)
        tokens = tokenizer(row_str, truncation=True, padding=True, return_tensors="pt")
        model_output = model(**tokens)
        vector = F.normalize(torch.mean(model_output.last_hidden_state, dim=1), dim=1).squeeze().detach().numpy()
        vectors.append(vector)
    vectors = np.array(vectors)

    # indexing the embeddings with FAISS
    index = faiss.IndexFlatIP(vector_dimension) #IP stands for "inner product". If you have normalized vectors, the inner product becomes cosine similarity
    index.add(vectors)

    # Loading UofT dataset and dropping all columns but DOB and Name
    uoft_nodes_path = "../data/UofT_nodes.csv"
    uoft_nodes_df = pd.read_csv(uoft_nodes_path)
    drop_list = ['CUST_ADD_DT', 'OCPTN_NM', 'RES_CNTRY_CA',
            'CNTRY_OF_INCOME_CA', 'PEP_FL', 'CASH_SUM_IN', 'CASH_CNT_IN',
            'CASH_SUM_OUT', 'CASH_CNT_OUT', 'WIRES_SUM_IN', 'WIRES_CNT_IN',
            'WIRES_SUM_OUT', 'WIRES_CNT_OUT', 'COUNTRY_RISK_INCOME',
            'COUNTRY_RISK_RESIDENCY', 'RISK', 'CUSTOMER_ID']
    # drop_list = ['RISK', 'CUSTOMER_ID'] # This approach did not help the performance
    uoft_features_df = uoft_nodes_df.drop(columns=drop_list)

    # convering customer name + DOB to embedding vectors
    customer_vectors = []
    for i in range(uoft_features_df.shape[0]):
        if i%500 == 0:
            print("working on row", i+1, "of", uoft_features_df.shape[0])
        row_str = ','.join(convert_to_str(list(uoft_features_df.iloc[i].values)))
        #print(row_str)
        tokens = tokenizer(row_str, truncation=True, padding=True, return_tensors="pt")
        model_output = model(**tokens)
        vector = F.normalize(torch.mean(model_output.last_hidden_state, dim=1), dim=1).squeeze().detach().numpy()
        customer_vectors.append(vector)
    customer_vectors = np.array(customer_vectors)

    # conducting teh efficient similarity search on the big data
    scores, indices = index.search(customer_vectors, 1) #1 means the top score
    # print("score is", scores[0][0])
    # print("score is", scores)

    # creating the final DF
    # bringing the occupation risks as a feature
    ocp_risk_path = "../data/UofT_occupation_risk.csv" # ocp stands for occupation
    ocp_risk_df = pd.read_csv(ocp_risk_path)
    ocp_risk_dict = dict(zip(ocp_risk_df["code"], ocp_risk_df["occupation_risk"])) #to be used for column transformation (OCPTN_NM -> OCPTN_RISK)
    risk_sheet = uoft_nodes_df.copy(deep=True)
    risk_sheet = risk_sheet[['CUSTOMER_ID', 'NAME', 'GENDER', 'BIRTH_DT',
                             'CUST_ADD_DT', 'OCPTN_NM', 'RES_CNTRY_CA',
                             'CNTRY_OF_INCOME_CA', 'PEP_FL', 'CASH_SUM_IN',
                             'CASH_CNT_IN', 'CASH_SUM_OUT', 'CASH_CNT_OUT',
                             'WIRES_SUM_IN', 'WIRES_CNT_IN', 'WIRES_SUM_OUT',
                             'WIRES_CNT_OUT', 'COUNTRY_RISK_INCOME',
                             'COUNTRY_RISK_RESIDENCY', 'RISK']]
    risk_sheet.insert(6, "OCPTN_RISK", "n") #insert the new column after 'OCPTN_NM'
    risk_sheet["OCPTN_RISK"] = risk_sheet["OCPTN_NM"].map(ocp_risk_dict)
    risk_sheet.insert(7, "QUERY_RISK", "n") # incorporating the calculated risks based on name and DOB query
    for i in range(uoft_features_df.shape[0]):
        risk_sheet["QUERY_RISK"][i] = scores[i][0]
    risk_sheet.to_csv("../results/task1.csv", index=False)
