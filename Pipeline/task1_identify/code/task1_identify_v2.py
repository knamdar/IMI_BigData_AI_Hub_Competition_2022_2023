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
import numpy as np
from datetime import datetime


def count_digits(string):
    return sum(item.isdigit() for item in string)


if __name__ == "__main__":
    open_data_path = "../data/opensanctions.csv"
    open_Data_df = pd.read_csv(open_data_path)
    open_Data_target_df = open_Data_df[open_Data_df.columns[open_Data_df.columns.isin(["name", "birth_date"])]]
    open_Data_target_df = open_Data_target_df.rename(columns={"name": "NAME", "birth_date": "BIRTH_DT"})

    uoft_nodes_path = "../data/UofT_nodes.csv"
    uoft_nodes_df = pd.read_csv(uoft_nodes_path)
    uoft_target_df = uoft_nodes_df[uoft_nodes_df.columns[uoft_nodes_df.columns.isin(["BIRTH_DT", "NAME", "CUSTOMER_ID"])]]

    matched_nameDates_from_uoft = pd.merge(uoft_target_df, open_Data_target_df, on=["NAME", "BIRTH_DT"], suffixes=["", "_openData"])
    matched_names_from_uoft = pd.merge(uoft_target_df, open_Data_target_df, on=['NAME'], suffixes=["", "_openData"])
    matched_names_from_uoft['BIRTH_DT']= pd.to_datetime(matched_names_from_uoft['BIRTH_DT'], format='%Y %m %d' )
    for i in range(matched_names_from_uoft.shape[0]):
        dob = np.nan
        if pd.isna(matched_names_from_uoft['BIRTH_DT_openData'][i]):
            continue
        try:
            dob = datetime.strptime(matched_names_from_uoft['BIRTH_DT_openData'][i], '%Y-%m-%d')
            #print()
            #print("OK")
        except:
            try:
                dob = datetime.strptime(matched_names_from_uoft['BIRTH_DT_openData'][i], '%Y')
            except:
                try:
                    dob = datetime.strptime(matched_names_from_uoft['BIRTH_DT_openData'][i], '%Y-%m')
                except:
                    #print("error. We set dob to nan to red flag the customer")
                    print(matched_names_from_uoft['BIRTH_DT_openData'][i])
                    dob = matched_names_from_uoft['BIRTH_DT_openData'][i].split(';')
        matched_names_from_uoft['BIRTH_DT_openData'][i] = dob

    matched_names_from_uoft.insert(0, "bad_actor", 0)
    unique_names = list(np.unique(matched_names_from_uoft["NAME"]))
    print("Number of unique names in matched_names_from_uoft:", len(unique_names))
    for i in range(matched_names_from_uoft.shape[0]):
        matched_names_from_uoft["BIRTH_DT"].iloc[i]=np.datetime64(matched_names_from_uoft["BIRTH_DT"].iloc[i])
        if type(matched_names_from_uoft["BIRTH_DT_openData"].iloc[i])==list:
            for j in range(len(matched_names_from_uoft["BIRTH_DT_openData"].iloc[i])):
                (matched_names_from_uoft["BIRTH_DT_openData"].iloc[i])[j] = np.datetime64((matched_names_from_uoft["BIRTH_DT_openData"].iloc[i])[j])
            print(matched_names_from_uoft["BIRTH_DT_openData"].iloc[i])
        elif pd.isna(matched_names_from_uoft["BIRTH_DT_openData"].iloc[i]):
            matched_names_from_uoft["BIRTH_DT_openData"].iloc[i] = np.datetime64("NaT")
        else:
            matched_names_from_uoft["BIRTH_DT_openData"].iloc[i]=np.datetime64(matched_names_from_uoft["BIRTH_DT_openData"].iloc[i])
    for name in unique_names:
        filtered_df = matched_names_from_uoft[matched_names_from_uoft.NAME==name]
        index_list = filtered_df.index.values.tolist()
        for ind in index_list:
            if type(filtered_df.loc[[ind],"BIRTH_DT_openData"].values[0])==list:
                for date in filtered_df.loc[[ind],"BIRTH_DT_openData"].values[0]:
                    if (filtered_df.loc[[ind],"BIRTH_DT"].values[0].astype('datetime64[M]')==date.astype('datetime64[M]')):
                        #print("dates matched")
                        matched_names_from_uoft.loc[[ind]]["bad_actor"] = 1
            else:
                if pd.isna(filtered_df.loc[[ind],"BIRTH_DT_openData"]).values[0]:
                    matched_names_from_uoft["bad_actor"].iloc[[ind]]=1
                if (filtered_df.loc[[ind],"BIRTH_DT"].values[0].astype('datetime64[M]')==filtered_df.loc[[ind],"BIRTH_DT_openData"].values[0].astype('datetime64[M]')):
                    #print("dates matched")
                    matched_names_from_uoft.loc[[ind]]["bad_actor"] = 1

    print("Number bad actors found:", sum(matched_names_from_uoft["bad_actor"]))
    Bad_Actors = matched_names_from_uoft[matched_names_from_uoft["bad_actor"]==1]
    Bad_Actors = Bad_Actors.drop_duplicates('CUSTOMER_ID', keep='last')
    Bad_Actors = matched_nameDates_from_uoft.append(Bad_Actors)
    Bad_Actors = Bad_Actors[Bad_Actors.columns[Bad_Actors.columns.isin(["NAME", "CUSTOMER_ID"])]]
    Bad_Actors.to_csv("../results/bad_actors_v2.csv", index=False)