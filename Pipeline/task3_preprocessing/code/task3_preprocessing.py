#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Created in Feb 2023

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


if __name__ == "__main__":
    risk_sheet_path = "../data/risk_sheet.xlsx"
    risk_sheet_df = pd.read_excel(risk_sheet_path)

    bad_actors_v2_path = "../data/bad_actors_v2.csv"
    bad_actors_v2_df = pd.read_csv(bad_actors_v2_path)
    bad_actors_v2_df = bad_actors_v2_df.drop(["NAME"], axis=1)

    overlap_df = pd.merge(risk_sheet_df, bad_actors_v2_df, how="inner", on=["CUSTOMER_ID"], suffixes=["", "_"])
    nonoverlap_df = risk_sheet_df[~risk_sheet_df["CUSTOMER_ID"].isin(overlap_df["CUSTOMER_ID"])]
    overlap_df.insert(0, "TargetLabel", "BadActor")
    nonoverlap_df.insert(0, "TargetLabel", "Normal")

    risk_sheet_df = overlap_df.append(nonoverlap_df)

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
    risk_sheet_df['BIRTH_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).year
    risk_sheet_df['BIRTH_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).month
    risk_sheet_df['BIRTH_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).day
    risk_sheet_df['CUST_ADD_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).year
    risk_sheet_df['CUST_ADD_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).month
    risk_sheet_df['CUST_ADD_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).day
    risk_sheet_df["TargetLabel"].replace(['Normal', 'BadActor'],
                                                  [0, 1], inplace=True)

    edges_sheet_path = "../data/UofT_edges.csv"
    edges_data = pd.read_csv(edges_sheet_path)
    edges_data = edges_data.astype({'source':'int', 'target':'int'})

    # Some ids available in edges_data are missing in the nodes. We manually create them and assign a unique group id to them
    current_id_list = list(risk_sheet_df["CUSTOMER_ID"])
    max_id = max(max(current_id_list), edges_data["source"].max(), edges_data["target"].max())
    for i in range(max_id+1):
        if not i in current_id_list:
            risk_sheet_df = risk_sheet_df.append(pd.Series(), ignore_index=True)
            risk_sheet_df.loc[risk_sheet_df.index[-1],:] = 0
            risk_sheet_df.loc[risk_sheet_df.index[-1],"RISK"] = 4
            risk_sheet_df.loc[risk_sheet_df.index[-1],"TargetLabel"] = 4
            risk_sheet_df.loc[risk_sheet_df.index[-1],"CUSTOMER_ID"] = i
    risk_sheet_df = risk_sheet_df.astype({'CUSTOMER_ID':'int', 'RISK':'int', 'TargetLabel':'int'})
    risk_sheet_df.to_excel("../results/preprocessed_risk_sheet.xlsx", index=False)
