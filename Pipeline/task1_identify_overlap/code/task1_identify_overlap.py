#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Dec 2022

Team 44
2022-23 IMI BIGDataAIHUB Case Competition
"""

# Importing the required libraries ############################################
"""
Note: We could accelerate the operations using cuDF from RAPIDS
Nonetheless, team members had installation issues depending on the platform they used
"""
import pandas as pd


def count_digits(string):
    return sum(item.isdigit() for item in string)


if __name__ == "__main__":
    bad_actors_v2_path = "../data/bad_actors_v2.csv"
    bad_actors_v2_df = pd.read_csv(bad_actors_v2_path)

    bad_actors_path = "../data/bad_actors.csv"
    bad_actors_df = pd.read_csv(bad_actors_path)

    overlap_df = pd.merge(bad_actors_v2_df, bad_actors_df, on=["CUSTOMER_ID"], suffixes=["_v2", "_v1"])
    print("overlap of v1 and v2 is:", overlap_df.shape[0], "cases")
    print(overlap_df.CUSTOMER_ID.values.astype(int))

