#!/usr/bin/env python3

# This script preprocesses bad actors database and customer dataset, samples it
# And then runs fuzzwuzzy's partial_token_sort_ratio on the samples datasets
# This script takes two command line arguments corresponding,
# arg1 = directory location of the bad actors database (targets.simple.csv)
# arg2 = directory location of the customer dataset (UofT_nodes.csv)
# arg3 = This is an optional third argument to indicate whether or not to sample,
# if arg3 = 'false', script will be run on full datasets, otherwise indicate ONE
# integer value which to sample down the two datasets


# Get stuff
import pandas as pd
import numpy as np
from unidecode import unidecode
from rapidfuzz import fuzz
import time
import sys

# get the start time
start = time.time()

# Function to preproces badDf
def ppBadDf (badDf, mergeAliases = True):

    # Get only person schema from badDf
    badDf_sub = badDf.loc[badDf["schema"]=="Person"]
    # Truncate to columns of interest, drop nas,  drop duplicates
    badDf_sub = badDf_sub[["id","name","aliases","birth_date"]].dropna(subset=['name']).drop_duplicates(subset =['name'])
    # Clean names: convert all non-english characters, replace symbols, lowercase for easier comparison
    badDf_sub['name'] = badDf_sub['name'].apply(unidecode).str.replace(r'[^\w\s]', '', regex = True).str.lower()
    # Get rid of person name from badDf
    badDf_sub = badDf_sub.loc[badDf_sub["name"]!="person"]
    # Clean aliases: Fill nas, replace seperator ; with a gap, convert non-english characters, replace symbols, lower case
    badDf_sub['aliases'] = badDf_sub['aliases'].fillna('').str.replace(';', ' ').apply(unidecode).str.replace(r'[^\w\s]', '', regex = True).str.lower()
    # if statement for merging aliases with names
    if mergeAliases == True:
        # Merge names and aliase
        badDf_sub["name"] = badDf_sub["name"] + badDf_sub["aliases"]
        # Remove aliases
        badDf_sub = badDf_sub.drop('aliases', axis=1)

        return(badDf_sub)

    elif mergeAliase == False:

        return(badDf_sub)

# Function to preprocess custDf
def ppCustDf (custDf, mergeAliases = True):

    # Truncate to columns of interest, drop nas
    custDf_sub = custDf[["CUSTOMER_ID", "NAME", "BIRTH_DT", "GENDER"]].dropna(subset=['NAME']).drop_duplicates(subset =['NAME'])
    # Clean names: convert all non-english characters, replace symbols, lowercase for easier comparison
    custDf_sub['NAME'] = custDf_sub['NAME'].apply(unidecode).str.replace(r'[^\w\s]', '', regex = True).str.lower()
    # Rename col names to be consistent with bad df
    custDf_sub = custDf_sub.rename(columns={'CUSTOMER_ID': 'id', 'NAME': 'name', 'BIRTH_DT': 'birth_date', 'GENDER': 'gender'})


    return(custDf_sub)

# Function to get random name samples from dataframe
def getSample(df, X = 200):
    # Spit out random X elements from dataframes
    rand = np.random.choice(np.arange(1, df.shape[0]), size=X)
    df_rand = df.iloc[rand,]

    return(df_rand)

# Function to run fuzz with partial_token_sort_ratio
def runFuzz(df1, df2):
    A = df1["name"].tolist()
    B = df2["name"].tolist()
    C = df1["gender"].tolist()
    D = df1["birth_date"].tolist()
    E = df2["birth_date"].tolist()

    indA = range(0,len(A))
    indB = range(0,len(B))

    # Use numpy's meshgrid function to create a grid of all pairs of elements in A and B
    A_grid, B_grid = np.meshgrid(A, B)
    indA_grid, indB_grid = np.meshgrid(indA, indB)

    # Use vectorized function to compute ratios for all pairs
    ratios = np.vectorize(fuzz.partial_token_sort_ratio)(A_grid, B_grid)

    indA_flat = indA_grid.flatten()
    indB_flat = indB_grid.flatten()


    df = pd.DataFrame()
    df = df.assign(custName=A_grid.flatten(),
                    badName = B_grid.flatten(),
                    ratio = ratios.flatten(),
                    custGender = [C[i] for i in indA_flat],
                    custBirth = [D[i] for i in indA_flat],
                    badBirth = [E[i] for i in indB_flat])

    df_sorted = df.sort_values(by='ratio', ascending=False)

    # Print top 10 matches
    print(df_sorted.iloc[1:10,:])

# Load bad actors csv file 'targets.simple.csv'
badDf = pd.read_csv(sys.argv[1], low_memory=False)

# Load customer csv file 'UofT_nodes.csv'
custDf = pd.read_csv(sys.argv[2])

# Get preprocessed subsetted badDf
badDf = ppBadDf(badDf, mergeAliases = True)

# Get preprocessed subsetted custDf
custDf = ppCustDf(custDf)

# To sample or not to sample
if len(sys.argv) == 4:
    N = sys.argv[3]
    if N.isnumeric():
        custDf = getSample(custDf, X = N)
        badDf = getSample(badDf, X = N)
    elif N == "false":
        runFuzz(custDf, badDf)
elif len(sys.argv) == 3:
    custDf = getSample(custDf, X = 200)
    badDf = getSample(badDf, X = 200)
    runFuzz(custDf, badDf)

# get the end time
end = time.time()

# get the execution time
elapsed = end - start

print('Exec time:', elapsed, 'seconds')
