#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Mar 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition

@author: Ernest (Khashayar) Namdar

Inspired by:
Hands-On Genetic Algorithms with Python
Applying genetic algorithms to solve real-world deep learning and artificial intelligence problems
by Eyal Wirsansky
"""
from deap import base
from deap import creator
from deap import tools

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import hyperparameter_tuning_genetic_test
import elitism
import pandas as pd #replace with cudf
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import time
import pickle

start = time.time()

def balance_dset(df, target, seed):
    np.random.seed(seed)
    high_risk_data = df[df[target]==2].copy()
    labels = df[target].copy()
    all_mid_risk = labels[labels == 1]
    all_low_risk = labels[labels == 0]
    mid_risk_to_keep = np.random.choice(all_mid_risk.index, size=high_risk_data.shape[0], replace=False)
    low_risk_to_keep = np.random.choice(all_low_risk.index, size=high_risk_data.shape[0], replace=False)
    mid_risk_data = df.iloc[mid_risk_to_keep].copy()
    low_risk_data = df.iloc[low_risk_to_keep].copy()
    new_df = pd.concat([high_risk_data, mid_risk_data, low_risk_data], axis=0)
    return new_df


# boundaries for ADABOOST parameters:
# "n_estimators": 1..100
# "learning_rate": 0.01..100
# "algorithm": 0, 1
# [n_estimators, learning_rate, algorithm]:
BOUNDS_LOW =  [0.01, 1900, 4.51]
BOUNDS_HIGH = [0.05, 2500, 7.49]

NUM_OF_PARAMS = len(BOUNDS_HIGH)

# Genetic Algorithm constants:
POPULATION_SIZE = 20
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.5   # probability for mutating an individual
MAX_GENERATIONS = 5
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
THRESHOLD = 0.2

risk_sheet_path = "../data/risk_sheet.xlsx" # Loading the preprocessed spreadsheet of features and ground truths
risk_sheet_df = pd.read_excel(risk_sheet_path)

# quantizing the features
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
# separating year, month, and day of DOBs andwhen  customers joined the bank as standalone features
risk_sheet_df['BIRTH_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).year
risk_sheet_df['BIRTH_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).month
risk_sheet_df['BIRTH_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['BIRTH_DT']).day
risk_sheet_df['CUST_ADD_DT_YEAR'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).year
risk_sheet_df['CUST_ADD_DT_MONTH'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).month
risk_sheet_df['CUST_ADD_DT_DAY'] = pd.DatetimeIndex(risk_sheet_df['CUST_ADD_DT']).day
balanced_risk_sheet_df = balance_dset(risk_sheet_df, "RISK", RANDOM_SEED) # make the dataset balanced
y = balanced_risk_sheet_df['RISK']
to_drop = ["RISK", "CUSTOMER_ID", "NAME", 'BIRTH_DT', 'CUST_ADD_DT']
X = balanced_risk_sheet_df.drop(to_drop, axis=1)


def ga_grid_search_experiment(X, y, RANDOM_SEED):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
    X_dev, _, y_dev, _ = train_test_split(X_temp, y_temp, test_size=1-THRESHOLD, random_state=RANDOM_SEED)
    # create the classifier accuracy test class:
    test = hyperparameter_tuning_genetic_test.CatboostClassification(X_dev, y_dev, RANDOM_SEED)
    
    toolbox = base.Toolbox()
    
    # define a single objective, maximizing fitness strategy:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    # create the Individual class based on list:
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # define the hyperparameter attributes individually:
    for i in range(NUM_OF_PARAMS):
        # "hyperparameter_0", "hyperparameter_1", ...
        toolbox.register("hyperparameter_" + str(i),
                         random.uniform,
                         BOUNDS_LOW[i],
                         BOUNDS_HIGH[i])
    
    # create a tuple containing an attribute generator for each param searched:
    hyperparameters = ()
    for i in range(NUM_OF_PARAMS):
        hyperparameters = hyperparameters + \
                          (toolbox.__getattribute__("hyperparameter_" + str(i)),)
    
    # create the individual operator to fill up an Individual instance:
    toolbox.register("individualCreator",
                     tools.initCycle,
                     creator.Individual,
                     hyperparameters,
                     n=1)
    
    # create the population operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    
    # fitness calculation
    def classificationAccuracy(individual):
        return test.getAccuracy(individual),
    
    toolbox.register("evaluate", classificationAccuracy)
    
    # genetic operators:mutFlipBit
    
    # genetic operators:
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate",
                     tools.cxSimulatedBinaryBounded,
                     low=BOUNDS_LOW,
                     up=BOUNDS_HIGH,
                     eta=CROWDING_FACTOR)
    
    toolbox.register("mutate",
                     tools.mutPolynomialBounded,
                     low=BOUNDS_LOW,
                     up=BOUNDS_HIGH,
                     eta=CROWDING_FACTOR,
                     indpb=1.0 / NUM_OF_PARAMS)
    
    
    # Genetic Algorithm flow:
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    
    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    
    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)
    
    # print best solution found:
    print("- Best solution is: ")
    print("params = ", test.formatParams(hof.items[0]))
    print("Accuracy = %1.5f" % hof.items[0].fitness.values[0])
    
    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    
    # # plot statistics:
    # sns.set_style("whitegrid")
    # plt.plot(maxFitnessValues, color='red')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel('Generation')
    # plt.ylabel('Max / Average Fitness')
    # plt.title('Max and Average fitness over Generations')
    # plt.show()
    
    list_of_paramstrings = test.formatParams(hof.items[0]).replace(" ", "").split(",")
    list_of_params = []
    for item in list_of_paramstrings:
        lst = item.split("=")
        list_of_params.append(lst[1])
    
    best_params = {"learning_rate":float(list_of_params[0]),
                   "iterations":int(list_of_params[1]),
                   "depth":int(list_of_params[2]),
                   "silent":True}
    clf = CatBoostClassifier(**best_params)
    
    clf.fit(X_dev, y_dev)
    predictions = clf.predict_proba(X_test)
    return roc_auc_score(y_test, predictions, multi_class="ovr"), best_params

N = 30
experiment_AUCs = []
best_params_list = []
for i in range(N):
    print("working on experiment number", i+1)
    roc_auc, best_params = ga_grid_search_experiment(X, y, i)
    experiment_AUCs.append(roc_auc)
    best_params_list.append(best_params)

pickle.dump(experiment_AUCs, open("../results/experiment_AUCs_GA.p", "wb"))
pickle.dump(best_params_list, open("../results/best_hypers_GA.p", "wb"))


end = time.time()
duration = end-start

print("The run was completed in: ", int(duration/60), "minutes and ", int(duration%60), "seconds")