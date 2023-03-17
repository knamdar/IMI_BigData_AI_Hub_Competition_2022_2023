#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 Created in Mar 2023

Team 44
2022-23 IMI BIGDataAIHUB Case Competition

@author: Ernest (Khashayar) Namdar
"""

from sklearn import model_selection
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd #replace with cudf

class ScoringAUC():
    """Score AUC for multiclass problems.
    Average of one against all.
    """
    def __call__(self, clf, X, y, **kwargs):

        # # Generate predictions
        y_pred = clf.predict_proba(X)
        # if hasattr(clf, 'decision_function'):
        #     y_pred = clf.decision_function(X)
        # elif hasattr(clf, 'predict_proba'):
        #     y_pred = clf.predict_proba(X)
        # else:
        #     y_pred = clf.predict(X)

        # score
        # classes = set(y)
        # if y_pred.ndim == 1:
        #     y_pred = y_pred[:, np.newaxis]

        # _score = list()
        # for ii, this_class in enumerate(classes):
        #     _score.append(roc_auc_score(y == this_class,
        #                                 y_pred[:, ii]))
        #     if (ii == 0) and (len(classes) == 2):
        #         _score[0] = 1. - _score[0]
        #         break
        auc = roc_auc_score(y, y_pred, multi_class="ovr")
        return auc#np.mean(_score, axis=0)


class CatboostClassification:

    NUM_FOLDS = 4

    def __init__(self, X, y, randomSeed):

        self.randomSeed = randomSeed
        self.X = X
        self.y = y
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS)#, random_state=self.randomSeed)

    # CatBoost [learning_rate, iterations, depth]:
    # "learning_rate": float
    # "iterations": integer
    # "depth": integer
    def convertParams(self, params):
        learning_rate = params[0]        # no conversion needed
        iterations = round(params[1])  # round to nearest integer
        depth = round(params[2])  # round to nearest integer
        return learning_rate, iterations, depth

    def getAccuracy(self, params):
        learning_rate, iterations, depth = self.convertParams(params)
        self.classifier = CatBoostClassifier(random_state=self.randomSeed,
                                             learning_rate=learning_rate,
                                             iterations=iterations,
                                             depth=depth,
                                             silent=True
                                             )

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring=ScoringAUC())#"roc_auc")#roc_auc_score(multi_class="ovr"))
        print("results", cv_results)
        return cv_results.mean()

    def formatParams(self, params):
        return "'learning_rate'=%1.3f, 'iterations'=%3d, 'depth'=%3d" % (self.convertParams(params))
