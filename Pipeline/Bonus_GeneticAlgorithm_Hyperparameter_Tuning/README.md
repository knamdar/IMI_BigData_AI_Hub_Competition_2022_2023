# Genetic-Algorithm-based Hyperparameter Tuning

1. code:
    + elitism.py: utility file for implementing elitism for GA
    + Result_Visualization.py: utility file for analyzing and plotting the results
    + hyperparameter_tuning_genetic_test.py: creating the CatBoost classifier class for GA optimization context
    + GA_grid_search.py: The main code for Genetic-Algorithm-based Hyperparameter Tuning
    + conventional_grid_search.py: conventional grid search
    + baseline.py: running the model with default hyperparameters
2. data: The directory for storing the data
    + risk_sheet.xlsx (from task1)
    + catboost_aucs_baseline.p: pickle file from baseline.py (should be manually moved from results to data)
    + catboost_aucs_GridSearch.p: pickle file from conventional_grid_search.py (should be manually moved from results to data)
    + catboost_aucs_GA.p: pickle file from GA_grid_search.py (should be manually moved from results to data)
    + best_params_list_GridSearch.p: pickle file from conventional_grid_search.py (should be manually moved from results to data)
3. results: The directory for storing the results and logs
