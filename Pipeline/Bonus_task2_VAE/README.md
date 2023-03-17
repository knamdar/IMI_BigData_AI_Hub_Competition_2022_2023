Task2_AIHub.ipynb is a Jupyter Notebook that can be run to train 5 models for risk classification. Task2_AIHub_Evaluation.ipynb evaluates the performance of the saved models, outputting the AUROCs and confusion matrix statistics for each model. Both of these notebooks requires the root_path variable to be filled in with the absolute path to the directory where the code is being run.

This directory should include a subdirectory named "data" that contains risk_sheet.xlsx.

The trained model is found in task2_model.pt.