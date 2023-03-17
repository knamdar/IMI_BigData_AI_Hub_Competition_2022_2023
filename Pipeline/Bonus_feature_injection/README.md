# Feature Injection
1. code:
    + cf_matrix.py: utility file for plotting confusion matrices
    + dataset.py: creating a DGL graph to be used for data augmentation (from task3, for RISK classification)
    + dataset_B.py: creating a DGL graph to be used for data augmentation (from task3, for Bad Actor Identification)
    + final_boss.py: RISK classification using CatBoost using graph-augmented data
    + final_boss_B.py: Bad Actor Identification using CatBoost using graph-augmented data
2. data: The directory for storing the data
    + UofT_edges.csv
    + preprocessed_risk_sheet.xlsx: from task3 preprocessing
3. results: The directory for storing the results
