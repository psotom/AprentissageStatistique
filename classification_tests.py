from random_forest import *

import numpy as np
import pandas as pd

from sklearn.datasets import (
    make_classification,
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_openml
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --------------------------------------------------
# 2. Prepare Datasets
# --------------------------------------------------

def get_synthetic_datasets():
    datasets = []
    # 1) Basic classification with 2 classes
    X1, y1 = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,  # 5 truly informative
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42
    )
    datasets.append(('Synthetic_Classification', X1, y1))

    return datasets

def get_real_datasets():
    datasets = []
    # Iris
    iris = load_iris()
    datasets.append(('Iris', iris.data, iris.target))
    # Wine
    wine = load_wine()
    datasets.append(('Wine', wine.data, wine.target))
    # Breast Cancer
    cancer = load_breast_cancer()
    datasets.append(('BreastCancer', cancer.data, cancer.target))
    # Madelon https://www.openml.org/search?type=data&status=active&id=1485
    madelon_id = 1485
    madelon = fetch_openml(data_id=madelon_id, as_frame=False)
    datasets.append(('Madelon', madelon.data.astype(float), madelon.target.astype(int)))
    # Arcene https://www.openml.org/search?type=data&status=active&id=1458
    arcene_id = 1458
    arcene = fetch_openml(data_id=arcene_id, as_frame=False)
    datasets.append(('Arcene', arcene.data.astype(float), arcene.target.astype(int)))
    # Dexter https://www.openml.org/search?type=data&status=active&id=4136
    # dexter_id = 4136
    # dexter = fetch_openml(data_id=dexter_id, as_frame=False)
    # datasets.append(('Dexter', dexter.data.astype(float), dexter.target.astype(int)))
    return datasets

# --------------------------------------------------
# 3. Define Different Variations of Your Custom RandomForest
# --------------------------------------------------

def get_custom_rf_variations():
    """
    Return a dictionary of different RandomForest configurations.
    Adjust or add more as needed.
    """
    variations = {
        'RF_default': RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=2, n_features=None, bagging=True, split_criterion='gini'),
        'RF_simple': RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=2, n_features=None, bagging=False, split_criterion='middle')
    }
    return variations

# --------------------------------------------------
# 4. Experiment Function (Manual Cross-Validation)
# --------------------------------------------------

def run_experiments(
    datasets, 
    rf_variations, 
    cv_splits=5, 
    results_file="classification_results2.csv"
):
    """
    Runs Stratified K-Fold cross-validation for each dataset
    and each RF variation, returns a DataFrame of results, and
    saves the results to a CSV file.

    :param datasets: List of (dataset_name, X, y) tuples
    :param rf_variations: Dictionary: variation_name -> RF_model_instance
    :param cv_splits: Number of CV folds (default=5)
    :param results_file: CSV filename to save results (default='results.csv')
    :return: Pandas DataFrame of all results
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []

    # Calculate total iterations (for progress bar)
    total_tasks = len(datasets) * len(rf_variations) * cv_splits

    with tqdm(total=total_tasks, desc="Running Experiments") as pbar:
        for ds_name, X, y in datasets:
            print("Dataset: ", ds_name)
            for var_name, rf_model in rf_variations.items():
                scores = []
                for train_idx, test_idx in skf.split(X, y):

                    # Fit model
                    rf_model.fit(X[train_idx], y[train_idx])

                    # Predict
                    y_pred = rf_model.predict(X[test_idx])

                    # Evaluate
                    acc = accuracy_score(y[test_idx], y_pred)
                    scores.append(acc)

                    # Update progress bar
                    pbar.update(1)

                # Store results
                results.append({
                    'Dataset': ds_name,
                    'RF_Variation': var_name,
                    'Accuracy_Mean': np.mean(scores),
                    'Accuracy_Std': np.std(scores),
                    'CV_Splits': cv_splits
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV for later use
    results_df.to_csv(results_file, index=False)

    return results_df

# --------------------------------------------------
# 5. Main Execution
# --------------------------------------------------

if __name__ == "__main__":

    # Gather datasets
    synthetic_datasets = get_synthetic_datasets()
    real_datasets = get_real_datasets()
    all_datasets = synthetic_datasets + real_datasets

    # Define RandomForest variations
    rf_variations = get_custom_rf_variations()

    # Run experiments
    results_df = run_experiments(all_datasets, rf_variations, cv_splits=5)

    # Print or save results
    print("\nExperiment Results:")
    print(results_df.sort_values(by=['Dataset', 'Accuracy_Mean'], ascending=[True, False]))
