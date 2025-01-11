# regression_test.py

from random_forest import MyRandomForestRegressor

import numpy as np
import pandas as pd

from sklearn.datasets import (
    make_regression,
    make_friedman1,
    load_diabetes,
    fetch_california_housing
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# --------------------------------------------------
# 1. Prepare Datasets
# --------------------------------------------------

def get_synthetic_datasets_reg():
    datasets = []
    # A basic regression dataset
    X1, y1 = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=5,
        noise=10.0,
        random_state=42
    )
    datasets.append(('Synthetic_Regression', X1, y1))

    X2, y2 = make_friedman1(
        n_samples=1000,
        n_features=10,
        noise=1.0,
        random_state=42
    )
    datasets.append(('Friedman1', X2, y2))
    return datasets

def get_real_datasets_reg():
    datasets = []

    # 1) California Housing
    cali = fetch_california_housing(as_frame=False)
    X_cali, y_cali = cali.data, cali.target
    #datasets.append(('CaliforniaHousing', X_cali, y_cali))

    # 2) Diabetes dataset
    diab = load_diabetes()
    X_diab, y_diab = diab.data, diab.target
    datasets.append(('Diabetes', X_diab, y_diab))

    # You can add more regression datasets from OpenML or Kaggle
    return datasets

# --------------------------------------------------
# 2. Define Different Variations of Custom RandomForestRegressor
# --------------------------------------------------

def get_custom_rf_variations_reg():
    """
    Return a dictionary of different RandomForestRegressor configurations.
    """
    variations = {
        'RF_default': MyRandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, n_features=10, bagging=True),
        'RF_sklearn': RandomForestRegressor(n_estimators=100, max_depth=10, max_features=10),
        'RF_simple': MyRandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2, n_features=10, bagging=True, split_criterion="middle"),
    }
    return variations

# --------------------------------------------------
# 3. Experiment Function (CV for Regression)
# --------------------------------------------------

def run_experiments_reg(
    datasets, 
    rf_variations, 
    cv_splits=1, 
    results_file="regression_results.csv"
):
    """
    Runs K-Fold cross-validation for each dataset
    and each RF variation, and saves the results.

    We'll compute both MSE and R^2 here, just as examples.
    """
    results = []

    total_tasks = len(datasets) * len(rf_variations) * cv_splits

    with tqdm(total=total_tasks, desc="Running Regression Experiments") as pbar:
        for ds_name, X, y in datasets:
            print("Dataset:", ds_name)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            for var_name, rf_model in rf_variations.items():
                mse_scores = []
                r2_scores = []

                # Fit model
                rf_model.fit(X_train, y_train)

                # Predict
                y_pred = rf_model.predict(X_test)

                # Evaluate
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mse_scores.append(mse)
                r2_scores.append(r2)

                pbar.update(1)

                # Store results
                results.append({
                    'Dataset': ds_name,
                    'RF_Variation': var_name,
                    'MSE': np.mean(mse_scores),
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV for later use
    results_df.to_csv(results_file, index=False)

    return results_df

# --------------------------------------------------
# 4. Main Execution
# --------------------------------------------------

if __name__ == "__main__":

    # 1) Gather datasets
    synthetic_datasets = get_synthetic_datasets_reg()
    real_datasets = get_real_datasets_reg()
    all_datasets = synthetic_datasets + real_datasets

    # 2) Define RandomForestRegressor variations
    rf_variations = get_custom_rf_variations_reg()

    # 3) Run experiments
    results_df = run_experiments_reg(
        all_datasets, 
        rf_variations, 
        cv_splits=1, 
        results_file="regression_results.csv"
    )

    # 4) Print or save results
    print("\nExperiment Results (Regression):")
    print(results_df.sort_values(by=['Dataset', 'MSE'], ascending=[True, False]))
    print(f"\nResults have been saved to 'regression_results.csv'.")
