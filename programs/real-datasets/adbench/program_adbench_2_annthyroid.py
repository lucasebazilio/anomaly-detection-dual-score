# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import itertools
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, matthews_corrcoef
from scipy.spatial.distance import mahalanobis
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.lmdd import LMDD
from math import ceil
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.mcd import MCD
from pyod.models.gmm import GMM
from sklearn.preprocessing import StandardScaler



# Ignore all warnings
warnings.filterwarnings('ignore')


########################################

# List of ADBench datasets
adbench_datasets = [
    "2_annthyroid.npz"
]

# Dictionary to store all loaded datasets
datasets = {}

# Load each dataset
for dataset in adbench_datasets:
    data = np.load(dataset)
    X = data["X"]  # Features
    y = data["y"]  # Labels (0 = normal, 1 = anomaly)

    datasets[dataset] = {"X": X, "y": y}

print(f" Loaded {len(datasets)} datasets from ADBench.")


############################################

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cof import COF
from pyod.models.sod import SOD
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM



# Define local & global techniques (matching real-world dataset part)
local_methods = {
    "LOF": LOF(),
    "KNN": KNN(),
    "COF": COF(),
    "SOD": SOD(),
    "LMDD": LMDD(),
}

global_methods = {
    "PCA": PCA(),
    "HBOS": HBOS(),
    "OCSVM": OCSVM(),
    "IForest": IForest(),
    "MCD": MCD(),
    "GMM": GMM()
}

# Weight combinations to experiment
weight_pairs = [
    (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4),
    (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)
]

##################################################################
from sklearn.preprocessing import MinMaxScaler

def normalize_scores(df, local_col="Local_Anomaly_Score", global_col="Global_Anomaly_Score"):
    """
    Normalizes local and global anomaly scores to a [0,1] range.

    Parameters:
    - df: Pandas DataFrame containing local and global anomaly scores.
    - local_col: Name of the column with local anomaly scores.
    - global_col: Name of the column with global anomaly scores.

    Returns:
    - DataFrame with normalized local and global scores.
    """
    scaler = MinMaxScaler()
    df[[local_col, global_col]] = scaler.fit_transform(df[[local_col, global_col]])
    return df


def combine_scores(df, weight_local=0.5, weight_global=0.5):
    """
    Computes a combined anomaly score using a weighted sum of local and global scores.

    Parameters:
    - df: Pandas DataFrame with normalized local and global scores.
    - weight_local: Weight assigned to the local anomaly score.
    - weight_global: Weight assigned to the global anomaly score.

    Returns:
    - DataFrame with a new column 'Dual_Anomaly_Score'.
    """
    total_weight = weight_local + weight_global
    df['Dual_Anomaly_Score'] = ((weight_local / total_weight) * df['Local_Anomaly_Score']) + \
                               ((weight_global / total_weight) * df['Global_Anomaly_Score'])
    return df


from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_anomaly_detection(df, score_col="Dual_Anomaly_Score"):
    """
    Evaluates anomaly detection performance using AUC-ROC and Precision-Recall AUC.

    Parameters:
    - df: Pandas DataFrame containing the anomaly scores and true labels.
    - score_col: Column name of the anomaly score to evaluate.

    Returns:
    - A dictionary with AUC-ROC and AUC-PR scores.
    """
    y_true = df['Label']
    y_scores = df[score_col]

    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)

    return {"AUC-ROC": auc_roc, "AUC-PR": auc_pr}



from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

def evaluate_anomaly_detection(df, score_col="Dual_Anomaly_Score"):
    """
    Evaluates anomaly detection performance using multiple metrics.

    Parameters:
    - df: Pandas DataFrame containing the anomaly scores and true labels.
    - score_col: Column name of the anomaly score to evaluate.

    Returns:
    - A dictionary with multiple evaluation metrics.
    """
    y_true = df['Label']
    y_scores = df[score_col]

    # Convert scores into binary labels based on a threshold (top N% as anomalies)
    threshold = np.percentile(y_scores, 100 * (sum(y_true) / len(y_true)))  # Cutoff based on true anomaly proportion
    y_pred = (y_scores >= threshold).astype(int)

    results = {
        "AUC-ROC": roc_auc_score(y_true, y_scores),
        "AUC-PR": average_precision_score(y_true, y_scores),
        "F1-score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred)  # Balanced metric for imbalanced datasets
    }
    return results



###################################################################################################



# Import evaluation metrics
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler

def normalize_scores(scores):
    """Normalizes anomaly scores between 0 and 1."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()

def apply_anomaly_detection(X, method):
    """Fits an anomaly detection model and returns the scores."""
    model = method.fit(X)
    return model.decision_scores_  # Extract anomaly scores



# Store results
results = []

# Iterate through all datasets
for dataset_name, data in datasets.items():
    X, y = data["X"], data["y"]

    # Print debugging information
    total_instances = X.shape[0]
    print(f"🔹 Processing Dataset: {dataset_name} (Total Instances: {total_instances}, Features: {X.shape[1]})")

    processed_instances = 0  # Track how many instances have been processed

    for local_name, local_model in local_methods.items():
        for global_name, global_model in global_methods.items():
            for w_local, w_global in weight_pairs:
                # Apply local & global methods
                local_scores = apply_anomaly_detection(X, local_model)
                global_scores = apply_anomaly_detection(X, global_model)

                # Normalize scores
                local_scores = normalize_scores(local_scores)
                global_scores = normalize_scores(global_scores)

                # Combine scores with weighted sum
                combined_scores = (w_local * local_scores) + (w_global * global_scores)

                # Store results
                results.append({
                    "Dataset": dataset_name,
                    "Local Method": local_name,
                    "Global Method": global_name,
                    "Weight Local": w_local,
                    "Weight Global": w_global,
                    "AUC-ROC": roc_auc_score(y, combined_scores),
                    "AUC-PR": average_precision_score(y, combined_scores),
                    "F1-score": f1_score(y, (combined_scores > np.percentile(combined_scores, 95)).astype(int)),
                    "Precision": precision_score(y, (combined_scores > np.percentile(combined_scores, 95)).astype(int)),
                    "Recall": recall_score(y, (combined_scores > np.percentile(combined_scores, 95)).astype(int)),
                    "MCC": matthews_corrcoef(y, (combined_scores > np.percentile(combined_scores, 95)).astype(int))
                })

                # Update processed instances
                processed_instances += 2  # Increment by 2

                # Print frequently for first 50 instances, then every 100
                if processed_instances <= 50 or processed_instances % 100 == 0:
                    print(f"    Processed {processed_instances} instances so far in {dataset_name}")

    print(f" Finished processing dataset: {dataset_name} (Total Processed Instances: {processed_instances})\n")


# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv("adbench_results_2_annthyroid.csv", index=False)

print("Results saved to 'adbench_results_2_annthyroid.csv'.")

# The code below only works when we are in a notebook environment
# Download results
#from google.colab import files

# Download the CSV file
#files.download("adbench_results.csv")

