# Install required libraries
!pip install numpy pandas scikit-learn matplotlib scipy pyod

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

from sklearn.datasets import make_blobs

def generate_synthetic_dataset(n_samples=1000, n_features=2, n_clusters=3, anomaly_ratio=0.05, density_factor=1.0, anomaly_type="global"):
    """
    Generates a synthetic dataset with configurable properties.

    Parameters:
    - n_samples: Total number of data points.
    - n_features: Number of features.
    - n_clusters: Number of normal data clusters.
    - anomaly_ratio: Proportion of anomalies in the dataset.
    - density_factor: Controls cluster density (higher = tighter clusters).
    - anomaly_type: Type of anomaly ("global", "local", "mixed").

    Returns:
    - Pandas DataFrame with labeled normal and anomaly data.
    """
    normal_samples = int(n_samples * (1 - anomaly_ratio))
    anomalies = n_samples - normal_samples

    # Generate normal data
    X_normal, _ = make_blobs(n_samples=normal_samples, centers=n_clusters, n_features=n_features, cluster_std=1.0/density_factor, random_state=42)

    # Generate anomalies
    mean = np.mean(X_normal, axis=0)
    cov = np.cov(X_normal, rowvar=False) * 5
    X_anomaly = np.random.multivariate_normal(mean, cov, size=anomalies)

    # Create dataset
    X = np.vstack((X_normal, X_anomaly))
    y = np.hstack((np.zeros(len(X_normal)), np.ones(len(X_anomaly))))  # 0: Normal, 1: Anomaly

    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
    df['Label'] = y
    return df

# Generate multiple synthetic datasets with different properties

# Define parameter ranges
n_samples_list = [500, 1000, 2000, 5000]  # Small to large datasets
anomaly_ratios = [0.01, 0.05, 0.10, 0.20,0.30]  # Different proportions of anomalies
density_factors = [0.25,0.5, 1.0, 1.5]  # Compact vs spread-out clusters
feature_dimensions = [2, 5, 10]  # Low-dimensional (for visualization) and high-dimensional cases
anomaly_types = ["global", "local", "mixed"]  # Different anomaly types

# Generate datasets dynamically
datasets = {}
dataset_id = 1
for (n_samples, anomaly_ratio, density_factor, n_features, anomaly_type) in itertools.product(
    n_samples_list, anomaly_ratios, density_factors, feature_dimensions, anomaly_types):

    dataset_name = f"Dataset_{dataset_id}"
    datasets[dataset_name] = generate_synthetic_dataset(
        n_samples=n_samples, n_features=n_features, anomaly_ratio=anomaly_ratio,
        density_factor=density_factor, anomaly_type=anomaly_type
    )
    dataset_id += 1

# Display total datasets generated
print(f"Total datasets generated: {len(datasets)}")


#########################################################

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cof import COF
from pyod.models.sod import SOD

def apply_local_anomaly_detection(df, method="LOF"):
    """
    Applies a local anomaly detection method to the dataset.

    Parameters:
    - df: DataFrame containing the dataset.
    - method: Local anomaly detection method.

    Returns:
    - DataFrame with an added column 'Local_Anomaly_Score'.
    """
    X = df.drop(columns=['Label']).values

    local_methods = {
        "LOF": LOF(),
        "KNN": KNN(),
        "COF": COF(),
        "SOD": SOD(),
        "LMDD": LMDD(),
    }

    if method not in local_methods:
        raise ValueError(f"Invalid method. Choose from {list(local_methods.keys())}")

    # Fit model and store anomaly scores
    detector = local_methods[method]
    detector.fit(X)
    df['Local_Anomaly_Score'] = detector.decision_scores_

    return df


#############################################################

import numpy as np
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler

def apply_global_anomaly_detection(df, method="PCA"):
    """
    Applies a global anomaly detection method.

    Parameters:
    - df: DataFrame containing the dataset.
    - method: Global anomaly detection method.

    Returns:
    - DataFrame with an added column 'Global_Anomaly_Score'.
    """
    X = df.drop(columns=['Label']).values

    global_methods = {
        "PCA": PCA(),
        "HBOS": HBOS(),
        "OCSVM": OCSVM(),
        "IForest": IForest(),
        "MCD": MCD(),
        "GMM": GMM()
    }

    if method not in global_methods:
        raise ValueError(f"Invalid method. Choose from {list(global_methods.keys())}")

    # Fit model and store anomaly scores
    detector = global_methods[method]
    detector.fit(X)
    df['Global_Anomaly_Score'] = detector.decision_scores_

    return df


#######################################################################

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


##################################################

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

#######################################

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



################################################################

local_methods = {
    'LOF': LOF(),
    'KNN': KNN(),
    'COF': COF(),
    'SOD': SOD(),
    'LMDD': LMDD()
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


local_methods = {
    'LOF': LOF(),
    'KNN': KNN(),
    'COF': COF(),
    'SOD': SOD(),
    'LMDD': LMDD()
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

# Determine the split point
dataset_names = list(datasets.keys())
half_split = ceil(len(dataset_names) / 2)

# Select first half of datasets
datasets_first_half = {k: datasets[k] for k in dataset_names[:half_split]}

print(f"Processing first half: {len(datasets_first_half)} datasets...")

# Store results
results = []

# Initialize dataset counter
dataset_counter = 0

# Iterate through the first half of synthetic datasets
for dataset_name, df in datasets_first_half.items():
    dataset_counter += 1  # Increment counter

    for local_name, local_model in local_methods.items():
        for global_name, global_model in global_methods.items():
            for w_local, w_global in weight_pairs:
                # Apply local & global anomaly detection
                df_local = apply_local_anomaly_detection(df.copy(), method=local_name)
                df_global = apply_global_anomaly_detection(df_local.copy(), method=global_name)

                # Normalize Scores
                df_combined = normalize_scores(df_global)

                # Combine Scores
                df_combined = combine_scores(df_combined, weight_local=w_local, weight_global=w_global)

                # Evaluate
                scores = evaluate_anomaly_detection(df_combined)

                # Store results
                results.append({
                    "Dataset": dataset_name,
                    "Local Method": local_name,
                    "Global Method": global_name,
                    "Weight Local": w_local,
                    "Weight Global": w_global,
                    **scores  # Add all evaluation metrics dynamically
                })

    # Print progress every 20 datasets processed
    if dataset_counter % 2 == 0:
        print(f"Processed {dataset_counter}/{len(datasets_first_half)} datasets...")

# Convert to DataFrame
df_synthetic_first_half = pd.DataFrame(results)

# Save first half to CSV
df_synthetic_first_half.to_csv("synthetic_ad_results_part1.csv", index=False)

print("First half of synthetic datasets processed and saved.")

from google.colab import files

# Download the CSV file
files.download("synthetic_ad_results_part1.csv")

