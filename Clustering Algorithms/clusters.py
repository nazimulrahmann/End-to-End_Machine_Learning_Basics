import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold

# ==============================================
# Import ALL Clustering Models
# ==============================================

# Partition-based Clustering
from sklearn.cluster import KMeans, MiniBatchKMeans

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering, Birch

# Density-based Clustering
from sklearn.cluster import DBSCAN, OPTICS, HDBSCAN

# Model-based Clustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Advanced Clustering
from sklearn.cluster import SpectralClustering, AffinityPropagation, MeanShift
from cluster import CLARA, PAM, FCM  # Requires pip install pyclustering, skfuzzy

# Neural Network-based
from sklearn.neural_network import SOM  # Requires pip install sklearn-som

# ==============================================
# Data Preparation
# ==============================================

# Load your data (replace with your actual data loading)
# X, y = load_data()  # y is optional (for evaluation metrics that need ground truth)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('variance_threshold', VarianceThreshold(threshold=0.0)),  # Remove zero-variance features
    ('scaler', StandardScaler()),  # Standard scaling works well for most clustering algorithms
    ('dim_reduction', PCA(n_components=0.95))  # Optional: reduce dimensionality for better performance
])

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# ==============================================
# Define ALL Clustering Models with Parameter Grids
# ==============================================

models = {
    # ========== Partition-based Clustering ==========
    'K-Means': {
        'model': KMeans(random_state=42),
        'params': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8],
            'init': ['k-means++', 'random'],
            'n_init': [10, 20, 30],
            'max_iter': [100, 200, 300],
            'algorithm': ['lloyd', 'elkan']
        }
    },

    'Mini-Batch K-Means': {
        'model': MiniBatchKMeans(random_state=42),
        'params': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8],
            'init': ['k-means++', 'random'],
            'n_init': [3, 5, 10],
            'max_iter': [100, 200],
            'batch_size': [100, 256, 512]
        }
    },

    # ========== Hierarchical Clustering ==========
    'Agglomerative': {
        'model': AgglomerativeClustering(),
        'params': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8],
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
            'linkage': ['ward', 'complete', 'average', 'single']
        }
    },

    'BIRCH': {
        'model': Birch(),
        'params': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, None],
            'threshold': [0.1, 0.5, 1.0],
            'branching_factor': [20, 30, 50]
        }
    },

    # ========== Density-based Clustering ==========
    'DBSCAN': {
        'model': DBSCAN(),
        'params': {
            'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
            'min_samples': [3, 5, 10, 15],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }
    },

    'OPTICS': {
        'model': OPTICS(),
        'params': {
            'min_samples': [3, 5, 10],
            'max_eps': [np.inf, 1.0, 2.0],
            'metric': ['euclidean', 'manhattan', 'cosine'],
            'cluster_method': ['xi', 'dbscan']
        }
    },

    'HDBSCAN': {
        'model': HDBSCAN(),
        'params': {
            'min_cluster_size': [5, 10, 15],
            'min_samples': [None, 3, 5, 10],
            'metric': ['euclidean', 'manhattan', 'cosine'],
            'cluster_selection_method': ['eom', 'leaf']
        }
    },

    # ========== Model-based Clustering ==========
    'Gaussian Mixture': {
        'model': GaussianMixture(random_state=42),
        'params': {
            'n_components': [2, 3, 4, 5, 6, 7, 8],
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'init_params': ['kmeans', 'random'],
            'max_iter': [100, 200, 300]
        }
    },

    'Bayesian Gaussian Mixture': {
        'model': BayesianGaussianMixture(random_state=42),
        'params': {
            'n_components': [2, 3, 4, 5, 6, 7, 8],
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'init_params': ['kmeans', 'random'],
            'max_iter': [100, 200, 300],
            'weight_concentration_prior_type': ['dirichlet_process', 'dirichlet_distribution']
        }
    },

    # ========== Advanced Clustering ==========
    'Spectral Clustering': {
        'model': SpectralClustering(random_state=42),
        'params': {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8],
            'affinity': ['nearest_neighbors', 'rbf', 'precomputed'],
            'n_neighbors': [5, 10, 15],
            'assign_labels': ['kmeans', 'discretize', 'cluster_qr']
        }
    },

    'Affinity Propagation': {
        'model': AffinityPropagation(),
        'params': {
            'damping': [0.5, 0.7, 0.9],
            'max_iter': [100, 200, 300],
            'convergence_iter': [10, 15, 20],
            'affinity': ['euclidean', 'precomputed']
        }
    },

    'Mean Shift': {
        'model': MeanShift(),
        'params': {
            'bandwidth': [None, 0.5, 1.0, 1.5],
            'bin_seeding': [True, False],
            'min_bin_freq': [1, 3, 5]
        }
    },

    # ========== Fuzzy Clustering ==========
    'Fuzzy C-Means': {
        'model': FCM(),
        'params': {
            'n_clusters': [2, 3, 4, 5, 6],
            'm': [1.1, 1.5, 2.0],
            'max_iter': [100, 200],
            'error': [0.001, 0.005, 0.01]
        }
    },

    # ========== Neural Network-based ==========
    'Self-Organizing Map': {
        'model': SOM(m=3, n=1, dim=len(X_preprocessed[0])),  # Adjust m and n based on expected clusters
        'params': {
            'lr': [0.1, 0.5, 1.0],
            'sigma': [0.5, 1.0, 2.0],
            'n_iterations': [100, 200, 500]
        }
    }
}


# ==============================================
# Evaluation Function
# ==============================================

def evaluate_clustering(model, X, y_true=None):
    # Some models like DBSCAN don't have predict method, only fit_predict
    if hasattr(model, 'predict'):
        labels = model.fit_predict(X)
    else:
        model.fit(X)
        if hasattr(model, 'labels_'):
            labels = model.labels_
        elif hasattr(model, 'predict'):
            labels = model.predict(X)
        else:
            raise ValueError("Model doesn't have predict or labels_ attribute")

    metrics = {}

    # Internal evaluation metrics (don't need ground truth)
    metrics['silhouette'] = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else -1
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else float('inf')

    # External evaluation metrics (need ground truth)
    if y_true is not None:
        metrics['adjusted_rand'] = adjusted_rand_score(y_true, labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(y_true, labels)

    print("\nClustering Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics, labels


# ==============================================
# Model Training and Evaluation
# ==============================================

results = {}
best_models = {}
all_labels = {}

for name, config in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training and evaluating {name}")
    print(f"{'=' * 50}")

    try:
        # Skip models that require additional packages if not installed
        if name in ['Fuzzy C-Means', 'Self-Organizing Map']:
            try:
                model = config['model']
            except:
                print(f"Skipping {name} - required package not installed")
                continue

        # Create model instance
        model = config['model']

        # For models that don't support parameter grids
        if 'params' not in config:
            metrics, labels = evaluate_clustering(model, X_preprocessed, y if 'y' in locals() else None)
            results[name] = metrics
            best_models[name] = model
            all_labels[name] = labels
            continue

        # Parameter grid search
        from sklearn.model_selection import ParameterGrid

        best_score = -1 if name != 'DBSCAN' else float('inf')  # For DBSCAN, lower is better for Davies-Bouldin
        best_params = None

        # Create parameter grid
        param_grid = ParameterGrid(config['params'])

        for params in param_grid:
            try:
                # Set parameters
                model.set_params(**params)

                # Evaluate
                metrics, labels = evaluate_clustering(model, X_preprocessed, y if 'y' in locals() else None)

                # Determine the best score (silhouette by default)
                current_score = metrics['silhouette']
                if name == 'DBSCAN':
                    current_score = -metrics['davies_bouldin']  # For DBSCAN, we prefer lower Davies-Bouldin

                # Update best model
                if (current_score > best_score and name != 'DBSCAN') or (
                        current_score > best_score and name == 'DBSCAN'):
                    best_score = current_score
                    best_params = params
                    best_metrics = metrics
                    best_labels = labels
            except Exception as e:
                print(f"Failed with params {params}: {str(e)}")
                continue

        if best_params is not None:
            # Train best model with best params
            model.set_params(**best_params)
            model.fit(X_preprocessed)

            results[name] = best_metrics
            best_models[name] = model
            all_labels[name] = best_labels

            print(f"\nBest parameters for {name}:")
            print(best_params)
        else:
            print(f"Could not find valid parameters for {name}")
            results[name] = {'error': 'No valid parameters found'}

    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        results[name] = {'error': str(e)}

# ==============================================
# Model Comparison
# ==============================================

print("\nModel Comparison:")
if 'y' in locals():
    # If we have ground truth, use external metrics
    comparison = pd.DataFrame.from_dict(results, orient='index')
    comparison = comparison[~comparison.index.isin([k for k, v in results.items() if 'error' in v])]
    print(comparison.sort_values(by='adjusted_rand', ascending=False))
else:
    # If no ground truth, use internal metrics
    comparison = pd.DataFrame.from_dict(results, orient='index')
    comparison = comparison[~comparison.index.isin([k for k, v in results.items() if 'error' in v])]
    print(comparison.sort_values(by='silhouette', ascending=False))

# ==============================================
# Visualization (Optional)
# ==============================================

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_clusters(X, labels, title):
    # Reduce to 2D for visualization
    if X.shape[1] > 2:
        X_vis = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        X_vis = X

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Visualize the best performing clusters
if 'y' in locals():
    best_model_name = comparison['adjusted_rand'].idxmax()
else:
    best_model_name = comparison['silhouette'].idxmax()

visualize_clusters(X_preprocessed, all_labels[best_model_name], f'Best Clustering: {best_model_name}')

# ==============================================
# Save Best Model
# ==============================================

print(f"\nBest model is: {best_model_name}")

# Save the best model
from joblib import dump

dump({
    'model': best_models[best_model_name],
    'preprocessor': preprocessor,
    'labels': all_labels[best_model_name]
}, 'best_clustering_model.joblib')

# Save all results to CSV
comparison.sort_values(by='silhouette' if 'y' not in locals() else 'adjusted_rand',
                       ascending=False).to_csv('clustering_model_comparison.csv')

print("\nClustering evaluation complete!")