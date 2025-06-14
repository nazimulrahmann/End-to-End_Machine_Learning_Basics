import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score,
                             max_error, mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# ==============================================
# Import ALL Regression Models
# ==============================================

# Linear Models
from sklearn.linear_model import (LinearRegression,
                                  Ridge,
                                  Lasso,
                                  ElasticNet,
                                  BayesianRidge,
                                  ARDRegression,
                                  SGDRegressor,
                                  HuberRegressor,
                                  TheilSenRegressor,
                                  RANSACRegressor,
                                  PassiveAggressiveRegressor,
                                  OrthogonalMatchingPursuit)

# Tree-based Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              AdaBoostRegressor,
                              ExtraTreesRegressor,
                              BaggingRegressor,
                              StackingRegressor,
                              VotingRegressor,
                              HistGradientBoostingRegressor)

# SVM Models
from sklearn.svm import SVR, LinearSVR, NuSVR

# Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

# Neural Networks
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Gaussian Processes
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic

# Ensemble Methods
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Other regressors
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor

# ==============================================
# Data Preparation
# ==============================================

# Load your data (replace with your actual data loading)
# X, y = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler()),  # Standard scaling works well for most models
    ('feature_selector', SelectKBest(f_regression, k='all'))  # Can adjust 'k' as needed
])

# ==============================================
# Define ALL Models with Parameter Grids
# ==============================================

models = {
    # ========== Linear Models ==========
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {
            'model__fit_intercept': [True, False],
            'model__positive': [True, False]
        }
    },

    'Ridge Regression': {
        'model': Ridge(random_state=42),
        'params': {
            'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },

    'Lasso Regression': {
        'model': Lasso(random_state=42),
        'params': {
            'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'model__selection': ['cyclic', 'random']
        }
    },

    'ElasticNet': {
        'model': ElasticNet(random_state=42),
        'params': {
            'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'model__selection': ['cyclic', 'random']
        }
    },

    'Bayesian Ridge': {
        'model': BayesianRidge(),
        'params': {
            'model__alpha_1': [1e-6, 1e-5, 1e-4],
            'model__alpha_2': [1e-6, 1e-5, 1e-4],
            'model__lambda_1': [1e-6, 1e-5, 1e-4],
            'model__lambda_2': [1e-6, 1e-5, 1e-4]
        }
    },

    'ARD Regression': {
        'model': ARDRegression(),
        'params': {
            'model__alpha_1': [1e-6, 1e-5, 1e-4],
            'model__alpha_2': [1e-6, 1e-5, 1e-4],
            'model__lambda_1': [1e-6, 1e-5, 1e-4],
            'model__lambda_2': [1e-6, 1e-5, 1e-4],
            'model__threshold_lambda': [100, 1000, 10000]
        }
    },

    'SGD Regressor': {
        'model': SGDRegressor(random_state=42),
        'params': {
            'model__loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
        }
    },

    'Huber Regressor': {
        'model': HuberRegressor(),
        'params': {
            'model__epsilon': [1.1, 1.35, 1.5],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__max_iter': [100, 200, 300]
        }
    },

    'Theil-Sen Regressor': {
        'model': TheilSenRegressor(random_state=42),
        'params': {
            'model__max_subpopulation': [1000, 5000, 10000],
            'model__n_subsamples': [None, 100, 200],
            'model__max_iter': [100, 300, 500]
        }
    },

    'RANSAC Regressor': {
        'model': RANSACRegressor(random_state=42),
        'params': {
            'model__min_samples': [None, 0.1, 0.5, 0.9],
            'model__residual_threshold': [None, 1.0, 2.0],
            'model__max_trials': [50, 100, 200]
        }
    },

    # ========== Tree-based Models ==========
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None]
        }
    },

    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None],
            'model__bootstrap': [True, False]
        }
    },

    'Extra Trees': {
        'model': ExtraTreesRegressor(random_state=42),
        'params': {
            'model__n_estimators': [100, 200],
            'model__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'model__max_features': ['sqrt', 'log2', None]
        }
    },

    # ========== Boosting Models ==========
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0],
            'model__min_samples_split': [2, 5]
        }
    },

    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 6, 9],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0],
            'model__gamma': [0, 0.1, 0.2],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [0, 0.1, 1]
        }
    },

    'LightGBM': {
        'model': LGBMRegressor(random_state=42),
        'params': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__num_leaves': [31, 63, 127],
            'model__max_depth': [-1, 10, 20],
            'model__min_child_samples': [20, 50],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [0, 0.1, 1]
        }
    },

    'CatBoost': {
        'model': CatBoostRegressor(random_state=42, verbose=0),
        'params': {
            'model__iterations': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__depth': [4, 6, 8],
            'model__l2_leaf_reg': [1, 3, 5]
        }
    },

    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1.0],
            'model__loss': ['linear', 'square', 'exponential']
        }
    },

    'Hist Gradient Boosting': {
        'model': HistGradientBoostingRegressor(random_state=42),
        'params': {
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_iter': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [20, 50, 100]
        }
    },

    # ========== SVM Models ==========
    'SVR': {
        'model': SVR(),
        'params': {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto', 0.1, 1],
            'model__degree': [2, 3, 4],  # For poly kernel
            'model__epsilon': [0.01, 0.1, 0.5]
        }
    },

    'Linear SVR': {
        'model': LinearSVR(random_state=42),
        'params': {
            'model__C': [0.1, 1, 10],
            'model__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'model__dual': [True, False],
            'model__epsilon': [0.01, 0.1, 0.5]
        }
    },

    'NuSVR': {
        'model': NuSVR(),
        'params': {
            'model__nu': [0.1, 0.5, 0.8],
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto', 0.1, 1]
        }
    },

    # ========== Nearest Neighbors ==========
    'KNN Regressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9, 11],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'model__p': [1, 2]  # 1: Manhattan, 2: Euclidean
        }
    },

    'Radius Neighbors Regressor': {
        'model': RadiusNeighborsRegressor(),
        'params': {
            'model__radius': [1.0, 2.0, 5.0],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    },

    # ========== Neural Networks ==========
    'MLP Regressor': {
        'model': MLPRegressor(random_state=42, early_stopping=True),
        'params': {
            'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'model__activation': ['relu', 'tanh', 'logistic'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'model__learning_rate_init': [0.001, 0.01]
        }
    },

    # ========== Gaussian Processes ==========
    'Gaussian Process': {
        'model': GaussianProcessRegressor(random_state=42),
        'params': {
            'model__kernel': [None,
                              RBF(),
                              ConstantKernel() * RBF(),
                              RationalQuadratic()],
            'model__alpha': [1e-10, 1e-5, 1e-2],
            'model__normalize_y': [True, False]
        }
    },

    # ========== Other Regressors ==========
    'Kernel Ridge': {
        'model': KernelRidge(),
        'params': {
            'model__alpha': [0.1, 1.0, 10.0],
            'model__kernel': ['linear', 'rbf', 'polynomial'],
            'model__gamma': [None, 0.1, 1.0],
            'model__degree': [2, 3, 4]
        }
    },

    'PLS Regression': {
        'model': PLSRegression(),
        'params': {
            'model__n_components': [1, 2, 3, 5],
            'model__scale': [True, False],
            'model__max_iter': [500, 1000]
        }
    },

    'Dummy Regressor': {
        'model': DummyRegressor(),
        'params': {
            'model__strategy': ['mean', 'median', 'quantile', 'constant']
        }
    }
}


# ==============================================
# Evaluation Function
# ==============================================

def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'explained_variance': explained_variance_score(y_test, y_pred),
        'max_error': max_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred)
    }

    print("\nRegression Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


# ==============================================
# Model Training and Evaluation
# ==============================================

results = {}
best_models = {}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, config in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training and tuning {name}")
    print(f"{'=' * 50}")

    try:
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])

        # Update params to include pipeline prefix
        params = {f'model__{key}': value for key, value in config['params'].items()}

        grid_search = GridSearchCV(
            pipeline,
            param_grid=params,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        results[name] = evaluate_regression_model(best_models[name], X_test, y_test)

        print(f"\nBest parameters for {name}:")
        print(grid_search.best_params_)

    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        results[name] = {'error': str(e)}

# ==============================================
# Model Comparison
# ==============================================

print("\nModel Comparison:")
comparison = pd.DataFrame.from_dict(results, orient='index')
# Remove models that failed
comparison = comparison[~comparison.index.isin([k for k, v in results.items() if 'error' in v])]
print(comparison.sort_values(by='r2', ascending=False))

# ==============================================
# Ensemble Methods
# ==============================================

print("\nBuilding Ensemble Models...")

# Select top models for ensemble
top_models = comparison.nlargest(5, 'r2').index.tolist()

# Create a list of base estimators for stacking
estimators = [(name, best_models[name].named_steps['model']) for name in top_models]

# Stacking Regressor
stacking_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1
    ))
])

stacking_reg.fit(X_train, y_train)
print("\nStacking Regressor Performance:")
stacking_metrics = evaluate_regression_model(stacking_reg, X_test, y_test)
results['Stacking'] = stacking_metrics

# Voting Regressor
voting_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('model', VotingRegressor(
        estimators=estimators,
        n_jobs=-1
    ))
])

voting_reg.fit(X_train, y_train)
print("\nVoting Regressor Performance:")
voting_metrics = evaluate_regression_model(voting_reg, X_test, y_test)
results['Voting'] = voting_metrics

# ==============================================
# Final Comparison
# ==============================================

print("\nFinal Model Comparison:")
final_comparison = pd.DataFrame.from_dict(results, orient='index')
final_comparison = final_comparison[~final_comparison.index.isin([k for k, v in results.items() if 'error' in v])]
print(final_comparison.sort_values(by='r2', ascending=False))

# ==============================================
# Save Best Model
# ==============================================

best_model_name = final_comparison['r2'].idxmax()
best_model = best_models.get(best_model_name,
                             stacking_reg if best_model_name == 'Stacking' else voting_reg)

print(f"\nBest model is: {best_model_name}")

# Save the best model
from joblib import dump

dump(best_model, 'best_regressor_model.joblib')

# Save all results to CSV
final_comparison.sort_values(by='r2', ascending=False).to_csv('regression_model_comparison.csv')

print("\nModel training and evaluation complete!")