import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, log_loss)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# ==============================================
# Import ALL Classification Models
# ==============================================

# Linear Models
from sklearn.linear_model import (LogisticRegression,
                                  RidgeClassifier,
                                  SGDClassifier,
                                  PassiveAggressiveClassifier)

# Tree-based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier,
                              BaggingClassifier,
                              StackingClassifier,
                              VotingClassifier,
                              HistGradientBoostingClassifier)

# SVM Models
from sklearn.svm import SVC, LinearSVC, NuSVC

# Naive Bayes Models
from sklearn.naive_bayes import (GaussianNB,
                                 BernoulliNB,
                                 MultinomialNB,
                                 ComplementNB,
                                 CategoricalNB)

# Discriminant Analysis
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

# Nearest Neighbors
from sklearn.neighbors import (KNeighborsClassifier,
                               RadiusNeighborsClassifier,
                               NearestCentroid)

# Neural Networks
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Ensemble Methods
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Semi-supervised
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# Other classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest  # For anomaly detection

# ==============================================
# Data Preparation
# ==============================================

# Load your data (replace with your actual data loading)
# X, y = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler()),  # Standard scaling works well for most models
    ('feature_selector', SelectKBest(f_classif, k='all'))  # Can adjust 'k' as needed
])

# Handle target encoding if needed
if y.dtype == 'object' or isinstance(y[0], str):
    le = LabelEncoder()
    y = le.fit_transform(y)

# ==============================================
# Define ALL Models with Parameter Grids
# ==============================================

models = {
    # ========== Linear Models ==========
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l1', 'l2', 'elasticnet', None],
            'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'model__class_weight': [None, 'balanced']
        }
    },

    'Ridge Classifier': {
        'model': RidgeClassifier(random_state=42),
        'params': {
            'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'model__class_weight': [None, 'balanced']
        }
    },

    'SGD Classifier': {
        'model': SGDClassifier(random_state=42),
        'params': {
            'model__loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
            'model__penalty': ['l1', 'l2', 'elasticnet'],
            'model__alpha': [0.0001, 0.001, 0.01, 0.1],
            'model__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
        }
    },

    # ========== Tree-based Models ==========
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'model__criterion': ['gini', 'entropy', 'log_loss'],
            'model__max_depth': [None, 5, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', None]
        }
    },

    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
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
        'model': ExtraTreesClassifier(random_state=42),
        'params': {
            'model__n_estimators': [100, 200],
            'model__criterion': ['gini', 'entropy'],
            'model__max_features': ['sqrt', 'log2', None],
            'model__class_weight': [None, 'balanced', 'balanced_subsample']
        }
    },

    # ========== Boosting Models ==========
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0],
            'model__min_samples_split': [2, 5]
        }
    },

    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
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
        'model': LGBMClassifier(random_state=42),
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
        'model': CatBoostClassifier(random_state=42, verbose=0),
        'params': {
            'model__iterations': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__depth': [4, 6, 8],
            'model__l2_leaf_reg': [1, 3, 5]
        }
    },

    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 1.0],
            'model__algorithm': ['SAMME', 'SAMME.R']
        }
    },

    'Hist Gradient Boosting': {
        'model': HistGradientBoostingClassifier(random_state=42),
        'params': {
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_iter': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [20, 50, 100]
        }
    },

    # ========== SVM Models ==========
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto', 0.1, 1],
            'model__degree': [2, 3, 4],  # For poly kernel
            'model__class_weight': [None, 'balanced']
        }
    },

    'Linear SVM': {
        'model': LinearSVC(random_state=42),
        'params': {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l1', 'l2'],
            'model__loss': ['hinge', 'squared_hinge'],
            'model__dual': [True, False]
        }
    },

    'NuSVC': {
        'model': NuSVC(random_state=42, probability=True),
        'params': {
            'model__nu': [0.1, 0.5, 0.8],
            'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto', 0.1, 1]
        }
    },

    # ========== Naive Bayes Models ==========
    'Gaussian Naive Bayes': {
        'model': GaussianNB(),
        'params': {
            'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },

    'Bernoulli Naive Bayes': {
        'model': BernoulliNB(),
        'params': {
            'model__alpha': [0.1, 0.5, 1.0, 2.0],
            'model__binarize': [None, 0.0, 0.5]
        }
    },

    'Multinomial Naive Bayes': {
        'model': MultinomialNB(),
        'params': {
            'model__alpha': [0.1, 0.5, 1.0, 2.0],
            'model__fit_prior': [True, False]
        }
    },

    'Complement Naive Bayes': {
        'model': ComplementNB(),
        'params': {
            'model__alpha': [0.1, 0.5, 1.0, 2.0],
            'model__fit_prior': [True, False],
            'model__norm': [True, False]
        }
    },

    # ========== Discriminant Analysis ==========
    'LDA': {
        'model': LinearDiscriminantAnalysis(),
        'params': {
            'model__solver': ['svd', 'lsqr', 'eigen'],
            'model__shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
        }
    },

    'QDA': {
        'model': QuadraticDiscriminantAnalysis(),
        'params': {
            'model__reg_param': [0.0, 0.1, 0.5, 1.0]
        }
    },

    # ========== Nearest Neighbors ==========
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'model__n_neighbors': [3, 5, 7, 9, 11],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'model__p': [1, 2]  # 1: Manhattan, 2: Euclidean
        }
    },

    'Radius Neighbors': {
        'model': RadiusNeighborsClassifier(),
        'params': {
            'model__radius': [1.0, 2.0, 5.0],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    },

    'Nearest Centroid': {
        'model': NearestCentroid(),
        'params': {
            'model__metric': ['euclidean', 'manhattan', 'cosine'],
            'model__shrink_threshold': [None, 0.1, 0.5, 1.0]
        }
    },

    # ========== Neural Networks ==========
    'MLP': {
        'model': MLPClassifier(random_state=42, early_stopping=True),
        'params': {
            'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'model__activation': ['relu', 'tanh', 'logistic'],
            'model__alpha': [0.0001, 0.001, 0.01],
            'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'model__learning_rate_init': [0.001, 0.01]
        }
    },

    # ========== Other Classifiers ==========
    'Gaussian Process': {
        'model': GaussianProcessClassifier(random_state=42),
        'params': {
            'model__kernel': [None, 'RBF', 'DotProduct'],
            'model__max_iter_predict': [100, 200]
        }
    },

    'Dummy Classifier': {
        'model': DummyClassifier(random_state=42),
        'params': {
            'model__strategy': ['most_frequent', 'prior', 'stratified', 'uniform']
        }
    },

    # ========== Semi-supervised ==========
    'Label Propagation': {
        'model': LabelPropagation(),
        'params': {
            'model__kernel': ['rbf', 'knn'],
            'model__gamma': [0.1, 0.5, 1.0],
            'model__n_neighbors': [3, 5, 7]
        }
    },

    'Label Spreading': {
        'model': LabelSpreading(),
        'params': {
            'model__kernel': ['rbf', 'knn'],
            'model__alpha': [0.1, 0.5, 0.9],
            'model__n_neighbors': [3, 5, 7]
        }
    }
}


# ==============================================
# Evaluation Function
# ==============================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    if y_proba is not None and len(np.unique(y_test)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['log_loss'] = log_loss(y_test, y_proba)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


# ==============================================
# Model Training and Evaluation
# ==============================================

results = {}
best_models = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        results[name] = evaluate_model(best_models[name], X_test, y_test)

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
print(comparison.sort_values(by='accuracy', ascending=False))

# ==============================================
# Ensemble Methods
# ==============================================

print("\nBuilding Ensemble Models...")

# Select top models for ensemble
top_models = comparison.nlargest(5, 'accuracy').index.tolist()

# Create a list of base estimators for stacking
estimators = [(name, best_models[name].named_steps['model']) for name in top_models]

# Stacking Classifier
stacking_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        stack_method='auto',
        n_jobs=-1
    ))
])

stacking_clf.fit(X_train, y_train)
print("\nStacking Classifier Performance:")
stacking_metrics = evaluate_model(stacking_clf, X_test, y_test)
results['Stacking'] = stacking_metrics

# Voting Classifier (Soft Voting)
voting_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    ))
])

voting_clf.fit(X_train, y_train)
print("\nVoting Classifier Performance:")
voting_metrics = evaluate_model(voting_clf, X_test, y_test)
results['Voting'] = voting_metrics

# ==============================================
# Final Comparison
# ==============================================

print("\nFinal Model Comparison:")
final_comparison = pd.DataFrame.from_dict(results, orient='index')
final_comparison = final_comparison[~final_comparison.index.isin([k for k, v in results.items() if 'error' in v])]
print(final_comparison.sort_values(by='accuracy', ascending=False))

# ==============================================
# Save Best Model
# ==============================================

best_model_name = final_comparison['accuracy'].idxmax()
best_model = best_models.get(best_model_name,
                             stacking_clf if best_model_name == 'Stacking' else voting_clf)

print(f"\nBest model is: {best_model_name}")

# Save the best model
from joblib import dump

dump(best_model, 'best_classifier_model.joblib')

# Save all results to CSV
final_comparison.sort_values(by='accuracy', ascending=False).to_csv('model_comparison_results.csv')

print("\nModel training and evaluation complete!")