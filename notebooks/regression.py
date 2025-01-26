import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer
)

from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.ensemble import (
    RandomForestRegressor,
)

from xgboost import XGBRegressor

""""'Ridge Regression': {
    'model': Ridge(),
    'params': {
        'alpha': [0.5, 1, 2, 5, 10],  # Around the best value of 1
    }
},
'Lasso Regression': {
    'model': Lasso(),
    'params': {
        'alpha': [0.0005, 0.001, 0.005, 0.01],  # Around the best value of 0.001
        'max_iter': [5000]
    }
},
'Elastic Net': {
    'model': ElasticNet(),
    'params': {
        'alpha': [0.005, 0.01, 0.05],  # Around the best value of 0.01
        'l1_ratio': [0.05, 0.1, 0.15],  # Around the best value of 0.1
        'max_iter': [5000]
    }
},"""


def preprocess_data(df, selected_features, log_transform_features=None):
    X = df[selected_features].copy()

    # One-hot encode MgFe
    mgfe_encoded = pd.get_dummies(X['MgFe'], prefix='MgFe')
    X = X.drop('MgFe', axis=1)
    X = pd.concat([X, mgfe_encoded], axis=1)

    # Log transform other features
    if log_transform_features is None:
        log_transform_features = {
            'age_mean_mass': True,
            'velDisp_ppxf_res': True,
            '[M/H]_mean_mass': True  # Added based on histogram
        }

    for feature, do_log in log_transform_features.items():
        if do_log and feature in X.columns:
            X[feature] = np.log10(X[feature] + 1e-10)

    return X

def plot_feature_importance(model, feature_names):
    # Get feature importances (works for RandomForest, XGBoost, and GradientBoosting)
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importance = model.named_steps['regressor'].feature_importances_

        # Sort features by importance
        indices = np.argsort(importance)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        print("AAAAA")
        plt.show()

        # Print numerical values
        for i in indices:
            print(f"{feature_names[i]}: {importance[i]:.4f}")
    else:
        print("This model doesn't support feature importances")

def plot_regression_results(y_true, y_pred, model_name):

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)

    # Diagonal line representing perfect predictions
    min_val, max_val = 0,1 #min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    ax.set_title(f'{model_name} Regression Results')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')

    text = (f'R² = {r2:.3f}\n'
            f'RMSE = {rmse:.3f}\n'
            f'MAE = {mae:.3f}')
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    return fig


def comprehensive_regression_analysis(
        X,
        y,
        test_size=0.2,
        random_state=42,
        log_transform_features=None
):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define models with their parameter grids
    models = {
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        },
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}  # No hyperparameters to tune
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=random_state),
            'params': {
                'n_estimators': [75, 150],  # Around the best value of 100
                'max_depth': [10, 25, None],  # Add some specific depths, keep None as an option
                'min_samples_split': [2, 5],  # Around the best value of 2
                'max_features': ['sqrt', 'log2'],
                'max_samples': [0.6, 0.8, 1.0]  # Add bootstrap sample size
            }
        },
    }

    # Results storage
    results = {}

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=1,
        figsize=(10, 4 * len(models))
    )

    for i, (name, model_info) in enumerate(models.items()):
        print(f"\nEvaluating {name}...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model_info['model'])
        ])

        if model_info['params']:
            grid_search = GridSearchCV(
                pipeline,
                param_grid={f'regressor__{k}': v for k, v in model_info['params'].items()},
                cv=5,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)

            # Best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            plot_feature_importance(best_model, X.columns.tolist())
            # Predictions
            y_pred = best_model.predict(X_test)

            # Store results
            results[name] = {
                'best_params': best_params,
                'y_pred': y_pred,
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }

            print("Best Parameters:", best_params)
        else:
            # For models without hyperparameters
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            results[name] = {
                'y_pred': y_pred,
                'r2_score': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }

        # Visualization
        plot_regression_results(y_test, results[name]['y_pred'], name)
        plt.show()

    print("\n--- Regression Model Comparison ---")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"R² Score: {result['r2_score']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"MAE: {result['mae']:.4f}")
        if 'best_params' in result:
            print("Best Hyperparameters:", result['best_params'])

    return results


def main():
    # Load data
    df = pd.read_csv('../data/E-INSPIRE_I_master_catalogue.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    print("length=====", len(df))
    # Select features
    selected_features = ['MgFe', '[M/H]_mean_mass', 'velDisp_ppxf_res', 'age_mean_mass']

    # Preprocess data
    log_transform_config = {
        'age_mean_mass': False,
        'velDisp_ppxf_res': False,
        'MgFe': False,
        '[M/H]_mean_mass': False
    }

    X = preprocess_data(
        df,
        selected_features,
        log_transform_features=log_transform_config
    )
    y = df['DoR'].values


    results = comprehensive_regression_analysis(
        X,
        y,
        test_size=0.2,
        random_state=42,
        log_transform_features=log_transform_config
    )


if __name__ == "__main__":
    main()