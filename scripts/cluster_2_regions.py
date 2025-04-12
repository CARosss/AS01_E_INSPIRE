import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.gridspec as gridspec
import os

def plot_combined_regression_and_residuals_single_threshold(df, features, target='DoR', n_splits=5, model_params=None,
                                           threshold=0.6, output_path=None):

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 15,
    })

    # Default model parameters if none provided
    if model_params is None:
        seed = 42  # Define a seed for reproducibility
        model_params = {
            'max_depth': 22,
            'max_features': 0.793,
            'max_samples': 0.500,
            'min_samples_leaf': 1,
            'min_samples_split': 4,
            'n_estimators': 460,
            'random_state': seed
        }

    X = df[features]
    y = df[target]

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=model_params.get('random_state', 42))

    # Initialize arrays to store all predictions
    all_true = np.zeros_like(y)
    all_pred = np.zeros_like(y)
    all_residuals = np.zeros_like(y)
    fold_indices = np.zeros_like(y, dtype=int)  # Track which fold each point belongs to
    fold_results = {
        'r2_scores': [],
        'indices': [],
        'predictions': [],
        'true_values': [],
        'true_values_all': y.values,
        'feature_importances': []  # Store feature importances from each fold
    }

    # Define colors for different folds
    fold_colors = plt.cm.tab10(np.linspace(0, 1, n_splits))

    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        # Split and scale data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model and make predictions
        rf_model = RandomForestRegressor(**model_params)
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)

        # Store feature importances
        fold_results['feature_importances'].append(
            dict(zip(features, rf_model.feature_importances_))
        )

        # Store results for this fold
        r2 = r2_score(y_test, y_pred)
        fold_results['r2_scores'].append(r2)
        fold_results['indices'].append(test_idx)
        fold_results['predictions'].append(y_pred)
        fold_results['true_values'].append(y_test.values)

        # Store predictions and residuals at the correct indices
        all_true[test_idx] = y_test
        all_pred[test_idx] = y_pred
        all_residuals[test_idx] = y_test - y_pred
        fold_indices[test_idx] = fold  # Track which fold this point belongs to

        print(f"Fold {fold}/{n_splits} - R²: {r2:.4f}")

    # Calculate overall R² for all folds combined
    overall_r2 = r2_score(all_true, all_pred)
    mean_fold_r2 = np.mean(fold_results['r2_scores'])
    std_fold_r2 = np.std(fold_results['r2_scores'])

    # Calculate overall RMSE and MAE
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae = mean_absolute_error(all_true, all_pred)

    print(f"\nCross-validation results:")
    print(f"Mean fold R²: {mean_fold_r2:.4f} (±{std_fold_r2:.4f})")
    print(f"Overall R² (all predictions): {overall_r2:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")

    # Common plot limits
    min_val = min(min(all_true), min(all_pred))
    max_val = max(max(all_true), max(all_pred))
    buffer = (max_val - min_val) * 0.02
    plot_min = min_val - buffer
    plot_max = max_val + buffer

    # Create figure with gridspec for layout control
    fig = plt.figure(figsize=(10, 16))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    # Top subplot: True vs Predicted
    ax1 = plt.subplot(gs[0])

    # Plot each fold with different colors - x=predicted, y=true
    for fold in range(1, n_splits + 1):
        mask = fold_indices == fold
        ax1.scatter(all_pred[mask], all_true[mask], alpha=0.7, s=35,
                    color=fold_colors[fold - 1], edgecolors='k', linewidths=0.5,
                    label=f'Fold {fold} (R²: {fold_results["r2_scores"][fold - 1]:.4f})')

    # Add perfect prediction line
    ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'r--',
             label='Perfect prediction', linewidth=1.5)

    # Add vertical line at threshold
    ax1.axvline(x=threshold, color='gray', alpha=0.2, linestyle='-.', linewidth=2.0)

    # Add small gray label next to the line
    ax1.text(threshold + 0.01, plot_max - buffer, f'DoR = {threshold}',
             style='italic', color='gray', fontsize=10, va='top', rotation=90)

    # Customize top plot
    ax1.set_ylabel(f'True {target}', fontsize=25)
    # Don't show x-label on top plot
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)

    # Add textbox with performance metrics for top plot
    props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
    textstr = '\n'.join((
        r'$\mathrm{Mean}\ R^2 = %.4f\ (\pm%.4f)$' % (mean_fold_r2, std_fold_r2),
    ))
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', bbox=props)

    # Add legend to top plot
    ax1.legend(loc='lower right', fontsize=10)

    # Add grid and styling to top plot
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='both', direction='in', labelsize=14)
    ax1.minorticks_on()
    for spine in ax1.spines.values():
        spine.set_color('lightgray')

    # Hide x-tick labels on top plot
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Bottom subplot: Residuals vs Predicted
    ax2 = plt.subplot(gs[1], sharex=ax1)  # Share x-axis with top plot

    # Plot residuals vs predicted values for each fold with different colors
    for fold in range(1, n_splits + 1):
        mask = fold_indices == fold
        ax2.scatter(all_pred[mask], all_residuals[mask], alpha=0.7, s=35,
                    color=fold_colors[fold - 1], edgecolors='k', linewidths=0.5,
                    label=f'Fold {fold}')

    # Add horizontal line at y=0
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Perfect prediction')

    # Calculate region-specific RMSE by predicted DoR
    below_mask = all_pred <= threshold
    above_mask = all_pred > threshold

    below_rmse = np.sqrt(mean_squared_error(all_true[below_mask], all_pred[below_mask])) if np.any(below_mask) else np.nan
    above_rmse = np.sqrt(mean_squared_error(all_true[above_mask], all_pred[above_mask])) if np.any(above_mask) else np.nan

    # Add vertical line at threshold
    ax2.axvline(x=threshold, color='gray', alpha=0.2, linestyle='-.', linewidth=2.0)

    # Add label next to the line
    ymin, ymax = ax2.get_ylim()
    ax2.text(threshold + 0.01, ymin * 0.9, f'DoR = {threshold}',
             style='italic', color='gray', fontsize=10, va='bottom', rotation=90)

    # Add region-specific metrics in a textbox for bottom plot
    props = dict(boxstyle='square', facecolor='white', alpha=0.8, edgecolor='lightgray')
    textstr = '\n'.join((
        r'$\mathrm{Overall\ RMSE} = %.4f$' % (rmse,),
        r'$\mathrm{RMSE\ (DoR \leq %.2f)} = %.4f$' % (threshold, below_rmse),
        r'$\mathrm{RMSE\ (DoR > %.2f)} = %.4f$' % (threshold, above_rmse)
    ))
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
             verticalalignment='top', bbox=props, horizontalalignment='left')

    # Customize bottom plot
    ax2.set_xlabel(f'Predicted {target}', fontsize=25)  # Only bottom plot gets x-label
    ax2.set_ylabel('Residuals (True - Predicted)', fontsize=16)

    # Add grid and styling to bottom plot
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=14)
    ax2.minorticks_on()
    for spine in ax2.spines.values():
        spine.set_color('lightgray')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    # Create a consolidated DataFrame with all k-fold predictions
    all_indices = []
    all_predictions = []

    for i in range(len(fold_results['indices'])):
        all_indices.extend(fold_results['indices'][i])
        all_predictions.extend(fold_results['predictions'][i])

    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        'index': all_indices,
        'predicted_DoR': all_predictions
    })

    # Sort by index to match the original DataFrame order
    pred_df = pred_df.sort_values('index').reset_index(drop=True)

    # Add consolidated predictions to fold_results
    fold_results['pred_df'] = pred_df
    fold_results['all_true'] = all_true
    fold_results['all_pred'] = all_pred
    fold_results['all_residuals'] = all_residuals
    fold_results['fold_indices'] = fold_indices
    fold_results['plot_limits'] = {'min': plot_min, 'max': plot_max}

    # Average feature importances across folds
    avg_importances = {}
    for feature in features:
        avg_importances[feature] = np.mean([fold_imp[feature] for fold_imp in fold_results['feature_importances']])

    fold_results['avg_feature_importances'] = avg_importances

    return fig, fold_results

def plot_feature_importance(fold_results, features, output_path=None, feature_display_names=None):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern",
        "figure.dpi": 300,
        "font.size": 15,
    })

    # Get average feature importances
    avg_importances = fold_results['avg_feature_importances']

    # Sort features by importance
    sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
    feature_names = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]

    # If display names are provided, use them for the plot
    if feature_display_names:
        display_names = [feature_display_names.get(name, name) for name in feature_names]
    else:
        display_names = feature_names

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create horizontal bar chart
    bars = ax.barh(range(len(feature_names)), importances, height=0.6, align='center')

    # Customize colors based on importance
    colors = plt.cm.Blues(np.linspace(0.5, 1, len(feature_names)))
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])

    # Add feature names and customize plot
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Feature Importance (MDI)', fontsize=16)

    # Add grid and styling
    ax.grid(True, axis='x', alpha=0.3)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=14)

    for spine in ax.spines.values():
        spine.set_color('lightgray')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

    return fig

def export_dor_groups_from_kfold_single_threshold(df, pred_df, output_path, threshold=0.6,
                                 csv_name='DoR_Groups_KFold.csv'):
    os.makedirs(output_path, exist_ok=True)

    # Create SDSS identifiers
    sdss_ids = [
        f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
        for plate, mjd, fiber in zip(df['plate'], df['mjd'], df['fiberid'])
    ]

    # Get predicted DoR values from k-fold cross-validation
    # Make sure indices are properly aligned
    dor_pred = np.zeros(len(df))
    for idx, pred_val in zip(pred_df['index'], pred_df['predicted_DoR']):
        dor_pred[idx] = pred_val

    # Create labels based on predicted DoR (binary classification)
    labels = pd.Series(0, index=df.index)  # below threshold
    labels[dor_pred > threshold] = 1  # above threshold

    # Save results
    results_df = pd.DataFrame({
        'SDSS_ID': sdss_ids,
        'Cluster': labels,
        'True_DoR': df['DoR'],
        'Predicted_DoR': dor_pred
    })

    filename = csv_name
    results_df.to_csv(filename, index=False)

    print(f"\nSaved DoR groups to {filename}")
    print("Group sizes (based on predicted DoR):")
    print(f"Predicted DoR <= {threshold}: {sum(labels == 0)}")
    print(f"Predicted DoR > {threshold}: {sum(labels == 1)}")

    plt.show()

    return results_df

def run_kfold_prediction_combined_single_threshold(df, features, target='DoR', n_splits=5, model_params=None,
                                  threshold=0.6, output_path='outputs/make_plots_output',
                                  combined_figure_name='regression_performance_2regions.pdf',
                                  feature_importance_name='feature_importance.pdf',
                                  csv_name='data/cluster_results/regression_clusters.csv',
                                  feature_display_names=None):
    import os

    os.makedirs(output_path, exist_ok=True)

    # Step 1: Run k-fold cross-validation with combined plot
    combined_figure_path = os.path.join(output_path, combined_figure_name)
    fig, results = plot_combined_regression_and_residuals_single_threshold(
        df=df,
        features=features,
        target=target,
        n_splits=n_splits,
        model_params=model_params,
        threshold=threshold,
        output_path=combined_figure_path,
    )

    # Step 2: Plot feature importance
    importance_path = os.path.join(output_path, feature_importance_name)
    importance_fig = plot_feature_importance(
        fold_results=results,
        features=features,
        output_path=importance_path,
        feature_display_names=feature_display_names
    )

    # Step 3: Extract the prediction DataFrame from results
    pred_df = results['pred_df']

    # Step 4: Export DoR groups based on k-fold predictions
    results_df = export_dor_groups_from_kfold_single_threshold(
        df=df,
        pred_df=pred_df,
        threshold=threshold,
        output_path=output_path,
        csv_name=csv_name
    )

    return fig, importance_fig, results, results_df

def run_clustering_single_threshold(file='data/E-INSPIRE_I_master_catalogue.csv', threshold=0.6, output_path='outputs/make_plots_output'):
    FEATURES = [
        'MgFe',
        '[M/H]_mean_mass',
        '[M/H]_err_mass',
        'velDisp_ppxf_res',
        'velDisp_ppxf_err_res',
        'age_mean_mass',
        'age_err_mass',
        # 'meanRad_r',
        # 'logM*',
    ]

    MODEL_PARAMS = {
        'max_depth': 8,
        'max_features': 0.8,
        'max_samples': 0.7,
        'min_samples_leaf': 3,
        'min_samples_split': 5,
        'n_estimators': 50,
        'random_state': 1
    }

    FEATURE_DISPLAY_NAMES = {
        'MgFe': r'$[\mathrm{Mg}/\mathrm{Fe}]$ (dex)',  # Magnesium to Iron ratio
        '[M/H]_mean_mass': r'$[\mathrm{M}/\mathrm{H}]$ (dex)',  # Metallicity
        '[M/H]_err_mass': r'$[\mathrm{M}/\mathrm{H}]$ error (dex)',  # Metallicity error
        'velDisp_ppxf_res': r'$\sigma_{\star}$ (km/s)',  # Velocity dispersion
        'velDisp_ppxf_err_res': r'$\sigma_{\star}$ error (km/s)',  # Velocity dispersion error
        'age_mean_mass': r'Age (Gyr)',  # Age in Gigayears
        'age_err_mass': r'Age error (Gyr)',  # Age error
        # 'meanRad_r': r'$R_{\mathrm{mean}}$ (kpc)',  # Mean radius
        # 'logM*': r'log($M_{\star}/M_{\odot}$)'  # Stellar mass
    }

    print(MODEL_PARAMS)
    print(FEATURES)

    # Read your data
    df = pd.read_csv(file)

    threshold_str = str(threshold)[2:]

    print(f"Running single-threshold regression analysis with threshold={threshold}...")
    fig, importance_fig, results, results_df = run_kfold_prediction_combined_single_threshold(
        df=df,
        features=FEATURES,
        target='DoR',
        n_splits=5,
        model_params=MODEL_PARAMS,
        threshold=threshold,
        output_path=output_path,
        combined_figure_name='regression_performance_'+threshold_str+'_2regions.pdf',
        feature_importance_name='feature_importance_'+threshold_str+'_2regions.pdf',
        csv_name='outputs/cluster_results/regression_clusters.csv',
        feature_display_names=FEATURE_DISPLAY_NAMES
    )

    return fig, importance_fig, results, results_df

if __name__ == "__main__":
    run_clustering_single_threshold(file='data/E-INSPIRE_I_master_catalogue.csv', threshold=0.6, output_path='outputs/make_plots_output')