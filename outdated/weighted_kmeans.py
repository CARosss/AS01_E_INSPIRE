# weighted_kmeans_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import seaborn as sns
from itertools import combinations


def create_dor_classes(dor_values):
    classes = np.zeros_like(dor_values, dtype=int)
    classes[(dor_values >= 0.3) & (dor_values < 0.6)] = 1
    classes[dor_values >= 0.6] = 2
    return classes


def preprocess_data(df, selected_features):
    X = df[selected_features].copy()

    if 'age_mean_mass' in X.columns:
        X['age_mean_mass'] = np.log10(X['age_mean_mass'])
    if 'velDisp_ppxf_res' in X.columns:
        X['velDisp_ppxf_res'] = np.log10(X['velDisp_ppxf_res'])

    return X


def calculate_cluster_score(labels, dor_values):

    # Get mean DoR for each cluster
    cluster_means = np.array([np.mean(dor_values[labels == i]) for i in range(3)])

    # Ensure clusters are ordered by DoR (lowest to highest)
    sorted_indices = np.argsort(cluster_means)
    remapped_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(sorted_indices):
        remapped_labels[labels == old_label] = new_label

    # Calculate mean absolute difference between each point's DoR
    # and its assigned cluster's mean DoR
    differences = np.abs(dor_values - cluster_means[remapped_labels])

    # Convert to a score between 0 and 1 where 1 is best
    # We use exp(-x) to convert differences to similarities
    score = np.exp(-np.mean(differences))

    return score


def generate_weight_combinations(min_weight=0.05, max_weight=0.85, n_steps=20):
    weights = np.linspace(min_weight, max_weight, n_steps)
    valid_combinations = []

    for w1 in weights:
        for w2 in weights:
            for w3 in weights:
                w4 = 1 - (w1 + w2 + w3)
                if min_weight <= w4 <= max_weight:
                    valid_combinations.append([w1, w2, w3, w4])

    return np.array(valid_combinations)


def refine_best_weights(X_scaled, y_dor, best_weights, delta=0.01):
    improved_weights = best_weights.copy()
    best_score = calculate_cluster_score(
        KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled * best_weights),
        y_dor
    )

    improved = True
    while improved:
        improved = False
        for i in range(len(best_weights)):
            for change in [-delta, delta]:
                test_weights = improved_weights.copy()
                test_weights[i] += change
                test_weights = test_weights / test_weights.sum()

                labels = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled * test_weights)
                score = calculate_cluster_score(labels, y_dor)

                if score > best_score:
                    improved_weights = test_weights
                    best_score = score
                    improved = True

    return improved_weights, best_score


def cross_validate_weights(X_scaled, y_dor, weights, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_dor[train_idx], y_dor[val_idx]

        # Train on training set
        kmeans = KMeans(n_clusters=3, random_state=42)
        train_labels = kmeans.fit_predict(X_train * weights)

        # Predict on validation set
        val_labels = kmeans.predict(X_val * weights)

        # Get mean DoR for each cluster to determine proper ordering
        train_cluster_dors = [np.mean(y_train[train_labels == i]) for i in range(3)]
        ordering = np.argsort(train_cluster_dors)

        # Reorder validation labels according to training order
        val_labels_reordered = np.zeros_like(val_labels)
        for i, order in enumerate(ordering):
            val_labels_reordered[val_labels == order] = i

        # Calculate score with reordered labels
        score = calculate_cluster_score(val_labels_reordered, y_val)
        scores.append(score)

    return np.mean(scores), np.std(scores)

def plot_score_landscape(results_df, X_columns):
    parameter_pairs = list(combinations(range(4), 2))
    n_pairs = len(parameter_pairs)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()

    for idx, (i, j) in enumerate(parameter_pairs):
        # Create 2D histogram of scores
        hist = axes[idx].hist2d(results_df[f'w{i + 1}'],
                                results_df[f'w{j + 1}'],
                                weights=results_df['score'],
                                bins=20,
                                cmap='viridis')
        plt.colorbar(hist[3], ax=axes[idx], label='Average Score')

        axes[idx].set_xlabel(f'{X_columns[i]} weight')
        axes[idx].set_ylabel(f'{X_columns[j]} weight')
        axes[idx].set_title(f'Score Landscape: {X_columns[i]} vs {X_columns[j]}')

    plt.tight_layout()
    plt.show()


def grid_kmeans_analysis(X, y_dor, n_steps=20, min_weight=0.05, max_weight=0.85, random_state=42):
    np.random.seed(random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate weight combinations
    print("Generating valid weight combinations...")
    weight_combinations = generate_weight_combinations(min_weight, max_weight, n_steps)
    print(f"Found {len(weight_combinations)} valid combinations")

    best_score = -1
    best_weights = None
    best_labels = None
    results = []

    # Try each combination
    for i, weights in enumerate(weight_combinations):
        if i % 100 == 0:
            print(f"Testing combination {i}/{len(weight_combinations)}")

        X_weighted = X_scaled * weights
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X_weighted)

        score = calculate_cluster_score(labels, y_dor)

        results.append({
            'weights': weights,
            'score': score,
            'labels': labels
        })

        if score > best_score:
            best_score = score
            best_weights = weights
            best_labels = labels

    # Refine best weights
    print("\nRefining best weights...")
    refined_weights, refined_score = refine_best_weights(X_scaled, y_dor, best_weights)

    if refined_score > best_score:
        best_weights = refined_weights
        best_score = refined_score
        best_labels = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_scaled * refined_weights)

    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_score_mean, cv_score_std = cross_validate_weights(X_scaled, y_dor, best_weights)

    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {'w1': r['weights'][0],
         'w2': r['weights'][1],
         'w3': r['weights'][2],
         'w4': r['weights'][3],
         'score': r['score']}
        for r in results
    ])

    # Plotting
    plot_score_landscape(results_df, X.columns)

    # Final visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Parameter weights
    axes[0, 0].bar(X.columns, best_weights)
    axes[0, 0].set_title('Best Parameter Weights')
    axes[0, 0].set_xticklabels(X.columns, rotation=45)

    # DoR distribution
    sns.boxplot(x=best_labels, y=y_dor, ax=axes[0, 1])
    axes[0, 1].set_title('DoR Distribution in Clusters')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('DoR')

    # Score distribution
    axes[1, 0].hist(results_df['score'], bins=30)
    axes[1, 0].axvline(best_score, color='r', linestyle='--',
                       label=f'Best Score: {best_score:.3f}')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].legend()

    # Cluster sizes
    cluster_sizes = np.bincount(best_labels)
    axes[1, 1].bar(range(len(cluster_sizes)), cluster_sizes)
    axes[1, 1].set_title('Cluster Sizes')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Galaxies')

    plt.tight_layout()
    plt.show()

    # Print results
    print("\nFinal Results:")
    print(f"Best score: {best_score:.3f}")
    print(f"Cross-validation score: {cv_score_mean:.3f} ± {cv_score_std:.3f}")
    print("\nCluster sizes:")

    cluster_dors = [np.mean(y_dor[best_labels == i]) for i in range(3)]
    print("\nCluster sizes and mean DoR:")
    for i, (size, mean_dor) in enumerate(zip(cluster_sizes, cluster_dors)):
        print(f"Cluster {i}: {size} galaxies, mean DoR = {mean_dor:.3f}")


    print("\nBest weights:")
    for feature, weight in zip(X.columns, best_weights):
        print(f"{feature}: {weight:.3f}")

    return best_weights, best_labels, results_df


def main():
    # Load data
    df = pd.read_csv('data/E-INSPIRE_I_master_catalogue.csv')
    selected_features = ['MgFe', '[M/H]_mean_mass', 'velDisp_ppxf_res', 'age_mean_mass']

    print("Initial DoR distribution:")
    print(f"DoR < 0.3: {len(df[df['DoR'] < 0.3])}")
    print(f"0.3 ≤ DoR < 0.6: {len(df[(df['DoR'] >= 0.3) & (df['DoR'] < 0.6)])}")
    print(f"DoR ≥ 0.6: {len(df[df['DoR'] >= 0.6])}")

    # Preprocess
    X = preprocess_data(df, selected_features)
    y_dor = df['DoR'].values

    best_weights, best_labels, results_df = grid_kmeans_analysis(X, y_dor, n_steps=50, max_weight=0.65)


if __name__ == "__main__":
    main()