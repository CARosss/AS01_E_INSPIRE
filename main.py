import os
from pathlib import Path
import pandas as pd
import numpy as np
from scripts.stack_spectra import stack_spectra
from scripts.stacked_ppxf_fitting import fit_spectra


def setup_directories():
    directories = [
        'data/stacked_fits',
        'outputs/ppxf_fits',
        'outputs/sfh_plots',
        'outputs/stacked_catalogues'
    ]

    # Clear existing output directories
    for directory in directories:
        if directory.startswith('outputs/'):
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f'Error deleting {file_path}: {e}')

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def verify_cluster_order(spectra_list, catalogue_path='data/E-INSPIRE_I_master_catalogue.csv'):
    """Verify cluster ordering by calculating mean DoR for each cluster"""
    catalogue = pd.read_csv(catalogue_path)

    cluster_stats = []
    for cluster_spectra in spectra_list:
        # Extract plate, mjd, fiber from filenames
        cluster_data = []
        for filename in cluster_spectra:
            parts = filename.replace('spec-', '').replace('.fits', '').split('-')
            plate, mjd, fiber = map(int, parts)

            # Find matching row in catalogue
            mask = (catalogue['plate'] == plate) & \
                   (catalogue['mjd'] == mjd) & \
                   (catalogue['fiberid'] == fiber)

            if any(mask):
                cluster_data.append(catalogue.loc[mask, 'DoR'].iloc[0])

        mean_dor = np.mean(cluster_data) if cluster_data else 0
        cluster_stats.append(mean_dor)

    # Check if clusters are in descending DoR order
    current_order = np.array(cluster_stats)
    expected_order = np.sort(current_order)[::-1]  # Sort in descending order

    if not np.array_equal(current_order, expected_order):
        # Reorder clusters to match descending DoR
        sort_indices = np.argsort(current_order)[::-1]
        return [spectra_list[i] for i in sort_indices]

    return spectra_list


def get_cluster_assignments():
    """Load and organize cluster assignments from all *_clusters.csv files"""
    cluster_dir = "data/cluster_results"
    cluster_groups = {}

    # Get all cluster files
    cluster_files = [f for f in os.listdir(cluster_dir) if f.endswith('_clusters.csv')]

    # Process each file
    for file in cluster_files:
        # Extract method name from filename (remove _clusters.csv)
        method = file.replace('_clusters.csv', '').upper()

        try:
            # Load the cluster results
            df = pd.read_csv(os.path.join(cluster_dir, file))

            # Create groups for clusters 0, 1, and 2
            groups = [
                df[df["Cluster"] == i]["SDSS_ID"].tolist()
                for i in range(3)  # We know each file has exactly 3 clusters
            ]

            # Verify and potentially reorder clusters
            groups = verify_cluster_order(groups)
            cluster_groups[method] = groups

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    return cluster_groups


def process_clusters():
    cluster_groups = get_cluster_assignments()

    # Define parameters for visualization (fixed for 3 clusters)
    n = 0.001635
    factors = [n * 4, n * 2.7, n * 2.5]
    colors = ['red', 'blue', 'green']

    # Process each clustering method
    for method, groups in cluster_groups.items():
        print(f"\n================================")
        print(f"Processing {method} clusters")

        # Create labels
        method_labels = [f"{method}_{i}" for i in range(3)]

        # Stack spectra for each cluster
        for spectra, color, label, factor in zip(groups, colors, method_labels, factors):
            print(f"\nStacking {label}")
            stack_spectra(spectra, factor, label)

        print(f"================================")


def main():
    # Setup
    setup_directories()

    # Run analysis
    process_clusters()
    fit_spectra(nrand=9)

    print("\nAnalysis complete!")
    print("Check outputs/stacked_catalogues for results.")


if __name__ == "__main__":
    main()