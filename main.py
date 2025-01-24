import os
from scripts.stack_spectra import stack_spectra
import pandas as pd
from scripts.stacked_ppxf_fitting import fit_spectra
from pathlib import Path


def get_cluster_files(cluster_dir):
    """Get all CSV files from cluster results directory."""
    cluster_path = Path(cluster_dir)
    return list(cluster_path.glob('*.csv'))


def get_method_name(filename):
    """Extract method name from filename (e.g., 'k-means' from 'k-means_clusters.csv')"""
    return filename.stem.split('_')[0].capitalize()


def create_cluster_groups(cluster_files, catalogue):
    cluster_groups = {}

    # Add DoR groups
    dor_clusters = []
    for threshold in [(0.6, float('inf')), (0.3, 0.6), (0, 0.3)]:
        dor_group = catalogue[(catalogue['DoR'] > threshold[0]) & (catalogue['DoR'] <= threshold[1])]
        file_list = [f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
                     for plate, mjd, fiber in zip(dor_group['plate'], dor_group['mjd'], dor_group['fiberid'])]
        dor_clusters.append(file_list)
    cluster_groups['DoR'] = dor_clusters

    # Add clustering method groups
    for file_path in cluster_files:
        df = pd.read_csv(file_path)
        method = get_method_name(file_path)
        n_clusters = max(df["Cluster"]) + 1
        cluster_groups[method] = [df[df["Cluster"] == i]["SDSS_ID"].tolist()
                                  for i in range(n_clusters)]

    return cluster_groups


def do_stacking():
    # Create output directory if it doesn't exist
    os.makedirs('data/stacked_fits', exist_ok=True)

    # Load catalogue
    catalogue = pd.read_csv("data/E-INSPIRE_I_master_catalogue.csv")

    # Get cluster result files
    cluster_files = get_cluster_files("data/cluster_results")

    # Create cluster groups
    cluster_groups = create_cluster_groups(cluster_files, catalogue)

    # Define colors and factors (extend if needed for methods with more clusters)
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # Add more colors if needed
    n = 0.001635
    base_factors = [4, 2.7, 2.5, 2.3, 2.1]  # Add more factors if needed
    factors = [n * f for f in base_factors]

    # Process each clustering method
    for method, groups in cluster_groups.items():
        print(f"\n================================")
        method_labels = [f"{method}_{i}" for i in range(len(groups))]
        method_colors = colors[:len(groups)]
        method_factors = factors[:len(groups)]

        for idx, (spectra, color, label, factor_val) in enumerate(
                zip(groups, method_colors, method_labels, method_factors)):
            print(f"\nStacking {label}")
            stack_spectra(spectra, factor_val, label)
        print(f"================================")


if __name__ == "__main__":
    do_stacking()
    fit_spectra()