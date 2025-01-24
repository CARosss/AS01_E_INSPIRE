import os
from pathlib import Path
import pandas as pd
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
        if directory.startswith('outputs/'):  # Only clear output directories, not data
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


def get_cluster_assignments():
    """Load and organize cluster assignments"""
    # Load raw results
    cluster_files = [
        "data/cluster_results/k-means_clusters.csv",
        "data/cluster_results/gmm_clusters.csv",
        "data/cluster_results/hierarchical_clusters.csv"
    ]

    cluster_results = {
        'KMeans': pd.read_csv(cluster_files[0]),
        'GMM': pd.read_csv(cluster_files[1]),
        'Hierarchical': pd.read_csv(cluster_files[2])
    }

    # Get DoR groups
    catalogue = pd.read_csv("data/E-INSPIRE_I_master_catalogue.csv")
    dor_clusters = []
    for threshold in [(0.6, float('inf')), (0.3, 0.6), (0, 0.3)]:
        dor_group = catalogue[(catalogue['DoR'] > threshold[0]) & (catalogue['DoR'] <= threshold[1])]
        file_list = [f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
                     for plate, mjd, fiber in zip(dor_group['plate'], dor_group['mjd'], dor_group['fiberid'])]
        dor_clusters.append(file_list)

    # Organize all clusters
    cluster_groups = {
        'DoR': dor_clusters,
        'Hierarchical': [
            cluster_results['Hierarchical'][cluster_results['Hierarchical']["Cluster"] == i]["SDSS_ID"].tolist()
            for i in range(3)],
        'KMeans': [cluster_results['KMeans'][cluster_results['KMeans']["Cluster"] == i]["SDSS_ID"].tolist()
                   for i in range(3)],
        'GMM': [cluster_results['GMM'][cluster_results['GMM']["Cluster"] == i]["SDSS_ID"].tolist()
                for i in range(max(cluster_results['GMM']["Cluster"]) + 1)]
    }

    return cluster_groups


def process_clusters():
    cluster_groups = get_cluster_assignments()

    # Define parameters for each group
    n = 0.001635
    factors = [n * 4, n * 2.7, n * 2.5]  # extend if needed
    colors = ['red', 'blue', 'green']  # extend if needed

    # Process each clustering method
    for method, groups in cluster_groups.items():
        print(f"\n================================")
        print(f"Processing {method} clusters")

        # Create labels and get appropriate number of factors/colors
        method_labels = [f"{method}_{i}" for i in range(len(groups))]
        method_colors = colors[:len(groups)]
        method_factors = factors[:len(groups)]

        # Stack spectra for each cluster
        for spectra, color, label, factor in zip(groups, method_colors,
                                                 method_labels, method_factors):
            print(f"\nStacking {label}")
            stack_spectra(spectra, factor, label)

        print(f"================================")


def main():
    # Setup
    setup_directories()

    # Run analysis
    process_clusters()
    fit_spectra()


    print("\nAnalysis complete!")
    print("Check outputs/stacked_catalogues for results.")


if __name__ == "__main__":
    main()