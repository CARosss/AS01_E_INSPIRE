# E-INSPIRE II

This project aims to analyze and cluster relic galaxies using various machine learning techniques. The goal is to gain insights into the properties and evolution of these galaxies.

## Data

The project utilizes the E-INSPIRE I master catalogue (`E-INSPIRE_I_master_catalogue.csv`) which contains key properties of the relic galaxy sample.

## Analysis Pipeline

The main analysis steps are:


1. **Clustering**: Apply different clustering algorithms including K-Means, Gaussian Mixture Models (GMM), and Hierarchical Clustering. Do so unweighted, weighted using regression results, and grid-searched for the best weights. 

2. **Regression**: Using a variety of regression methods to predict DoR
3. **Cluster Stacking**: Stack the spectra of galaxies within each cluster to create composite spectra for further analysis.

4. **Spectral Fitting**: Fit the stacked spectra using the pPXF (penalized pixel-fitting) method to derive stellar population properties.


## Results

The key outputs of the analysis include:

- Cluster assignments for each galaxy
- Stacked spectra for each cluster
- Stellar population properties derived from spectral fitting

All results are saved in the `data/` and `outputs/` directories.

## Usage

First, run all on the clustering and/or regression notebooks. 
To run the full analysis pipeline on clusters, simply execute:

```
python main.py
```

This will stack spectra, fit the stacked spectra, and generate all output files and plots.
