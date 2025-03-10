# E-INSPIRE II

This project aims to analyze and cluster relic galaxies using various machine learning techniques.

## Data

The project utilizes the E-INSPIRE I master catalogue (`E-INSPIRE_I_master_catalogue.csv`) which contains key properties of the relic galaxy sample.

## Analysis Pipeline

The main analysis steps are:

1. **Regression**: Using a variety of regression methods to predict DoR. run ml_grouping.ipynb
2. **Cluster Stacking**: Stack the spectra of galaxies within each cluster to create composite spectra for further analysis. Run main.py.
3. **Spectral Fitting**: Fit the stacked spectra using the pPXF (penalized pixel-fitting) method to derive stellar population properties. Done as part of the above. 


## Results

The key outputs of the analysis include:

- Cluster assignments for each galaxy
- Stacked spectra for each cluster
- Stellar population properties derived from spectral fitting

All results are saved in the `data/` and `outputs/` directories.