import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


def clean_and_normalize_spectrum(wave, flux, ivar):
    good_idx = np.isfinite(flux) & np.isfinite(ivar)
    wave, flux, ivar = wave[good_idx], flux[good_idx], ivar[good_idx]

    target_wavelength = 5300
    closest_idx = np.argmin(np.abs(wave - target_wavelength))
    normalizing_wave = wave[closest_idx]

    if abs(normalizing_wave - target_wavelength) > 2:
        raise ValueError(f"Closest wavelength {normalizing_wave:.2f} is more than 2Å from target 5300Å")

    normalizing_flux = flux[closest_idx]
    if not np.isfinite(normalizing_flux) or normalizing_flux == 0:
        raise ValueError(f"Invalid normalizing flux value: {normalizing_flux}")

    flux = flux / normalizing_flux
    ivar = ivar * (normalizing_flux ** 2)

    return wave, flux, ivar


def load_spectrum(filename):
    with fits.open(filename) as hdul:
        coadd = hdul[1].data
        flux = coadd['flux']
        loglam = coadd['loglam']
        ivar = coadd['ivar']

        specobj = hdul[2].data
        z = specobj['Z'][0]
        wavelength = 10 ** loglam
        wavelength *= 1 / (1 + z)

        wavelength, flux, ivar = clean_and_normalize_spectrum(wavelength, flux, ivar)
        return wavelength, flux, ivar


def calculate_sigma_diff(wave, sigma_gal, sigma_fin):
    c = 299792.458
    ln_wave = np.log(wave)
    d_ln_wave = (ln_wave[-1] - ln_wave[0]) / (len(wave) - 1)
    velscale = c * d_ln_wave

    sigma_diff_kms = np.sqrt(sigma_fin ** 2 - sigma_gal ** 2)
    sigma_gal_pxl = sigma_gal / velscale
    sigma_fin_pxl = sigma_fin / velscale
    sigma_diff = np.sqrt(sigma_fin_pxl ** 2 - sigma_gal_pxl ** 2)

    return sigma_diff, sigma_diff_kms


def smooth_spectra(data, vdisps):
    sigma_fin = max(vdisps)
    smoothed_data = []
    sigma_ds = []

    for (wave, flux, ivar), sigma_gal in zip(data, vdisps):
        if sigma_fin <= sigma_gal:
            smoothed_flux = flux
            sigma_diff_kms = sigma_gal
        else:
            sigma_diff, sigma_diff_kms = calculate_sigma_diff(wave, sigma_gal, sigma_fin)
            smoothed_flux = gaussian_filter1d(flux, sigma_diff)

        smoothed_data.append([wave, smoothed_flux, ivar])
        sigma_ds.append(sigma_diff_kms)

    return smoothed_data, sigma_ds, sigma_fin


def resample_to_common_grid(smoothed_data):
    wave_min = max([s[0][0] for s in smoothed_data]) + 0.1
    wave_max = min([s[0][-1] for s in smoothed_data]) - 0.1
    print("NUM POINTS::::", min([len(s[0]) for s in smoothed_data]))
    wave_common = np.logspace(np.log10(wave_min), np.log10(wave_max), num=3828)

    resampled_data = []
    for wave, flux, ivar in smoothed_data:
        f = interpolate.interp1d(wave, flux, bounds_error=True, fill_value=np.nan)
        f_ivar = interpolate.interp1d(wave, ivar, bounds_error=True, fill_value=0)
        resampled_data.append([wave_common, f(wave_common), f_ivar(wave_common)])

    return resampled_data


def combine_spectra(aligned_spectra):
    wavelength = aligned_spectra[0][0]
    fluxes = [spec[1] for spec in aligned_spectra]
    ivars = [spec[2] for spec in aligned_spectra]

    combined_flux = np.mean(fluxes, axis=0)

    # Combine inverse variances
    ivar_stack = np.stack(ivars)
    mask = np.all((ivar_stack == 0) | np.isinf(ivar_stack), axis=0)
    combined_ivar = np.zeros_like(ivars[0])

    valid = ~mask
    if np.any(valid):
        valid_mask = (ivar_stack != 0) & np.isfinite(ivar_stack)
        valid_count = np.sum(valid_mask, axis=0)
        sum_inv_ivar = np.sum(1.0 / np.where(valid_mask, ivar_stack, np.inf), axis=0)
        has_valid = valid_count > 0
        combined_ivar[has_valid] = valid_count[has_valid] ** 2 / sum_inv_ivar[has_valid]

    # Plot diagnostics
    error = np.sqrt(1 / combined_ivar)
    """plt.figure(figsize=(10, 6))
    plt.errorbar(wavelength, combined_flux, yerr=error,
                 fmt='-', color='blue', ecolor='lightgray',
                 alpha=0.3, capsize=3, label='Spectrum')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.title('Spectrum with Uncertainties')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, error)
    plt.xlabel('Wavelength')
    plt.ylabel('Error')
    plt.title('Error Values vs Wavelength')
    plt.grid(True, alpha=0.3)
    """
    return wavelength, combined_flux, combined_ivar


def save_fits(wavelength, flux, combined_ivar, cluster_name, mgfe, sigma_fin):
    coadd_data = np.zeros(len(wavelength), dtype=[
        ('flux', 'f8'), ('wave', 'f8'),
        ('ivar', 'f8'), ('wdisp', 'f8')
    ])

    coadd_data['flux'] = flux
    coadd_data['wave'] = wavelength
    coadd_data['ivar'] = combined_ivar
    coadd_data['wdisp'] = np.full_like(wavelength, 2.76 / 2.355)

    primary_hdu = fits.PrimaryHDU()
    coadd_hdu = fits.BinTableHDU(data=coadd_data, name='COADD')

    primary_hdu.header['HIERARCH NAME'] = f'stacked_{cluster_name}'
    primary_hdu.header['HIERARCH z'] = 0
    primary_hdu.header['HIERARCH ALPHA'] = mgfe
    primary_hdu.header['HIERARCH SIGMA'] = sigma_fin

    hdul = fits.HDUList([primary_hdu, coadd_hdu])
    output_file = f'data/stacked_fits/stacked_{cluster_name}.fits'
    hdul.writeto(output_file, overwrite=True)
    print(output_file, "has been made.")
    return output_file


def stack_spectra(spectra, factor, cluster_name):
    # Load data and get properties
    catalogue = pd.read_csv('data/E-INSPIRE_I_master_catalogue.csv')
    data = []
    mgfe = []
    vdisps = []
    ages = []
    metallicity = []
    DoRs = []

    for filename in spectra:
        wave, flux, ivar = load_spectrum("data/fits_shortlist/" + filename)
        plate, mjd, fiber = map(int, re.match(r'spec-(\d{4})-(\d{5})-(\d{4})\.fits', filename).groups())

        matching_row = catalogue[
            (catalogue['plate'] == plate) &
            (catalogue['mjd'] == mjd) &
            (catalogue['fiberid'] == fiber)
            ]

        data.append([wave, flux, ivar])
        mgfe.append(float(matching_row['MgFe'].iloc[0]))
        vdisps.append(float(matching_row["velDisp_ppxf_res"].iloc[0]))
        ages.append(matching_row['age_mean_mass'].iloc[0])
        metallicity.append(matching_row['[M/H]_mean_mass'].iloc[0])
        DoRs.append(float(matching_row['DoR'].iloc[0]))

    # Print statistics
    print(f"STATS: ({len(spectra)} items)")
    print("--> Max vdisp:", max(vdisps))
    print("--> mgfe avg:", np.mean(mgfe))
    print("--> age avg:", np.mean(ages))
    print("--> metallicity avg:", np.mean(metallicity))
    print("--> DoR avg:", np.mean(DoRs))
    print("s.dev's:")
    print("--> mgfe std:", np.std(mgfe))
    print("--> age std:", np.std(ages))
    print("--> metallicity std:", np.std(metallicity))
    print("--> DoR std:", np.std(DoRs))

    # Process spectra
    smoothed_data, sigma_ds, sigma_fin = smooth_spectra(data, vdisps)
    resampled_data = resample_to_common_grid(smoothed_data)
    wavelength, flux, combined_ivar = combine_spectra(resampled_data)

    # Save results
    mgfe_avg = round(np.mean(mgfe), 1)
    save_fits(wavelength, flux, combined_ivar, cluster_name, mgfe_avg, sigma_fin)


if __name__ == '__main__':
    os.makedirs('data/stacked_fits', exist_ok=True)

    files = [
        "data/cluster_results/k-means_clusters.csv",
        "data/cluster_results/gmm_clusters.csv",
        "data/cluster_results/hierarchical_clusters.csv",
        "data/cluster_results/dor_clusters.csv"
    ]

    # Load cluster results
    catalogue = pd.read_csv("data/E-INSPIRE_I_master_catalogue.csv")
    cluster_results = {
        'Hierarchical': pd.read_csv(files[2]),
        'KMeans': pd.read_csv(files[0]),
        'GMM': pd.read_csv(files[1]),
        'DoR': pd.read_csv(files[3]),
    }


    cluster_groups = {
        'DoR': [
            cluster_results['DoR'][cluster_results['DoR']["Cluster"] == i]["SDSS_ID"].tolist() for i
            in range(3)],
        'Hierarchical': [
            cluster_results['Hierarchical'][cluster_results['Hierarchical']["Cluster"] == i]["SDSS_ID"].tolist() for i
            in range(3)],
        'KMeans': [cluster_results['KMeans'][cluster_results['KMeans']["Cluster"] == i]["SDSS_ID"].tolist() for i in
                   range(3)],
        'GMM': [cluster_results['GMM'][cluster_results['GMM']["Cluster"] == i]["SDSS_ID"].tolist() for i in
                range(max(cluster_results['GMM']["Cluster"]) + 1)]
    }


    colors = ['red', 'blue', 'green', 'black']
    n = 0.001635
    factor = [n * 4, n * 2.7, n * 2.5]

    for method, groups in cluster_groups.items():
        print(f"\n================================")
        method_labels = [f"{method}_{i}" for i in range(len(groups))]
        method_colors = colors[:len(groups)]
        method_factors = factor[:len(groups)]

        for spectra, color, label, factor_val in zip(groups, method_colors, method_labels, method_factors):
            print(f"\nStacking {label}")
            stack_spectra(spectra, factor_val, label)
        print(f"================================")