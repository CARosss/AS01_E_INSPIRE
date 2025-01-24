from os import putenv
from time import perf_counter as clock
from pathlib import Path
from urllib import request
import numpy as np
from astropy.io import fits
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

def stack_spectra(spectra, factor, cluster_name):
    import numpy as np
    data = []

    def clean_and_normalize_spectrum(wave, flux, ivar):
        # Remove NaN and inf values
        good_idx = np.isfinite(flux) & np.isfinite(ivar)
        wave, flux, ivar = wave[good_idx], flux[good_idx], ivar[good_idx]

        # normalise to flux at 5300
        target_wavelength = 5300
        tolerance = 2
        closest_idx = np.argmin(np.abs(wave - target_wavelength))
        normalizing_wave = wave[closest_idx]

        if abs(normalizing_wave - target_wavelength) > tolerance:
            raise ValueError(f"Closest wavelength {normalizing_wave:.2f} is more than "
                             f"{tolerance}Å from target {target_wavelength}Å")

        normalizing_flux = flux[closest_idx]

        if not np.isfinite(normalizing_flux) or normalizing_flux == 0:
            raise ValueError(f"Invalid normalizing flux value: {normalizing_flux}")
        # print("stats of the final individual spectrum:")

        noise = (1/np.sqrt(ivar))
        SNR = flux/(1/np.sqrt(ivar))
        # print("SNR before normalising", SNR[0:5])
        # print("noise unnorm", noise[0:5])

        flux = flux / normalizing_flux
        ivar = ivar * (normalizing_flux ** 2)

        SNR = flux/(1/np.sqrt(ivar))
        noise = (1/np.sqrt(ivar))
        # print("noise norm", noise[0:5])
        # print("SNR after normalising", SNR[0:5])

        # print("Mean SNR and sum SNR of stacked", np.mean(SNR), np.sum(SNR))
        # print("Mean flux of an individual stacking:", np.mean(flux))
        # print("After normalization:")
        mask = (wave >= 6500) & (wave <= 7000)
        # print(f"Ivar stats in 6500-7000Å range:")
        # print(f"Min ivar in range: {np.min(ivar[mask])}")
        # print(f"Max ivar in range: {np.max(ivar[mask])}")

        # print("stats of the stacked spectrum:")
        return wave, flux, ivar


    def load_spectrum(filename):
        with fits.open(filename) as hdul:
            coadd = hdul[1].data  # Extension 1 contains the spectrum
            flux = coadd['flux']
            loglam = coadd['loglam']
            ivar = coadd['ivar']

            specobj = hdul[2].data
            z = specobj['Z'][0]
            wavelength = 10 ** loglam
            wavelength *= 1 / (1 + z)
            # Add these diagnostic prints
            # print(f"Initial ivar check for {filename}:")
            # print(f"Min ivar: {np.min(ivar)}")
            # print(f"Max ivar: {np.max(ivar)}")
            mask = (wavelength >= 6500) & (wavelength <= 7000)
            # print(f"Ivar stats in 6500-7000Å range:")
            # print(f"Min ivar in range: {np.min(ivar[mask])}")
            # print(f"Max ivar in range: {np.max(ivar[mask])}")
            wavelength, flux, ivar = clean_and_normalize_spectrum(wavelength, flux, ivar)

            return wavelength, flux, ivar

    def parse_sdss_filename(filename):
        # Extract numbers using regex
        match = re.match(r'spec-(\d{4})-(\d{5})-(\d{4})\.fits', filename)
        if match:
            plate, mjd, fiber = map(int, match.groups())
            return plate, mjd, fiber
        return None, None, None


    catalogue = pd.read_csv('data/E-INSPIRE_I_master_catalogue.csv')
    mgfe = []
    vdisps = []
    ages = []
    metallicity = []
    DoRs = []

    for filename in spectra:
        wave, flux, ivar = load_spectrum("data/fits_shortlist/" + filename)  # data[0]
        plate, mjd, fiber = parse_sdss_filename(filename)

        matching_row = catalogue[(catalogue['plate'] == plate) &
                                 (catalogue['mjd'] == mjd) &
                                 (catalogue['fiberid'] == fiber)]

        mgfe.append(float(matching_row['MgFe'].iloc[0]))
        vd = matching_row["velDisp_ppxf_res"].iloc[0]
        vdisps.append(float(vd))
        ages.append(matching_row['age_mean_mass'])
        metallicity.append(matching_row['[M/H]_mean_mass'])
        DoRs.append(float(matching_row['DoR'].iloc[0]))


        data.append([wave, flux, ivar])


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

    mgfe_avg = round(np.mean(mgfe), 1)  # Rounds to nearest 0.1
    vd_avg = round(np.mean(vdisps), 1)  # Rounds to nearest 0.1

    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    c = 299792.458  # speed of light in km/s


    def calculate_sigma_diff(wave, sigma_gal, sigma_fin):
        # For log-binned spectra:
        ln_wave = np.log(wave)
        d_ln_wave = (ln_wave[-1] - ln_wave[0]) / (len(wave) - 1)  # log step size
        velscale = c * d_ln_wave  # Velocity scale in km/s per pixel

        wave_ref = np.mean(wave)  # Use mean wavelength as reference
        sigma_diff_kms = np.sqrt(sigma_fin ** 2 - sigma_gal ** 2)
        # sigma_ds.append(sigma_diff)

        sigma_gal_pxl = sigma_gal / velscale  # Convert km/s to pixels
        sigma_fin_pxl = sigma_fin / velscale

        sigma_diff = np.sqrt(sigma_fin_pxl ** 2 - sigma_gal_pxl ** 2)

        return sigma_diff, sigma_diff_kms


    def smooth_spectrum_to_sigma(wave, flux, sigma_gal, sigma_fin):
        if sigma_fin <= sigma_gal:
            return flux, sigma_gal

        sigma_diff, sigma_diff_kms = calculate_sigma_diff(wave, sigma_gal, sigma_fin)
        # Apply smoothing with single sigma_diff value
        smoothed_flux = gaussian_filter1d(flux, sigma_diff)
        # print(f"Original sigma: {sigma_gal:.2f} km/s, Smoothing kernel: {sigma_diff:.2f} pixels")
        return smoothed_flux, sigma_diff_kms


    def smooth(data, sigma_gals, sigma_fin):
        smoothed_data = []
        sigma_ds = []
        for (wave, flux, ivar), sigma_gal in zip(data, sigma_gals):
            smoothed_flux, sigma_diff_kms = smooth_spectrum_to_sigma(wave, flux, sigma_gal, sigma_fin)
            smoothed_data.append([wave, smoothed_flux, ivar])
            sigma_ds.append(sigma_diff_kms)

        return smoothed_data, sigma_ds



    sigma_fin = max(vdisps)
    smoothed, sigma_ds = smooth(data, vdisps, sigma_fin)

    # sigma_ds_average = round(np.mean(sigma_ds), 1)  # Rounds to nearest 0.1
    # print(sigma_ds_average)

    from scipy import interpolate

    # resample using logbinning
    def resample_spectrum(wave, flux, ivar, new_wave):
        # Interpolate flux onto new wavelength grid
        f = interpolate.interp1d(wave, flux, bounds_error=True, fill_value=np.nan)
        new_flux = f(new_wave)
        f_ivar = interpolate.interp1d(wave, ivar, bounds_error=True, fill_value=0)
        new_ivar = f_ivar(new_wave)

        return new_flux, new_ivar


    # Create common wavelength grid
    wave_min = max([smoothed[i][0][0] for i in range(len(smoothed))]) +0.1 # Maximum of all minimum wavelengths
    wave_max = min([smoothed[i][0][-1] for i in range(len(smoothed))]) -0.1 # Minimum of all maximum wavelengths
    num_points = min([len(s[0]) for s in smoothed])
    print("NUM POINTS::::",num_points)
    wave_common = np.logspace(np.log10(wave_min), np.log10(wave_max), num=3828)

    # Resample all spectra
    resampled_data = []
    for spectrum in smoothed:
        new_flux, new_ivar = resample_spectrum(spectrum[0], spectrum[1], spectrum[2], wave_common)
        resampled_data.append([wave_common, new_flux, new_ivar])

    def safe_combine_ivar(*ivars):
        ivar_stack = np.stack(ivars)
        N = len(ivars)  # Number of spectra being combined

        # Only mask completely invalid points
        mask = np.all((ivar_stack == 0) | np.isinf(ivar_stack), axis=0)
        combined = np.zeros_like(ivars[0])

        # Combine valid values
        valid = ~mask
        if np.any(valid):
            # For each pixel, only use non-zero, finite values
            valid_mask = (ivar_stack != 0) & np.isfinite(ivar_stack)
            valid_count = np.sum(valid_mask, axis=0)

            # Sum of 1/ivar for valid points only
            sum_inv_ivar = np.sum(1.0 / np.where(valid_mask, ivar_stack, np.inf), axis=0)

            # Only combine where we have at least one valid measurement
            has_valid = valid_count > 0
            combined[has_valid] = valid_count[has_valid] ** 2 / sum_inv_ivar[has_valid]

        return combined


    def combine_spectra(aligned_spectra):
        # Extract components
        wavelength = aligned_spectra[0][0]  # All wavelengths should be the same
        fluxes = [spec[1] for spec in aligned_spectra]
        ivars = [spec[2] for spec in aligned_spectra]

        # Calculate mean flux
        combined_flux = np.mean(fluxes, axis=0)

        # Combine inverse variances
        combined_ivar = safe_combine_ivar(*ivars)
        noise = (1/np.sqrt(combined_ivar))
        # print("noise of stacked:",noise[0:5])

        snr = combined_flux / noise
        # print("Mean SNR and sum SNR of stacked", np.mean(snr), np.sum(snr))
        # print("Mean flux after stacking:", np.mean(combined_flux))





        error = np.sqrt(1 / combined_ivar)
        # error = safe_errors(combined_ivar)
        plt.figure(figsize=(10, 6))  # Make figure larger

        plt.errorbar(wavelength, combined_flux, yerr=error,
                     fmt='-', color='blue',  # Main line
                     ecolor='lightgray',  # Lighter error bars
                     alpha=0.3,  # Make error bars semi-transparent
                     capsize=3,  # Larger caps on error bars
                     label='Spectrum')

        # Add grid for easier reading
        plt.grid(True, alpha=0.3)

        # Add labels
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.title('Spectrum with Uncertainties')
        plt.legend()

        # Tight layout to prevent label clipping
        plt.tight_layout()
        # plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(wavelength, error)
        plt.xlabel('Wavelength')
        plt.ylabel('Error')
        plt.title('Error Values vs Wavelength')
        plt.grid(True, alpha=0.3)
        # plt.show()


        return wavelength, combined_flux, combined_ivar


    def safe_errors(stacked_ivar):
        errors = np.zeros_like(stacked_ivar)
        valid = (stacked_ivar > 0) & np.isfinite(stacked_ivar)
        errors[valid] = 1.0 / np.sqrt(stacked_ivar[valid])
        return errors


    wavelength, flux, combined_ivar = combine_spectra(resampled_data)

    def create_fits(wavelength, flux, combined_ivar, cluster_name, mgfe, sigma_fin):
        # Create structured array for spectral data
        coadd_data = np.zeros(len(wavelength), dtype=[
            ('flux', 'f8'),
            ('wave', 'f8'),
            ('ivar', 'f8'),
            ('wdisp', 'f8')
        ])

        coadd_data['flux'] = flux
        coadd_data['wave'] = wavelength
        coadd_data['ivar'] = combined_ivar
        # print("Combined Ivar", combined_ivar)

        coadd_data['wdisp'] = np.full_like(wavelength, 2.76/2.355)
        # print("Wdisp:", coadd_data['wdisp'])
        # print("top lambda:", coadd_data['wave'][:10])
        # print("bottom lambda:", coadd_data['wave'][-10:])
        # print("N_points:", len(coadd_data['wave']))
        # print("flux:",flux)

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

    create_fits(wavelength, flux, combined_ivar, cluster_name, mgfe_avg, sigma_fin)

if __name__ == '__main__':
    if not os.path.exists('data/stacked_fits'):
        os.makedirs('data/stacked_fits')

    files = ["data/cluster_results/k-means_clusters.csv", "data/cluster_results/gmm_clusters.csv",
             "data/cluster_results/hierarchical_clusters.csv"]

    hierarchical = pd.read_csv(files[2])
    kmeans = pd.read_csv(files[0])
    gmm = pd.read_csv(files[1])
    catalogue = pd.read_csv("data/E-INSPIRE_I_master_catalogue.csv")

    # Create file lists for DoR ranges
    dor_clusters = []
    for threshold in [(0.6, float('inf')), (0.3, 0.6), (0, 0.3)]:
        dor_group = catalogue[(catalogue['DoR'] > threshold[0]) & (catalogue['DoR'] <= threshold[1])]
        file_list = [f"spec-{int(plate):04d}-{int(mjd):05d}-{int(fiber):04d}.fits"
                     for plate, mjd, fiber in zip(dor_group['plate'], dor_group['mjd'], dor_group['fiberid'])]
        dor_clusters.append(file_list)

    cluster_groups = {
        'DoR': dor_clusters,
        'Hierarchical': [hierarchical[hierarchical["Cluster"] == i]["SDSS_ID"].tolist() for i in range(3)],
        'KMeans': [kmeans[kmeans["Cluster"] == i]["SDSS_ID"].tolist() for i in range(3)],
        'GMM': [gmm[gmm["Cluster"] == i]["SDSS_ID"].tolist() for i in range(max(gmm["Cluster"]) + 1)]
        # 'GMM': [gmm[gmm["Cluster"] == i]["SDSS_ID"].tolist() for i in range(3)]
    }

    colors = ['red', 'blue', 'green']
    n = 0.001635
    factor = [n * 4, n * 2.7, n * 2.5]

    for method, groups in cluster_groups.items():
        print(f"\n================================")
        # print(f"Processing {method} clustering")
        method_labels = [f"{method}_{i}" for i in range(len(groups))]
        method_colors = colors[:len(groups)]
        method_factors = factor[:len(groups)]

        for idx, (spectra, color, label, factor_val) in enumerate(
                zip(groups, method_colors, method_labels, method_factors)):
            print(f"\nStacking {label}")
            stack_spectra(spectra, factor_val, label)
        print(f"================================")